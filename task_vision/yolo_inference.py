#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from rclpy.qos import qos_profile_sensor_data
from hobot_dnn import pyeasy_dnn as dnn
import time
from rclpy.executors import ExternalShutdownException


class YoloInferenceNode(Node):
    def __init__(self):
        super().__init__('yolo_inference_node')
        self.bridge = CvBridge()

        # 1. 加载模型
        self.model_path = '/root/xeq/yolov5.bin'
        self.get_logger().info(f'正在加载 BPU 模型: {self.model_path}')
        self.models = dnn.load(self.model_path)
        self.model = self.models[0]
        self.get_logger().info('BPU 模型加载成功！')

        # 2. 图像订阅
        self.subscription = self.create_subscription(
            Image,
            '/aurora/rgb/image_raw',
            self.image_callback,
            qos_profile_sensor_data)

        # 3. 发布者
        self.drive_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.img_pub = self.create_publisher(Image, '/yolo_result_image', 10)

        # ===============================================
        # ⚙️ 核心调参区
        # ===============================================
        self.base_speed = 0.25  # 直道最高速度 (m/s)
        self.image_center_x = 320

        # --- PD 控制与阿克曼平滑 ---
        self.kp = 0.002  # 转向比例
        self.kd = 0.0015  # 转向阻尼(防画龙)
        self.max_steer = 0.6  # 最大转向角速度
        self.steer_alpha = 0.4  # 舵机平滑系数(0~1)

        # --- 避障与停车阈值 (基于平滑后的底边 Y_max 坐标) ---
        self.OBSTACLE_Y_THRES = 600  # 💡 锥桶底边达到 360 时触发避障
        self.QRCODE_Y_THRES = 620  # 💡 二维码底边达到 400 时触发停车
        # ===============================================

        self.CLASS_QR_code = 0
        self.CLASS_line = 1
        self.CLASS_end = 2
        self.CLASS_roadblock = 3
        self.NUM_CLASSES = 4

        self.state = "TRACKING"
        self.avoid_start_time = 0.0
        self.avoid_direction = 1
        self.frame_count = 0
        self.last_time = time.time()

        # 用于 PD 控制和滤波的历史记录
        self.last_error_x = 0.0
        self.last_angular_z = 0.0

        # 💡 新增：用于目标距离(Y坐标)防抖平滑的历史变量
        self.smoothed_obs_y = 0.0
        self.smoothed_qr_y = 0.0

        self.label_map = ["QR_code", "Line", "End", "Roadblock"]
        self.color_map = {
            0: (0, 0, 255),
            1: (0, 255, 0),
            2: (255, 0, 0),
            3: (0, 255, 255)
        }

    def stop_car(self):
        """安全停车"""
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.angular.z = 0.0
        self.drive_pub.publish(stop_msg)
        self.last_angular_z = 0.0
        self.get_logger().info('🛑 已向底盘发送停车指令！')

    def bgr2nv12(self, bgr_image):
        height, width = bgr_image.shape[:2]
        yuv420p = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YUV_I420)
        y = yuv420p[:height, :]
        u = yuv420p[height: height + height // 4, :].reshape(height // 2, width // 2)
        v = yuv420p[height + height // 4:, :].reshape(height // 2, width // 2)
        nv12 = np.empty((height * 3 // 2, width), dtype=np.uint8)
        nv12[:height, :] = y
        nv12[height:, 0::2] = u
        nv12[height:, 1::2] = v
        return nv12

    def decode_yolo_outputs(self, outputs, conf_thres=0.3, iou_thres=0.45):
        pred = np.squeeze(outputs[0].buffer)
        if pred.shape[0] < pred.shape[1]:
            pred = pred.transpose()

        obj_conf = pred[:, 4]
        valid_indices = obj_conf > conf_thres
        valid_pred = pred[valid_indices]

        if len(valid_pred) == 0:
            return []

        cls_probs = valid_pred[:, 5:]
        class_ids = np.argmax(cls_probs, axis=1)
        scores = valid_pred[:, 4] * cls_probs[np.arange(len(valid_pred)), class_ids]

        score_indices = scores > conf_thres
        valid_pred = valid_pred[score_indices]
        scores = scores[score_indices]
        class_ids = class_ids[score_indices]

        if len(valid_pred) == 0:
            return []

        cx = valid_pred[:, 0]
        cy = valid_pred[:, 1]
        w = valid_pred[:, 2]
        h = valid_pred[:, 3]

        x_min = cx - w / 2
        y_min = cy - h / 2

        boxes = np.column_stack((x_min, y_min, w, h)).astype(int).tolist()
        scores = scores.astype(float).tolist()
        class_ids = class_ids.astype(int).tolist()

        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)

        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                results.append([box[0], box[1], box[0] + box[2], box[1] + box[3], scores[i], class_ids[i]])
        return results

    def image_callback(self, msg):
        try:
            current_time = time.time()
            fps = 1.0 / (current_time - self.last_time) if (current_time - self.last_time) > 0 else 0
            self.last_time = current_time

            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv_image = cv2.flip(cv_image, -1)

            resized_image = cv2.resize(cv_image, (640, 640))
            nv12_image = np.ascontiguousarray(self.bgr2nv12(resized_image))

            outputs = self.model.forward(nv12_image)
            bboxes = self.decode_yolo_outputs(outputs, conf_thres=0.3)

            drive_msg = Twist()
            target_line_box = None
            closest_obstacle = None
            closest_qrcode = None

            for box in bboxes:
                x_min, y_min, x_max, y_max, score, class_id = box

                # 💡 取底边Y_max作为距离判定标准(比中心点稳得多)
                box_y_max = y_max

                color = self.color_map.get(class_id, (255, 255, 255))
                cv2.rectangle(resized_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
                cv2.circle(resized_image, (int((x_min + x_max) / 2), int(box_y_max)), 5, color, -1)  # 画底边中心点

                label_text = f"{self.label_map[class_id]} {score:.2f} Y:{int(box_y_max)}"
                cv2.putText(resized_image, label_text, (int(x_min), int(y_min) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 2)

                if class_id == self.CLASS_QR_code or class_id == self.CLASS_end:
                    if closest_qrcode is None or box_y_max > closest_qrcode[3]:
                        closest_qrcode = box
                elif class_id == self.CLASS_roadblock:
                    if closest_obstacle is None or box_y_max > closest_obstacle[3]:
                        closest_obstacle = box
                elif class_id == self.CLASS_line:
                    if target_line_box is None or box_y_max > target_line_box[3]:
                        target_line_box = box

            # ====== 💡 数据平滑处理 (EMA 滤波) ======
            current_obs_y = 0.0
            if closest_obstacle is not None:
                raw_obs_y = closest_obstacle[3]  # 获取真实的底边Y坐标
                if self.smoothed_obs_y == 0.0:
                    self.smoothed_obs_y = raw_obs_y
                else:
                    # 70%信任历史，30%信任新数据，消除抖动
                    self.smoothed_obs_y = 0.7 * self.smoothed_obs_y + 0.3 * raw_obs_y
                current_obs_y = self.smoothed_obs_y
            else:
                self.smoothed_obs_y = 0.0  # 视野中消失则重置

            current_qr_y = 0.0
            if closest_qrcode is not None:
                raw_qr_y = closest_qrcode[3]
                if self.smoothed_qr_y == 0.0:
                    self.smoothed_qr_y = raw_qr_y
                else:
                    self.smoothed_qr_y = 0.7 * self.smoothed_qr_y + 0.3 * raw_qr_y
                current_qr_y = self.smoothed_qr_y
            else:
                self.smoothed_qr_y = 0.0

            # ====== 1. 停车逻辑 ======
            if self.state != "STOP" and closest_qrcode is not None:
                if current_qr_y >= self.QRCODE_Y_THRES:
                    self.state = "STOP"
                    self.get_logger().info(f'🛑 二维码底边到达 {int(current_qr_y)}，执行停车！')

            if self.state == "STOP":
                drive_msg.linear.x = 0.0
                drive_msg.angular.z = 0.0
                self.drive_pub.publish(drive_msg)
                self.last_angular_z = 0.0
            else:
                # ====== 2. 避障触发逻辑 ======
                if self.state == "TRACKING" and closest_obstacle is not None:
                    if current_obs_y >= self.OBSTACLE_Y_THRES:
                        self.state = "AVOID_TURN_OUT"
                        self.avoid_start_time = current_time
                        obs_center_x = (closest_obstacle[0] + closest_obstacle[2]) / 2
                        self.avoid_direction = 1 if obs_center_x > self.image_center_x else -1
                        self.get_logger().info(f'⚠️ 锥桶底边达到 {int(current_obs_y)}，平滑触发避障！')

                # ====== 3. 避障执行状态机 ======
                if self.state.startswith("AVOID"):
                    time_elapsed = current_time - self.avoid_start_time
                    target_angular_z = 0.0
                    target_linear_x = self.base_speed * 0.8

                    if time_elapsed < 1.0:
                        self.state = "AVOID_TURN_OUT"
                        target_angular_z = 0.8 * self.avoid_direction
                    elif time_elapsed < 2.0:
                        self.state = "AVOID_GO_STRAIGHT"
                        target_angular_z = 0.0
                    elif time_elapsed < 3.2:
                        self.state = "AVOID_TURN_IN"
                        target_angular_z = -0.8 * self.avoid_direction
                    else:
                        self.state = "TRACKING"

                    if self.state == "AVOID_TURN_IN" and target_line_box is not None:
                        self.state = "TRACKING"

                    drive_msg.linear.x = target_linear_x
                    drive_msg.angular.z = target_angular_z
                    self.last_angular_z = target_angular_z

                # ====== 4. 阿克曼 PD 巡线控制 ======
                if self.state == "TRACKING":
                    if target_line_box is not None:
                        x_min, y_min, x_max, y_max, _, _ = target_line_box
                        line_center_x = (x_min + x_max) / 2

                        error_x = self.image_center_x - line_center_x
                        error_diff = error_x - self.last_error_x
                        self.last_error_x = error_x

                        raw_steer = (self.kp * error_x) + (self.kd * error_diff)
                        target_steer = max(min(raw_steer, self.max_steer), -self.max_steer)

                        smooth_steer = (self.steer_alpha * target_steer) + (
                                    (1.0 - self.steer_alpha) * self.last_angular_z)
                        self.last_angular_z = smooth_steer

                        speed_penalty = 1.0 - 0.4 * (abs(smooth_steer) / self.max_steer)
                        current_speed = self.base_speed * speed_penalty

                        drive_msg.linear.x = current_speed
                        drive_msg.angular.z = smooth_steer

                        cv2.line(resized_image, (self.image_center_x, 640),
                                 (int(line_center_x), int((y_min + y_max) / 2)), (0, 0, 255), 3)
                    else:
                        drive_msg.linear.x = self.base_speed * 0.5
                        drive_msg.angular.z = self.last_angular_z * 0.5
                        self.last_angular_z = drive_msg.angular.z

                self.drive_pub.publish(drive_msg)

            # ====== 日志打印 ======
            self.frame_count += 1
            if self.frame_count % 15 == 0:
                self.get_logger().info(
                    f'状态: {self.state} | FPS: {fps:.1f} | 平滑锥桶Y: {int(current_obs_y)} | '
                    f'平滑码Y: {int(current_qr_y)} | 线速: {drive_msg.linear.x:.2f} | 角速: {drive_msg.angular.z:.2f}'
                )

            # ====== 图像发布 ======
            try:
                cv2.putText(resized_image, f"FPS: {fps:.1f} Steer: {drive_msg.angular.z:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.line(resized_image, (self.image_center_x, 0), (self.image_center_x, 640), (0, 255, 0), 1)

                # 动态显示当前的判定阈值线，助你调参
                if closest_obstacle is not None:
                    cv2.line(resized_image, (0, self.OBSTACLE_Y_THRES), (640, self.OBSTACLE_Y_THRES), (0, 255, 255), 1)
                if closest_qrcode is not None:
                    cv2.line(resized_image, (0, self.QRCODE_Y_THRES), (640, self.QRCODE_Y_THRES), (0, 0, 255), 1)

                result_msg = self.bridge.cv2_to_imgmsg(resized_image, "bgr8")
                result_msg.header = msg.header
                self.img_pub.publish(result_msg)
            except Exception:
                pass

        except Exception as e:
            self.get_logger().error(f'处理图像时发生异常: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = YoloInferenceNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        node.get_logger().info('检测到 Ctrl+C，正在退出程序并停车...')
    finally:
        if rclpy.ok():
            node.stop_car()
            time.sleep(0.1)
            node.destroy_node()
            rclpy.try_shutdown()


if __name__ == '__main__':
    main()