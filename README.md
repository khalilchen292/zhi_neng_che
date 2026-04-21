# zhi_neng_che

208 的底盘驱动和视觉部分。

## 技术路线与模块分工（任务流程记忆）

系统按任务状态机运行，核心流程如下：

1. **A 区（右引导线）**：沿右侧引导线行驶，结合视觉避障，识别二维码（可在过半场后提前扫码）。
2. **黄通道**：基于深度信息做通道居中，同时避让锥桶。
3. **marker 计圈**：根据 marker/标记牌做圈数统计与任务阶段切换。
4. **回程（左引导线）**：驶离通道后沿左侧引导线返回 P 点停车。

禁行语义包含 `no_go_white_cross`（白色线交叉禁行区）。

## 模型可替换设计（YOLO/检测器）

为避免更换模型后硬编码崩溃，模型与类别映射集中在配置文件管理：

- 主配置：`/config/vision_model.yaml`
- 推理节点：`/vision/inference_node.py`
- 自检脚本：`/scripts/vision_self_check.py`

### 更换模型时需要修改

1. `model.weights_path`：模型权重路径（如新的 `.onnx`/引擎文件）。
2. `model.version`：当前模型版本标识（用于日志与复现记录）。
3. `class_name_to_semantic`：把模型类别名映射到内部语义（`cone`、`qrcode`、`marker`、`no_go_white_cross`）。

### 启动参数（保持默认兼容）

推理节点支持：

- `--config`：主配置文件路径（默认 `config/vision_model.yaml`）
- `--model-path`：覆盖模型路径（可选）
- `--mapping-file`：覆盖类别映射文件（可选）

示例：

```bash
python3 vision/inference_node.py
python3 vision/inference_node.py --model-path /abs/path/new_model.onnx
python3 vision/inference_node.py --mapping-file /abs/path/new_mapping.yaml
```

### 启动前最小自检

```bash
python3 scripts/vision_self_check.py
```

自检会验证：

- 模型文件是否存在
- 语义映射是否完整（`cone/qrcode/marker/no_go_white_cross`）
- 当前加载模型版本信息
