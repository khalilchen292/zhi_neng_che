# zhi_neng_che
208的底盘驱动和视觉部分

## 访问与路径核验（main）

- 已验证可访问仓库 `khalilchen292/zhi_neng_che` 的 `main` 分支。
- 已在仓库内搜索以下目标文件及等价路径：
  - `launch/star_all.launch.py`
  - `task_vision/yolo_inference.py`
- 搜索结果：`main` 分支当前仅包含本 README，未找到上述文件或等价视觉启动/推理脚本路径。

## 视觉模块运行与模型权重配置说明

当前 `main` 分支未包含视觉代码，因此暂无法在本仓库直接执行视觉启动流程。  
当后续同步视觉代码后，可按以下约定使用：

1. 运行视觉启动文件（示例命令，实际以落库后的包名/文件路径为准）：`ros2 launch <package_name> star_all.launch.py`
2. 配置/更换模型权重：
   - 通常可在 `yolo_inference.py` 中将权重路径配置项改为新的权重文件路径；或
   - 若 `star_all.launch.py` 提供权重参数，则通常优先通过 launch 参数传入权重路径（参数名以实际实现为准，如 `model_path` / `weights_file`）。

若视觉代码位于其他仓库（如 `task_vision`/`task_vision_pure`），请以该仓库中的实际路径为准。
