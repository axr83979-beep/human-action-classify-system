# 🔧 安装配置指南

## 系统要求

### 最低配置
- **操作系统**: Windows 10/11, macOS 10.14+, Linux
- **Python**: 3.8 或更高版本
- **内存**: 8GB RAM
- **磁盘空间**: 5GB 可用空间
- **网络**: 需要下载依赖包

### 推荐配置
- **操作系统**: Windows 10/11
- **Python**: 3.9 或 3.10
- **内存**: 16GB RAM
- **GPU**: NVIDIA GPU with CUDA (可选，加速训练)
- **磁盘空间**: 10GB 可用空间

---

## 📦 步骤 1: 安装 Python

### Windows
1. 访问 [python.org](https://www.python.org/downloads/)
2. 下载 Python 3.9 或 3.10
3. 运行安装程序
4. **重要**: 勾选 "Add Python to PATH"
5. 点击 "Install Now"

验证安装:
```bash
python --version
pip --version
```

### macOS
```bash
# 使用 Homebrew
brew install python@3.10
```

或访问 [python.org](https://www.python.org/downloads/) 下载安装包

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.10 python3-pip
```

---

## 📦 步骤 2: 安装依赖

### 方式 A: 使用 requirements.txt (推荐)

```bash
pip install -r requirements.txt
```

### 方式 B: 单独安装 (如果遇到问题)

```bash
# 核心深度学习框架
pip install tensorflow==2.12.0

# 计算机视觉
pip install opencv-python==4.8.0
pip install mediapipe==0.10.0

# UI框架
pip install gradio==4.0.0

# 数据处理
pip install numpy==1.24.0
pip install pandas==2.0.0
pip install scikit-learn==1.3.0

# 工具
pip install tqdm==4.65.0
pip install Pillow==10.0.0
```

### 使用国内镜像源 (中国用户)

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

其他镜像源:
- 阿里云: `https://mirrors.aliyun.com/pypi/simple/`
- 中国科技大学: `https://pypi.mirrors.ustc.edu.cn/simple/`
- 豆瓣: `https://pypi.douban.com/simple/`

---

## 📦 步骤 3: 验证安装

### 运行状态检查脚本

```bash
python check_status.py
```

预期输出:
```
============================================================
 Python Version Check
============================================================

Python 3.10.12
✅ Python version OK (3.8+ required)

============================================================
 Dependencies Check
============================================================

✅ tensorflow     2.12.0
✅ cv2            4.8.0
✅ gradio         4.0.0
✅ pandas         2.0.0
✅ numpy          1.24.0
✅ sklearn        1.3.0
✅ PIL            10.0.0

============================================================
 SUMMARY
============================================================

✅ Python Version
✅ Dependencies
✅ All checks passed!
```

### 手动验证

```python
# 测试 Python 版本
python -c "import sys; print(f'Python {sys.version}')"

# 测试 TensorFlow
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"

# 测试 OpenCV
python -c "import cv2; print(f'OpenCV {cv2.__version__}')"

# 测试 Gradio
python -c "import gradio as gr; print(f'Gradio {gr.__version__}')"
```

---

## 📦 步骤 4: 可选 - GPU 支持 (NVIDIA)

### 安装 CUDA Toolkit

1. 访问 [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. 下载适合你系统的版本
3. 安装 CUDA Toolkit 11.8 (推荐)
4. 添加 CUDA 到系统 PATH

### 安装 cuDNN

1. 访问 [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
2. 注册并下载 cuDNN 8.x
3. 解压到 CUDA 安装目录

### 安装 GPU 版本的 TensorFlow

```bash
pip uninstall tensorflow
pip install tensorflow[and-cuda]==2.12.0
```

### 验证 GPU 支持

```python
import tensorflow as tf

print("TensorFlow Version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# 测试 GPU
if tf.config.list_physical_devices('GPU'):
    print("✅ GPU available!")
else:
    print("⚠️  GPU not found, will use CPU")
```

---

## 📦 步骤 5: 数据集准备

### 自动下载 (如果数据集已包含)

检查数据集是否存在:
```bash
python verify_data.py
```

预期输出:
```
✅ Training samples: 12600
✅ Number of classes: 15
✅ No issues found!
```

### 手动准备 (如果数据集缺失)

1. 下载 Human Action Recognition 数据集
2. 解压到 `datasets/Human Action Recognition/` 目录
3. 确保包含以下文件:
   - `Training_set.csv`
   - `Testing_set.csv`
   - `train/` 目录
   - `test/` 目录

数据集目录结构:
```
datasets/
└── Human Action Recognition/
    ├── Training_set.csv
    ├── Testing_set.csv
    ├── train/
    │   ├── Image_1.jpg
    │   ├── Image_2.jpg
    │   └── ...
    └── test/
        ├── Image_10001.jpg
        ├── Image_10002.jpg
        └── ...
```

---

## 📦 步骤 6: 训练模型

### Windows 用户

双击运行:
```
train_model.bat
```

### Mac/Linux 用户

```bash
python advanced_pose_recognition.py --train
```

### 训练时间

- **CPU**: 2-4 小时
- **GPU**: 30-60 分钟

### 训练监控

训练过程中会显示:
```
Epoch 1/20
 394/394 [==============================] - 45s 112ms/step - loss: 1.2345 - accuracy: 0.6543 - val_loss: 0.9876 - val_accuracy: 0.7123

Epoch 2/20
 394/394 [==============================] - 40s 102ms/step - loss: 0.8765 - accuracy: 0.7654 - val_loss: 0.7654 - val_accuracy: 0.7890
...
```

训练完成后会生成:
- `efficientnetv2_final_model.h5` (~80MB)
- `efficientnetv2_label_encoder.pkl`
- `efficientnetv2_model_info.pkl`

---

## 📦 步骤 7: 启动系统

### 方式 A: 使用启动菜单 (推荐)

#### Windows

双击运行:
```
启动系统.bat
```

#### Mac/Linux

```bash
python launcher.py
```

然后选择:
- [1] 启动 Gradio Web 界面
- [2] 启动 OpenCV 实时识别

### 方式 B: 直接启动

#### Gradio Web 界面

```bash
python advanced_pose_recognition.py --run
```

浏览器会自动打开 http://localhost:7860

#### OpenCV 实时识别

```bash
python opencv_realtime_recognition.py
```

---

## 🔧 故障排除

### 问题 1: Python 找不到

**错误**: `'python' is not recognized`

**解决**:
1. 重新安装 Python
2. 勾选 "Add Python to PATH"
3. 或手动添加 Python 到 PATH:
   - 找到 Python 安装目录 (如 `C:\Python39`)
   - 添加到系统 PATH

### 问题 2: pip 安装失败

**错误**: `Could not find a version that satisfies the requirement`

**解决**:
```bash
# 升级 pip
python -m pip install --upgrade pip

# 使用镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题 3: TensorFlow 安装失败

**错误**: `Failed to build wheel for tensorflow`

**解决**:
```bash
# 使用预编译版本
pip install tensorflow==2.12.0

# Windows 用户，确保安装了 Visual C++ Redistributable
# https://aka.ms/vs/17/release/vc_redist.x64.exe
```

### 问题 4: CUDA 错误

**错误**: `CUDA out of memory` 或 `CUDA not found`

**解决**:
1. 检查 CUDA 安装: `nvcc --version`
2. 检查 TensorFlow GPU 支持:
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```
3. 减小批次大小: 修改 `advanced_pose_recognition.py`:
   ```python
   self.batch_size = 16  # 或 8
   ```

### 问题 5: 内存不足

**错误**: `MemoryError` 或系统卡顿

**解决**:
1. 减小批次大小
2. 使用更小的模型
3. 增加虚拟内存
4. 关闭其他程序

### 问题 6: 摄像头无法打开

**错误**: `Cannot open camera`

**解决**:
1. 检查摄像头权限
2. 尝试不同的摄像头索引:
   ```bash
   python opencv_realtime_recognition.py --camera 1
   ```
3. 检查摄像头是否被其他程序占用

### 问题 7: Gradio 无法启动

**错误**: `Gradio not found` 或无法访问界面

**解决**:
```bash
# 更新 Gradio
pip install --upgrade gradio

# 指定端口
python advanced_pose_recognition.py --run

# 手动访问 http://localhost:7860
```

---

## 📋 安装检查清单

使用此清单确保一切就绪:

- [ ] Python 3.8+ 已安装
- [ ] pip 可用
- [ ] 所有依赖包已安装
- [ ] `check_status.py` 显示全部通过
- [ ] 数据集已准备
- [ ] `verify_data.py` 显示数据集完整
- [ ] 模型已训练
- [ ] 可以启动系统
- [ ] 摄像头可以正常使用

---

## 📞 获取帮助

如果遇到问题:

1. 查看 `README.md` 了解基本使用
2. 查看 `QUICKSTART.md` 快速开始
3. 查看 `PROJECT_SUMMARY.md` 了解项目详情
4. 运行 `check_status.py` 检查系统状态
5. 运行 `verify_data.py` 验证数据集

---

## 🎉 安装完成!

现在你可以:
- 使用 `launcher.py` 打启动菜单
- 运行 `advanced_pose_recognition.py --run` 启动 Gradio
- 运行 `opencv_realtime_recognition.py` 使用 OpenCV

**享受高精度人体姿态识别！** 🚀
