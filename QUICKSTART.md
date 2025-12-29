# 🚀 快速开始指南

## 第一次使用？按这个顺序来！

### 步骤 1: 安装依赖 (5分钟)

```bash
pip install -r requirements.txt
```

如果遇到问题，可以分别安装：
```bash
pip install tensorflow
pip install opencv-python
pip install gradio
pip install pandas numpy scikit-learn
```

---

### 步骤 2: 验证数据集 (1分钟)

```bash
python verify_data.py
```

确保显示：
- ✅ Training samples: 12600
- ✅ Number of classes: 15
- ✅ No issues found!

---

### 步骤 3: 训练模型 (30-60分钟)

**Windows**: 双击 `train_model.bat`

**Mac/Linux**: 运行
```bash
python advanced_pose_recognition.py --train
```

训练完成后，会看到：
- ✅ Training completed!
- Final validation accuracy: 0.9500+

---

### 步骤 4: 启动系统 (1分钟)

**选项A: 使用启动菜单 (推荐)**

**Windows**: 双击 `启动系统.bat`

**Mac/Linux**: 运行
```bash
python launcher.py
```

选择:
- [1] 启动 Gradio Web 界面
- [2] 启动 OpenCV 实时识别

---

**选项B: 直接启动**

**Gradio Web界面** (推荐新手):
```bash
python advanced_pose_recognition.py --run
```
或双击 `start_gradio.bat`

**OpenCV实时识别** (适合实时使用):
```bash
python opencv_realtime_recognition.py
```
或双击 `start_opencv.bat`

---

## 📸 使用示例

### Gradio 界面

1. 浏览器会自动打开 http://localhost:7860
2. 点击 "Image Recognition" 标签页
3. 上传一张图片
4. 点击 "Predict" 按钮
5. 查看识别结果！

### 实时摄像头

1. 点击 "Real-time Webcam" 标签页
2. 允许浏览器访问摄像头
3. 站在摄像头前
4. 看到实时识别结果！

---

## 🎯 支持的动作

| 中文 | English |
|------|---------|
| 坐着 | sitting |
| 用电脑 | using_laptop |
| 拥抱 | hugging |
| 睡觉 | sleeping |
| 喝水 | drinking |
| 鼓掌 | clapping |
| 跳舞 | dancing |
| 骑车 | cycling |
| 打电话 | calling |
| 大笑 | laughing |
| 吃饭 | eating |
| 打架 | fighting |
| 听音乐 | listening_to_music |

---

## 💡 使用技巧

### 获得最佳识别效果

1. **光线充足**
   - 避免背光
   - 保持环境明亮

2. **姿势清晰**
   - 人物完整可见
   - 动作明确
   - 正面或侧面视角

3. **距离适中**
   - 离摄像头 1-2 米
   - 人物占据画面 50%+

### 实时识别时

- 保持动作 2-3 秒
- 系统会自动稳定
- 置信度 > 80% 才是可靠结果

---

## 🔧 常见问题快速解决

### Q: pip install 失败

**A**: 使用清华镜像源:
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q: 内存不足

**A**: 修改 `advanced_pose_recognition.py`:
```python
self.batch_size = 16  # 或 8
```

### Q: 训练太慢

**A**: 
- 使用GPU (NVIDIA)
- 减少训练轮数
- 使用更小的模型

### Q: 识别不准确

**A**:
- 检查图像质量
- 确认动作在支持列表中
- 重新训练更多轮次

### Q: 打不开摄像头

**A**:
- 检查摄像头权限
- 尝试更换摄像头索引:
```bash
python opencv_realtime_recognition.py --camera 1
```

---

## 📂 文件说明

| 文件 | 说明 |
|------|------|
| `launcher.py` | 启动菜单 (推荐使用) |
| `advanced_pose_recognition.py` | 主程序 (Gradio) |
| `opencv_realtime_recognition.py` | OpenCV版本 |
| `verify_data.py` | 数据验证工具 |
| `train_model.bat` | 训练启动脚本 |
| `启动系统.bat` | Windows启动菜单 |

---

## 🎉 下一步

1. ✅ 完成安装
2. ✅ 训练模型
3. ✅ 启动系统
4. 🎊 享受高精度人体姿态识别！

---

## 📞 需要帮助?

- 查看详细文档: `README.md`
- 查看高级文档: `README_ADVANCED.md`
- 验证数据: `python verify_data.py`

---

**开始使用吧！** 🚀
