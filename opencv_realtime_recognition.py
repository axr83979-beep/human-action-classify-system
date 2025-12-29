import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import time
from pathlib import Path

class OpenCVRealtimeRecognition:
    """OpenCV实时识别"""
    
    def __init__(self):
        """初始化"""
        self.model = None
        self.label_encoder = None
        self.class_names = []
        
        self.image_size = (224, 224)
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # 加载模型
        if self.load_model():
            print("✅ Model loaded successfully!")
        else:
            print("❌ Failed to load model. Please train first:")
            print("   python advanced_pose_recognition.py --train")
            exit(1)
    
    def load_model(self):
        """加载模型"""
        try:
            # 检查文件是否存在
            if not Path('efficientnetv2_final_model.h5').exists():
                print("Model file not found!")
                return False
            
            if not Path('efficientnetv2_label_encoder.pkl').exists():
                print("Label encoder not found!")
                return False
            
            # 加载模型
            self.model = keras.models.load_model('efficientnetv2_final_model.h5')
            
            # 加载标签编码器
            with open('efficientnetv2_label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            # 获取类别名称
            self.class_names = self.label_encoder.classes_
            
            print(f"Loaded {len(self.class_names)} action classes")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def preprocess_image(self, image):
        """预处理图像"""
        resized = cv2.resize(image, self.image_size)
        normalized = resized.astype(np.float32) / 255.0
        return normalized
    
    def predict(self, image):
        """预测"""
        try:
            # 预处理
            processed = self.preprocess_image(image)
            input_data = np.expand_dims(processed, axis=0)
            
            # 预测
            predictions = self.model.predict(input_data, verbose=0)
            
            # 获取Top 3
            top_3_idx = np.argsort(predictions[0])[-3:][::-1]
            
            results = []
            for idx in top_3_idx:
                class_name = self.label_encoder.inverse_transform([idx])[0]
                confidence = predictions[0][idx]
                results.append((class_name, confidence))
            
            return results
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return []
    
    def draw_interface(self, image, results, inference_time):
        """绘制界面"""
        h, w = image.shape[:2]
        
        # 侧边栏背景
        overlay = image.copy()
        cv2.rectangle(overlay, (w-300, 0), (w, h), (0, 0, 0), -1)
        image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
        
        # 标题
        y_pos = 40
        cv2.putText(image, "PREDICTION", (w-280, y_pos), 
                   cv2.FONT_HERSHEY_BOLD, 1.2, (0, 255, 255), 3)
        cv2.putText(image, "PREDICTION", (w-280, y_pos), 
                   cv2.FONT_HERSHEY_BOLD, 1.2, (0, 100, 100), 1)
        y_pos += 50
        
        # Top 3结果
        colors = [(255, 215, 0), (192, 192, 192), (205, 127, 50)]  # 金、银、铜
        
        for i, (action, conf) in enumerate(results):
            # 置信度条
            bar_width = int(conf * 200)
            cv2.rectangle(image, (w-280, y_pos), (w-280 + bar_width, y_pos + 25), 
                         colors[i], -1)
            cv2.rectangle(image, (w-280, y_pos), (w-80, y_pos + 25), 
                         (255, 255, 255), 2)
            
            # 文本
            percentage = conf * 100
            cv2.putText(image, f"{i+1}. {action}", (w-280, y_pos - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(image, f"{percentage:.1f}%", (w-270, y_pos + 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            y_pos += 60
        
        # 性能信息
        y_pos += 20
        cv2.putText(image, f"FPS: {self.fps:.1f}", (w-280, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30
        
        cv2.putText(image, f"Time: {inference_time*1000:.0f}ms", (w-280, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_pos += 30
        
        # 控制提示
        cv2.putText(image, "Q: Quit | S: Save", (w-280, h-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return image
    
    def run(self, camera_index=0):
        """运行实时识别"""
        # 打开摄像头
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("❌ Cannot open camera!")
            return
        
        # 设置分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n" + "="*50)
        print(" OpenCV Real-time Pose Recognition")
        print("="*50)
        print("\nControls:")
        print("  Q - Quit")
        print("  S - Save current frame")
        print("\nStarting camera...")
        print("="*50 + "\n")
        
        save_count = 0
        
        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # 镜像翻转
            frame = cv2.flip(frame, 1)
            
            # 计算FPS
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed > 1:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.start_time = time.time()
            
            # 预测
            start = time.time()
            results = self.predict(frame)
            inference_time = time.time() - start
            
            # 绘制界面
            frame = self.draw_interface(frame, results, inference_time)
            
            # 显示
            cv2.imshow('Advanced Pose Recognition', frame)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # 退出
                print("\nExiting...")
                break
            elif key == ord('s'):  # 保存
                save_count += 1
                filename = f'pose_capture_{save_count}.jpg'
                cv2.imwrite(filename, frame)
                print(f"✅ Saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Goodbye!")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='OpenCV Real-time Pose Recognition')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    
    args = parser.parse_args()
    
    system = OpenCVRealtimeRecognition()
    system.run(camera_index=args.camera)

if __name__ == "__main__":
    main()
