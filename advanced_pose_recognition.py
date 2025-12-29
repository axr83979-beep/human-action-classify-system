import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gradio as gr
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from pathlib import Path
from tqdm import tqdm
import time

class AdvancedPoseRecognition:
    """高级人体姿态识别系统"""
    
    def __init__(self, data_path='datasets/Human Action Recognition'):
        """初始化系统"""
        self.data_path = data_path
        self.train_csv = os.path.join(data_path, 'Training_set.csv')
        self.test_csv = os.path.join(data_path, 'Testing_set.csv')
        self.train_image_dir = os.path.join(data_path, 'train')
        self.test_image_dir = os.path.join(data_path, 'test')
        
        self.model = None
        self.label_encoder = None
        self.class_names = []
        
        self.image_size = (224, 224)
        self.batch_size = 32
        self.epochs = 50
        
        print("Advanced Pose Recognition System Initialized")
    
    def load_data(self):
        """加载数据集"""
        print("Loading dataset...")
        
        # 读取训练数据
        train_df = pd.read_csv(self.train_csv)
        print(f"Training samples: {len(train_df)}")
        
        # 获取类别标签
        self.class_names = sorted(train_df['label'].unique())
        print(f"Number of classes: {len(self.class_names)}")
        print(f"Classes: {', '.join(self.class_names)}")
        
        # 创建标签编码器
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.class_names)
        
        return train_df
    
    def build_model(self):
        """构建基于EfficientNetV2的高级模型"""
        print("Building EfficientNetV2 model...")
        
        # 使用EfficientNetV2-S作为基础模型
        base_model = keras.applications.EfficientNetV2S(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3),
            pooling='avg'
        )
        
        # 冻结基础模型
        base_model.trainable = False
        
        # 构建完整模型
        inputs = keras.Input(shape=(224, 224, 3))
        
        # 数据增强
        x = layers.RandomFlip('horizontal')(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        x = layers.RandomContrast(0.1)(x)
        
        # 预处理
        x = keras.applications.efficientnet_v2.preprocess_input(x)
        
        # 基础模型
        x = base_model(x, training=False)
        
        # 自定义分类层
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # 输出层
        outputs = layers.Dense(len(self.class_names), activation='softmax')(x)
        
        # 创建模型
        model = keras.Model(inputs, outputs, name='AdvancedPoseRecognition')
        
        return model, base_model
    
    def train(self):
        """训练模型"""
        print("Starting training...")
        
        # 加载数据
        train_df = self.load_data()
        
        # 构建模型
        model, base_model = self.build_model()
        
        # 编译模型
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 准备数据
        images = []
        labels = []
        
        print("Loading images...")
        for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Processing images"):
            img_path = os.path.join(self.train_image_dir, row['filename'])
            
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, self.image_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    labels.append(row['label'])
        
        images = np.array(images)
        labels = np.array(self.label_encoder.transform(labels))
        
        print(f"Successfully loaded {len(images)} images")
        
        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # 回调函数
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                'efficientnetv2_best_model.h5',
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # 第一阶段训练（冻结基础模型）
        print("\n=== Phase 1: Training with frozen base model ===")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # 第二阶段训练（解冻基础模型）
        print("\n=== Phase 2: Fine-tuning with unfrozen base model ===")
        base_model.trainable = True
        
        # 重新编译（使用更小的学习率）
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 继续训练
        history_fine = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # 保存最终模型
        model.save('efficientnetv2_final_model.h5')
        
        # 保存标签编码器
        with open('efficientnetv2_label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # 保存模型信息
        model_info = {
            'model_name': 'EfficientNetV2',
            'num_classes': len(self.class_names),
            'image_size': self.image_size,
            'class_names': self.class_names.tolist(),
            'final_accuracy': history_fine.history['val_accuracy'][-1]
        }
        
        with open('efficientnetv2_model_info.pkl', 'wb') as f:
            pickle.dump(model_info, f)
        
        self.model = model
        print("\n✅ Training completed!")
        print(f"Final validation accuracy: {history_fine.history['val_accuracy'][-1]:.4f}")
    
    def load_trained_model(self):
        """加载已训练的模型"""
        print("Loading trained model...")
        
        try:
            # 加载模型
            self.model = keras.models.load_model('efficientnetv2_final_model.h5')
            
            # 加载标签编码器
            with open('efficientnetv2_label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            # 加载模型信息
            with open('efficientnetv2_model_info.pkl', 'rb') as f:
                model_info = pickle.load(f)
                self.class_names = model_info['class_names']
            
            print("✅ Model loaded successfully!")
            print(f"Number of classes: {len(self.class_names)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict_image(self, image):
        """预测单张图像"""
        if self.model is None:
            if not self.load_trained_model():
                return None, None
        
        # 预处理
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 调整大小
        image_resized = cv2.resize(image, self.image_size)
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # 预测
        input_data = np.expand_dims(image_normalized, axis=0)
        predictions = self.model.predict(input_data, verbose=0)
        
        # 获取Top预测
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        
        results = []
        for idx in top_3_idx:
            class_name = self.label_encoder.inverse_transform([idx])[0]
            confidence = float(predictions[0][idx])
            results.append({
                'class': class_name,
                'confidence': confidence
            })
        
        return results[0]['class'], results[0]['confidence'], results
    
    def create_gradio_interface(self):
        """创建Gradio界面"""
        # 确保模型已加载
        if self.model is None:
            if not self.load_trained_model():
                return None
        
        def predict_and_display(image):
            """预测并显示结果"""
            if image is None:
                return None, "Please upload an image", ""
            
            start_time = time.time()
            top_class, top_confidence, all_results = self.predict_image(image)
            inference_time = time.time() - start_time
            
            # 准备结果文本
            result_text = f"### 🔍 Prediction Results\n\n"
            result_text += f"**Action:** {top_class}\n\n"
            result_text += f"**Confidence:** {top_confidence*100:.2f}%\n\n"
            result_text += f"**Inference Time:** {inference_time*1000:.1f}ms\n\n"
            result_text += "---\n\n### Top 3 Predictions:\n\n"
            
            for i, result in enumerate(all_results, 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                result_text += f"{emoji} **{result['class']}**: {result['confidence']*100:.2f}%\n"
            
            return image, result_text, f"{top_class} ({top_confidence*100:.1f}%)"
        
        def predict_webcam(image):
            """实时摄像头预测"""
            if image is None:
                return image, "Waiting for webcam..."
            
            top_class, top_confidence, _ = self.predict_image(image)
            
            # 在图像上绘制结果
            image_with_text = image.copy()
            cv2.putText(image_with_text, f"Action: {top_class}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image_with_text, f"Conf: {top_confidence*100:.1f}%", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            result_text = f"**Action:** {top_class}\n\n**Confidence:** {top_confidence*100:.2f}%"
            
            return image_with_text, result_text
        
        # 创建界面
        with gr.Blocks(theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="sky",
        )) as demo:
            gr.Markdown(
                """
                # 🤸 Advanced Human Pose Recognition System
                
                ## Built with EfficientNetV2 + Gradio
                
                This system uses state-of-the-art deep learning to recognize 15 different human actions with high accuracy.
                
                ---
                """
            )
            
            with gr.Tabs():
                # 图像识别标签页
                with gr.Tab("📷 Image Recognition"):
                    with gr.Row():
                        with gr.Column():
                            image_input = gr.Image(label="Upload Image", type="numpy")
                            predict_btn = gr.Button("🔮 Predict", variant="primary", size="lg")
                        
                        with gr.Column():
                            output_image = gr.Image(label="Result")
                            output_text = gr.Markdown(label="Prediction Results")
                            output_label = gr.Textbox(label="Quick Result")
                    
                    gr.Examples(
                        examples=[
                            [os.path.join(self.train_image_dir, f) for f in os.listdir(self.train_image_dir)[:4]]
                        ],
                        inputs=image_input
                    )
                    
                    predict_btn.click(
                        predict_and_display,
                        inputs=[image_input],
                        outputs=[output_image, output_text, output_label]
                    )
                
                # 实时摄像头标签页
                with gr.Tab("📹 Real-time Webcam"):
                    with gr.Row():
                        webcam_input = gr.Image(label="Webcam Feed", source="webcam", streaming=True)
                        webcam_output = gr.Image(label="Real-time Prediction")
                    
                    with gr.Row():
                        webcam_result = gr.Markdown(label="Detection Result")
                    
                    webcam_input.change(
                        predict_webcam,
                        inputs=[webcam_input],
                        outputs=[webcam_output, webcam_result]
                    )
            
            gr.Markdown(
                """
                ---
                
                ### 📊 Supported Actions (15 Classes)
                
                | Action | Action | Action | Action | Action |
                |--------|--------|--------|--------|--------|
                | sitting | using_laptop | hugging | sleeping | drinking |
                | clapping | dancing | cycling | calling | laughing |
                | eating | fighting | listening_to_music |  |  |
                
                ---
                
                ### 💡 Tips for Better Recognition
                
                - Ensure the person is clearly visible in the image
                - Good lighting conditions improve accuracy
                - The action should be clearly performed
                - Front or side view works best
                """
            )
        
        return demo
    
    def run(self):
        """运行Gradio应用"""
        demo = self.create_gradio_interface()
        if demo:
            demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
        else:
            print("Failed to create interface")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Pose Recognition System')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--run', action='store_true', help='Run the Gradio interface')
    parser.add_argument('--data_path', default='datasets/Human Action Recognition', help='Path to dataset')
    
    args = parser.parse_args()
    
    system = AdvancedPoseRecognition(data_path=args.data_path)
    
    if args.train:
        system.train()
    elif args.run:
        system.run()
    else:
        print("Please specify --train or --run")
        print("Example: python advanced_pose_recognition.py --run")

if __name__ == "__main__":
    main()
