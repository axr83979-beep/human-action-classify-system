"""
数据验证脚本
检查数据集是否完整，验证数据质量
"""

import os
import pandas as pd
from pathlib import Path
from collections import Counter
import cv2

def verify_dataset():
    """验证数据集"""
    print("="*60)
    print(" DATASET VERIFICATION")
    print("="*60)
    
    data_path = 'datasets/Human Action Recognition'
    train_csv = os.path.join(data_path, 'Training_set.csv')
    test_csv = os.path.join(data_path, 'Testing_set.csv')
    train_dir = os.path.join(data_path, 'train')
    test_dir = os.path.join(data_path, 'test')
    
    # 检查文件存在性
    print("\n1. Checking file existence...")
    files_to_check = [
        ('Training CSV', train_csv),
        ('Testing CSV', test_csv),
        ('Training Directory', train_dir),
        ('Testing Directory', test_dir)
    ]
    
    for name, path in files_to_check:
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"  {status} {name}: {path}")
    
    # 检查CSV文件
    print("\n2. Reading CSV files...")
    if os.path.exists(train_csv):
        train_df = pd.read_csv(train_csv)
        print(f"  ✅ Training samples: {len(train_df)}")
    else:
        print(f"  ❌ Training CSV not found!")
        return False
    
    if os.path.exists(test_csv):
        test_df = pd.read_csv(test_csv)
        print(f"  ✅ Testing samples: {len(test_df)}")
    else:
        print(f"  ❌ Testing CSV not found!")
        return False
    
    # 检查类别分布
    print("\n3. Analyzing class distribution...")
    label_counts = Counter(train_df['label'])
    print(f"  Number of classes: {len(label_counts)}")
    print(f"\n  Class distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"    - {label:20s}: {count:4d} samples")
    
    # 检查图像文件
    print("\n4. Checking image files...")
    missing_images = []
    corrupt_images = []
    sample_images = []
    
    print("  Scanning training images...")
    for idx, row in train_df.head(100).iterrows():  # 只检查前100张
        img_path = os.path.join(train_dir, row['filename'])
        
        if not os.path.exists(img_path):
            missing_images.append(row['filename'])
        else:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    corrupt_images.append(row['filename'])
                elif len(sample_images) < 5:
                    sample_images.append((row['filename'], row['label'], img.shape))
            except Exception as e:
                corrupt_images.append(row['filename'])
    
    print(f"  Checked {min(100, len(train_df))} images...")
    print(f"  ✅ Valid images: {min(100, len(train_df)) - len(missing_images) - len(corrupt_images)}")
    print(f"  ❌ Missing images: {len(missing_images)}")
    print(f"  ❌ Corrupt images: {len(corrupt_images)}")
    
    # 显示示例
    if sample_images:
        print("\n5. Sample images:")
        for filename, label, shape in sample_images[:5]:
            print(f"  - {filename}")
            print(f"    Label: {label}, Shape: {shape}")
    
    # 统计信息
    print("\n6. Dataset statistics:")
    print(f"  Total training samples: {len(train_df)}")
    print(f"  Total testing samples: {len(test_df)}")
    print(f"  Number of classes: {len(label_counts)}")
    print(f"  Average samples per class: {len(train_df) / len(label_counts):.1f}")
    
    # 问题检查
    print("\n7. Issues found:")
    issues = []
    
    if missing_images:
        issues.append(f"⚠️  {len(missing_images)} missing images")
    
    if corrupt_images:
        issues.append(f"⚠️  {len(corrupt_images)} corrupt images")
    
    if len(label_counts) < 10:
        issues.append(f"⚠️  Only {len(label_counts)} classes found (expected 15)")
    
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  ✅ No issues found!")
    
    print("\n" + "="*60)
    print(" VERIFICATION COMPLETE")
    print("="*60)
    
    # 返回验证结果
    is_valid = not (missing_images or corrupt_images or len(label_counts) < 10)
    return is_valid

def verify_model():
    """验证模型文件"""
    print("\n" + "="*60)
    print(" MODEL VERIFICATION")
    print("="*60)
    
    model_files = [
        ('Model', 'efficientnetv2_final_model.h5'),
        ('Label Encoder', 'efficientnetv2_label_encoder.pkl'),
        ('Model Info', 'efficientnetv2_model_info.pkl')
    ]
    
    print("\nChecking model files...")
    all_exist = True
    
    for name, filename in model_files:
        exists = os.path.exists(filename)
        status = "✅" if exists else "❌"
        print(f"  {status} {name}: {filename}")
        if exists:
            size = os.path.getsize(filename) / (1024 * 1024)  # MB
            print(f"      Size: {size:.2f} MB")
        else:
            all_exist = False
    
    print("\n" + "="*60)
    
    return all_exist

def main():
    """主函数"""
    print("\n🔍 Starting verification...\n")
    
    # 验证数据集
    data_valid = verify_dataset()
    
    # 验证模型
    model_valid = verify_model()
    
    # 总结
    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)
    
    print(f"\nDataset: {'✅ Valid' if data_valid else '❌ Invalid'}")
    print(f"Model:   {'✅ Valid' if model_valid else '❌ Not found'}")
    
    if model_valid:
        print("\n✅ You can start using the system:")
        print("   - Gradio: python advanced_pose_recognition.py --run")
        print("   - OpenCV: python opencv_realtime_recognition.py")
    else:
        print("\n❌ Model not found. Please train first:")
        print("   python advanced_pose_recognition.py --train")
    
    print("\n")

if __name__ == "__main__":
    main()
