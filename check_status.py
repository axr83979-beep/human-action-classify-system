"""
系统状态检查脚本
快速检查系统各个组件的状态
"""

import os
import sys
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("\n" + "="*60)
    print(" Python Version Check")
    print("="*60)

    version = sys.version_info
    print(f"\nPython {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 8:
        print("✅ Python version OK (3.8+ required)")
        return True
    else:
        print("❌ Python version too old (need 3.8+)")
        return False

def check_dependencies():
    """检查依赖包"""
    print("\n" + "="*60)
    print(" Dependencies Check")
    print("="*60)

    packages = {
        'tensorflow': ('2.12.0', 'pip install tensorflow'),
        'cv2': ('4.8.0', 'pip install opencv-python'),
        'gradio': ('4.0.0', 'pip install gradio'),
        'pandas': ('2.0.0', 'pip install pandas'),
        'numpy': ('1.24.0', 'pip install numpy'),
        'sklearn': ('1.3.0', 'pip install scikit-learn'),
        'PIL': ('10.0.0', 'pip install Pillow')
    }

    all_ok = True

    for package, (min_version, install_cmd) in packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')

            if package == 'sklearn':
                module = __import__('sklearn')
                version = module.__version__

            print(f"✅ {package:15s} {version}")
        except ImportError:
            print(f"❌ {package:15s} NOT INSTALLED")
            print(f"   → {install_cmd}")
            all_ok = False

    return all_ok

def check_dataset():
    """检查数据集"""
    print("\n" + "="*60)
    print(" Dataset Check")
    print("="*60)

    data_path = 'datasets/Human Action Recognition'

    # 检查目录
    if not Path(data_path).exists():
        print(f"❌ Dataset directory not found: {data_path}")
        return False

    print(f"✅ Dataset directory found")

    # 检查文件
    files_to_check = [
        'Training_set.csv',
        'Testing_set.csv',
        'train',
        'test'
    ]

    all_ok = True
    for filename in files_to_check:
        file_path = Path(data_path) / filename
        if file_path.exists():
            size = file_path.stat().st_size / 1024 if file_path.is_file() else 'dir'
            print(f"✅ {filename:25s} {size}")
        else:
            print(f"❌ {filename:25s} NOT FOUND")
            all_ok = False

    # 检查图像数量
    train_dir = Path(data_path) / 'train'
    if train_dir.exists():
        image_count = len(list(train_dir.glob('*.jpg')))
        print(f"\n📊 Training images: {image_count}")

        if image_count < 10000:
            print(f"⚠️  Low image count (expected ~12600)")
        else:
            print(f"✅ Image count OK")

    return all_ok

def check_model():
    """检查模型"""
    print("\n" + "="*60)
    print(" Model Check")
    print("="*60)

    model_files = [
        ('efficientnetv2_final_model.h5', 'Model'),
        ('efficientnetv2_label_encoder.pkl', 'Label Encoder'),
        ('efficientnetv2_model_info.pkl', 'Model Info')
    ]

    all_ok = True
    for filename, description in model_files:
        file_path = Path(filename)
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✅ {description:20s} {filename:40s} {size_mb:.1f} MB")
        else:
            print(f"❌ {description:20s} {filename:40s} NOT FOUND")
            all_ok = False

    return all_ok

def check_scripts():
    """检查脚本文件"""
    print("\n" + "="*60)
    print(" Scripts Check")
    print("="*60)

    scripts = [
        ('launcher.py', 'Launch Menu'),
        ('advanced_pose_recognition.py', 'Main Program'),
        ('opencv_realtime_recognition.py', 'OpenCV Version'),
        ('verify_data.py', 'Data Validator'),
        ('requirements.txt', 'Dependencies')
    ]

    all_ok = True
    for filename, description in scripts:
        if Path(filename).exists():
            print(f"✅ {description:20s} {filename}")
        else:
            print(f"❌ {description:20s} {filename} NOT FOUND")
            all_ok = False

    return all_ok

def check_system_resources():
    """检查系统资源"""
    print("\n" + "="*60)
    print(" System Resources")
    print("="*60)

    try:
        import psutil
        import platform

        # CPU
        cpu_count = psutil.cpu_count(logical=True)
        print(f"✅ CPU Cores: {cpu_count}")

        # Memory
        mem = psutil.virtual_memory()
        mem_gb = mem.total / (1024**3)
        print(f"✅ Total Memory: {mem_gb:.1f} GB")
        print(f"✅ Available: {mem.available / (1024**3):.1f} GB")

        # Disk
        disk = psutil.disk_usage('.')
        disk_gb = disk.free / (1024**3)
        print(f"✅ Free Disk Space: {disk_gb:.1f} GB")

        # GPU
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"✅ GPU: {len(gpus)} device(s) found")
                for gpu in gpus:
                    print(f"   - {gpu.name}")
            else:
                print("⚠️  No GPU found (will use CPU)")
        except:
            print("⚠️  Could not check GPU")

        return True

    except ImportError:
        print("⚠️  psutil not installed (pip install psutil)")
        return False

def print_summary(results):
    """打印总结"""
    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)

    checks = [
        ('Python Version', results['python']),
        ('Dependencies', results['deps']),
        ('Dataset', results['dataset']),
        ('Model', results['model']),
        ('Scripts', results['scripts']),
        ('Resources', results['resources'])
    ]

    for name, status in checks:
        icon = "✅" if status else "❌"
        print(f"{icon} {name}")

    print("\n" + "="*60)

    # 提供下一步建议
    print("\n📋 Next Steps:\n")

    if not results['deps']:
        print("1. Install dependencies:")
        print("   pip install -r requirements.txt\n")

    if not results['dataset']:
        print("2. Ensure dataset is in datasets/Human Action Recognition/\n")

    if not results['model']:
        print("3. Train the model:")
        print("   python advanced_pose_recognition.py --train")
        print("   or double-click train_model.bat\n")

    if all(results.values()):
        print("✅ All checks passed! You can start using the system:")
        print("   - Run launcher.py to open the menu")
        print("   - or run: python advanced_pose_recognition.py --run\n")

def main():
    """主函数"""
    print("\n" + "="*60)
    print(" "*10 + "SYSTEM STATUS CHECK")
    print("="*60)

    results = {
        'python': check_python_version(),
        'deps': check_dependencies(),
        'dataset': check_dataset(),
        'model': check_model(),
        'scripts': check_scripts(),
        'resources': check_system_resources()
    }

    print_summary(results)

    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
