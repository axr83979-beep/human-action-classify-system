import os
import sys
from pathlib import Path

def print_header():
    """打印标题"""
    print("\n" + "="*60)
    print(" "*15 + "🤸 人体姿态识别系统")
    print("="*60)
    print("\n   Advanced Human Pose Recognition System")
    print("   Powered by EfficientNetV2 + Gradio")
    print("\n" + "="*60)

def print_menu():
    """打印菜单"""
    print("\n   请选择操作:")
    print("\n   [1] 🚀 启动 Gradio Web 界面 (推荐)")
    print("   [2] 📹 启动 OpenCV 实时识别")
    print("   [3] 🎓 训练模型 (首次使用)")
    print("   [4] 🔍 验证数据集")
    print("   [5] 📖 查看文档")
    print("   [0] ❌ 退出")
    print("\n" + "="*60)

def check_dependencies():
    """检查依赖"""
    print("\n🔍 检查依赖...")

    required_packages = {
        'tensorflow': 'pip install tensorflow',
        'cv2': 'pip install opencv-python',
        'gradio': 'pip install gradio',
        'pandas': 'pip install pandas',
        'numpy': 'pip install numpy',
        'sklearn': 'pip install scikit-learn'
    }

    missing = []
    for package, install_cmd in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing.append(install_cmd)
            print(f"   ❌ {package}")
        else:
            print(f"   ✅ {package}")

    if missing:
        print("\n⚠️  缺少依赖包，请运行:")
        for cmd in set(missing):
            print(f"   {cmd}")
        print("\n   或一键安装: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ 所有依赖已安装!")
        return True

def check_model():
    """检查模型"""
    model_files = [
        'efficientnetv2_final_model.h5',
        'efficientnetv2_label_encoder.pkl'
    ]

    print("\n🔍 检查模型...")

    for filename in model_files:
        if Path(filename).exists():
            print(f"   ✅ {filename}")
        else:
            print(f"   ❌ {filename}")
            return False

    return True

def launch_gradio():
    """启动Gradio"""
    print("\n🚀 启动 Gradio Web 界面...")
    print("\n提示:")
    print("  - 界面将在浏览器中打开")
    print("  - 支持图片上传和实时摄像头")
    print("  - 按 Ctrl+C 停止服务")
    print("\n" + "="*60 + "\n")

    os.system("python advanced_pose_recognition.py --run")

def launch_opencv():
    """启动OpenCV"""
    print("\n📹 启动 OpenCV 实时识别...")
    print("\n提示:")
    print("  - 使用 OpenCV 窗口显示")
    print("  - Q: 退出")
    print("  - S: 保存当前帧")
    print("\n" + "="*60 + "\n")

    os.system("python opencv_realtime_recognition.py")

def train_model():
    """训练模型"""
    print("\n🎓 开始训练模型...")
    print("\n注意:")
    print("  - 训练时间: 30-60分钟")
    print("  - 需要足够的内存")
    print("  - 首次使用必须训练")
    print("\n" + "="*60 + "\n")

    confirm = input("确认开始训练? (y/n): ").lower()
    if confirm == 'y':
        os.system("python advanced_pose_recognition.py --train")
    else:
        print("已取消训练")

def verify_data():
    """验证数据"""
    print("\n🔍 验证数据集和模型...")
    print("\n" + "="*60 + "\n")

    os.system("python verify_data.py")

def show_docs():
    """显示文档"""
    print("\n📖 查看文档...")

    docs = {
        '1': 'README.md (快速开始)',
        '2': 'README_ADVANCED.md (详细文档)'
    }

    print("\n选择文档:")
    for key, doc in docs.items():
        exists = "✅" if Path(doc).exists() else "❌"
        print(f"   [{key}] {exists} {doc}")

    print("   [0] 返回")
    print()

    choice = input("请选择: ").strip()

    if choice == '1' and Path('README.md').exists():
        with open('README.md', 'r', encoding='utf-8') as f:
            print("\n" + f.read())
    elif choice == '2' and Path('README_ADVANCED.md').exists():
        with open('README_ADVANCED.md', 'r', encoding='utf-8') as f:
            print("\n" + f.read())

def main():
    """主函数"""
    while True:
        print_header()
        print_menu()

        choice = input("\n请输入选项 (0-5): ").strip()

        if choice == '0':
            print("\n👋 再见!\n")
            sys.exit(0)

        elif choice == '1':
            # 启动Gradio
            if not check_dependencies():
                continue

            if not check_model():
                print("\n⚠️  模型未找到!")
                print("   请先选择 [3] 训练模型")
                input("\n按回车继续...")
                continue

            launch_gradio()

        elif choice == '2':
            # 启动OpenCV
            if not check_dependencies():
                continue

            if not check_model():
                print("\n⚠️  模型未找到!")
                print("   请先选择 [3] 训练模型")
                input("\n按回车继续...")
                continue

            launch_opencv()

        elif choice == '3':
            # 训练模型
            if not check_dependencies():
                continue

            train_model()

        elif choice == '4':
            # 验证数据
            verify_data()
            input("\n按回车继续...")

        elif choice == '5':
            # 查看文档
            show_docs()
            input("\n按回车继续...")

        else:
            print("\n❌ 无效选项，请重新选择")
            input("\n按回车继续...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 再见!\n")
        sys.exit(0)
