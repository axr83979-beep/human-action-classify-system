@echo off
echo ========================================
echo Advanced Pose Recognition - Training
echo ========================================
echo.
echo This will train the EfficientNetV2 model.
echo Estimated time: 30-60 minutes
echo.
echo Make sure you have:
echo 1. Installed dependencies: pip install -r requirements.txt
echo 2. Dataset in datasets/Human Action Recognition/
echo.
echo Press Ctrl+C to cancel
pause
echo.
echo Starting training...
python advanced_pose_recognition.py --train
echo.
echo Training completed!
pause
