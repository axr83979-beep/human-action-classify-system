@echo off
echo ========================================
echo OpenCV Real-time Pose Recognition
echo ========================================
echo.
echo This will start the OpenCV-based real-time recognition.
echo.
echo Make sure:
echo 1. Model is trained: python advanced_pose_recognition.py --train
echo 2. Webcam is connected
echo.
echo Controls:
echo   Q - Quit
echo   S - Save frame
echo.
echo Press any key to start...
pause > nul
python opencv_realtime_recognition.py
pause
