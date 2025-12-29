@echo off
echo Starting Advanced Pose Recognition System...
echo.
echo Make sure you have installed all dependencies:
echo pip install -r requirements.txt
echo.
echo If this is your first time, train the model first:
echo python advanced_pose_recognition.py --train
echo.
echo Press any key to start the system...
pause > nul
python advanced_pose_recognition.py --run
pause
