:: to do: move this away from windows, batch files are horrible

@ECHO off 

:: classes - 1
SET /A classes=63 
SET /A real_classes="%classes%+1"
ECHO classes = %real_classes%

>output.txt (
    FOR /l %%x in (0, 1, %classes%) do (
        C:\toolkits\anaconda3-4.2.0\envs\tensorflow\python.exe DNN_regression_train.py --index %%x
    )
)