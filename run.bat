@echo off
rem AirFruit launcher - picks a Python that actually has the deps.
rem   run.bat                                  play with the camera
rem   run.bat --selftest 30                    timed regression run
rem   run.bat --selftest 40 --camera-live --visible
rem Do NOT run main.py with a bare "python": Anaconda base ships an old
rem msvcp140.dll next to python.exe that shadows the system one, so
rem PySide6's Qt6Core.dll fails with "procedure not found".
setlocal
set "HERE=%~dp0"
set "PY=D:\FruitNinjaEnv\Scripts\python.exe"
if not exist "%PY%" set "PY=%HERE%.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [AirFruit] no usable Python env found.
    echo             expected: D:\FruitNinjaEnv\Scripts\python.exe
    echo             or in-repo: %HERE%.venv\Scripts\python.exe
    pause
    exit /b 1
)
echo [AirFruit] python: %PY%
"%PY%" "%HERE%main.py" %*
set "CODE=%ERRORLEVEL%"
if not "%CODE%"=="0" pause
exit /b %CODE%