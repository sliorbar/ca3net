@echo off

:: Path to the specific Python executable
set PYTHON_EXEC=C:\ProgramData\Anaconda3\python.exe
:: Path to the Python file
set PYTHON_FILE=.\replay_stats_simulations.py

:: Number of iterations
set ITERATIONS=3

:: Run the Python script multiple times
for /L %%i in (1,1,%ITERATIONS%) do (
    echo Running iteration %%i
    python %PYTHON_FILE% asym sym "LTP pre only 20 ms tau %%i" alt 5000
)
