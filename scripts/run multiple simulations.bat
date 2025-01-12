@echo off

:: Path to the specific Python executable
set PYTHON_EXEC=C:\ProgramData\Anaconda3\python.exe
:: Path to the Python file
set PYTHON_FILE=C:\Users\lior_\Documents\Code\ca3net\scripts\replay_stats_simulations-param-sweep.py
set STDP_FILE=C:\Users\lior_\Documents\Code\ca3net\scripts\stdp.py
set SPIKES_FILE=C:\Users\lior_\Documents\Code\ca3net\scripts\generate_spike_train.py

:: Number of iterations
set ITERATIONS=100

:: Run the Python script multiple times
for /L %%i in (1,1,%ITERATIONS%) do (
    echo Running iteration %%i
    python %SPIKES_FILE% 
    python %STDP_FILE% asym sym 
    python %PYTHON_FILE% asym sym "Randomize start sym %%i" alt 5000
)
