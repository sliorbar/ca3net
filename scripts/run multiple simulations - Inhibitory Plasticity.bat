@echo off

:: Path to the specific Python executable
set PYTHON_EXEC=C:\Users\lior_\Documents\Code\ca3net\.venv\Scripts\python.exe
:: Path to the Python file
set PYTHON_FILE=C:\Users\lior_\Documents\Code\ca3net\scripts\replay_stats_simulations-Inhibitory-Plasticity.py
::set PYTHON_FILE=C:\Users\lior_\Documents\Code\ca3net\scripts\replay_stats_simulations.py
::set PYTHON_FILE=C:\Users\lior_\Documents\Code\ca3net\scripts\replay_stats_simulations-param-sweep.py
::set STDP_FILE=C:\Users\lior_\Documents\Code\ca3net\scripts\stdp-inh.py
set STDP_FILE=C:\Users\lior_\Documents\Code\ca3net\scripts\stdp.py
set SPIKES_FILE=C:\Users\lior_\Documents\Code\ca3net\scripts\generate_spike_train.py

:: Number of iterations
set ITERATIONS=1


:: Run the Python script multiple times
for /L %%i in (1,1,%ITERATIONS%) do (
    echo Running iteration %%i 
    ::python %SPIKES_FILE% 
    ::python %STDP_FILE% asym sym 
    python %PYTHON_FILE% asym sym "New datalayer pc-pc-h pc-ih bc-eAh %%i" alt 5000 
    ::python %PYTHON_FILE% asym sym "sSTDP Homeostasis 0.90 %%i - 1" alt 5000 
    ::python %PYTHON_FILE% asym sym "sSTDP Homeostasis 0.90 %%i - 2" alt 5000 
    ::python %PYTHON_FILE% asym sym "Rand start SynTag-real ah sk %%i - 3" alt 5000 
    ::python %PYTHON_FILE% asym sym "Rand start SynTag ah sk %%i - 4" alt 5000 
    :: Update SYN_Threshold
    ::set /A SYN_Threshold+=0.25

    
)
