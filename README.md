## Data-driven network model of CA3 - featuring sequence replay and ripple oscillation with plasticity during offline simulation

Code repository of the The Role of Plasticity in Modulating Replay Dynamics paper (Submitted to NeurIPS 2024)

To run:

    git clone https://github.com/sliorbar/ca3net.git
    cd ca3net
    pip3 install -r requirements.txt
    mkdir figures
    cd scripts
    python generate_spike_train.py  # generate CA3 like spike trains (as exploration of a maze)
    python stdp.py  # learns the recurrent weight (via STDP, based on the spiketrain)
    python spw_network_cpp.py  # creates the network, runs the simulation, analyses and plots results. You can change the simulation parameters in this file

The code will not run successfully without the simulation results database.
To install: 
    You'll need a Windows 10 or later machine with at least 16Gb of memory
    Download the Developer Edition of SQL server from https://www.microsoft.com/en-us/sql-server/sql-server-downloads
    Create an empty database named "CUNY" and a new sql user with admin rights on the database
    Run "Database\Database scripts.sql" - Create all tables first before proceding to other objects
    Modify "scripts/datalayer.py" file to use the new database created in previous steps.
To analyze the results:
    Install Power BI desktop from https://www.microsoft.com/en-us/download/details.aspx?id=58494
    Open the "Database/PBI - Synaptic Changes - NeurIPS.pbix" file
    Modify the database connection in Power BI to point to the new database created previously
    

