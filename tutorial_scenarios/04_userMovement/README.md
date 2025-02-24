One way to run the code:

```
export PYTHONPATH=$PYTHONPATH:~/YAFS/src/
cd YAFS/tutorial_scenarios/04_userMovement
python main.py
```


The project contains the next files:

```
├── data
│   ├── allocDefinition.json
│   └── appDefinition.json
├── logFile.log
├── logging.ini
├── main.py
├── readme.md
└── results
    ├── graph_binomial_tree_5.gexf
    ├── sim_trace.csv
    └── sim_trace_link.csv
```

- main.py controls this simulation and generates all the pieces.
- data/allocDefinition.json defines the allocation of app's instances
- data/appDefinition.json defines the description of the applications
- results/graph_binomial_tree_5 a figure of the resulting topology using NetworkX functions.
- results/sim_trace.csv the simulation traces, it contains the requests handled by each instance along the simulation.
- results/sim_trace_link.csv the simlation traces, it contains the network messages between the nodes generated by the requests.
  
  