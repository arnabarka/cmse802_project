# cmse802_project
```
cylinder-vortex-project/
│
├── data/                  # Stores benchmark datasets and any preprocessed data
│   └── benchmarks.csv     # Reference values for drag and Strouhal validation
│
├── src/                   # Core source code for the simulation and analysis
│   ├── solver.py          # Main finite difference Navier–Stokes solver
│   ├── poisson_solver.py  # Iterative solver for the pressure Poisson equation
│   ├── postprocess.py     # Postprocessing routines (drag/lift, Strouhal analysis)
│   └── ml_regression.py   # Optional regression-based ML model for Cd vs Re
│
├── notebooks/             # Jupyter notebooks for development and reporting
│   └── ProjectPlan.ipynb  # Homework 1 project planning notebook
│
├── results/               # Stores output generated from simulations
│   ├── plots/             # Flow field plots, drag/lift curves, vortex shedding
│   └── tables/            # RMSE validation tables and regression results
│
├── tests/                 # Unit tests to verify solver correctness
│   └── test_solver.py     # Tests for core solver and Poisson solver
│
├── README.md              # Project overview, objectives, setup instructions
```
└── requirements.txt       # List of required Python packages and dependencies

