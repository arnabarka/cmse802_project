# cmse802_project

### Directory Organization  

**Proposed Folder Structure:**
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


**Project Title:** Python-Based Simulation and Prediction of Vortex Shedding Behind a Circular Cylinder at Moderate Reynolds Numbers<br>

**Brief Description:**
This project will simulate and analyze vortex shedding behind a circular cylinder across a moderate Reynolds number range (Re = 100–1500) using Python-based finite difference methods. The study will compute drag and lift coefficients, estimate shedding frequencies, and capture wake patterns in different flow regimes. Results will be validated against benchmark experimental and numerical data using regression-based fits and RMSE error analysis. An optional regression-based ML model will also be explored to predict drag coefficient trends as a function of Reynolds number.

**Project Objectives:**
1. **Simulate** incompressible 2D cylinder flow at Reynolds numbers from 100 to 1500 using a Python-based finite difference solver.  
2. **Compute and analyze** drag and lift coefficients, as well as vortex shedding frequency, across the studied Reynolds range.  
3. **Validate** numerical predictions of Strouhal number and drag coefficient against benchmark data by applying regression fits and calculating RMSE error.  
4. **Explore** (optional) a regression-based ML model for predicting average drag coefficient from Reynolds number to complement solver results.<br>

**Instruction for setting up and running the code(Methodology):**
The project will use a **finite difference method (FDM)**–based numerical solver implemented entirely in Python to simulate the flow past a circular cylinder at Reynolds numbers between 100 and 1500. The approach balances physical fidelity with computational feasibility by focusing on 2D incompressible flow.  

**Algorithms and Models:**  
- **Governing Equations:** 2D incompressible Navier–Stokes equations.  
- **Numerical Method:** Projection (fractional-step) method for velocity–pressure coupling.  
- **Spatial Discretization:**  
  - Central differencing for diffusion terms.  
  - Upwind differencing for convection terms.  
- **Time Integration:** Explicit Euler or 2nd-order Runge–Kutta schemes.  
- **Pressure Poisson Equation:** Solved iteratively using Gauss–Seidel relaxation.  
- **Boundary Conditions:**  
  - No-slip on the cylinder surface.  
  - Uniform inflow at the left boundary.  
  - Convective outflow on the right.  
  - Symmetry or free-slip on top and bottom boundaries.  

**Code Structure:**  
- `initialize_grid()` → create computational domain and cylinder boundary.  
- `momentum_step()` → update velocity fields using discretized Navier–Stokes equations.  
- `pressure_poisson()` → solve for pressure correction using Gauss–Seidel iteration.  
- `apply_boundary_conditions()` → enforce inflow/outflow and cylinder no-slip conditions.  
- `compute_forces()` → calculate drag and lift coefficients from surface stresses.  
- `postprocess()` → extract Strouhal number, generate flow visualizations, and compare with benchmarks.  

**Computational Techniques from the Course:**  
- **Finite Difference Methods (FDM):** discretization of PDEs.  
- **Linear Algebra:** iterative solvers for the Poisson equation.  
- **ODE Integration:** time stepping of velocity fields.  
- **Stability Analysis:** CFL condition to determine allowable time step.  
- **Regression and RMSE:** validation of drag and Strouhal predictions against benchmark data.  
- **Optional ML:** regression-based model in Python (scikit-learn) for predicting drag coefficient trends with Reynolds number.  

**Planned Approach Overview:**  
1. Implement the solver in Python and test it at Re = 100 to reproduce classical vortex shedding.  
2. Gradually increase Reynolds number up to 1500, analyzing changes in wake structure.  
3. Compute drag coefficient and shedding frequency for each case.  
4. Compare simulation outputs with benchmark data using regression fits and RMSE error.  
5. (Optional) Use a regression-based ML model to provide an additional predictive comparison.  

**Dependencies and Requirements**  

This project requires the following Python libraries:  
- **NumPy** → numerical arrays and vectorized operations  
- **SciPy** → linear algebra solvers and numerical methods  
- **Matplotlib** → visualization of flow fields and results  
- **scikit-learn** (optional) → regression-based ML for drag coefficient prediction  


