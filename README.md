# 🌌 Gradient Echoes
> *Classical grit, quantum wit.*  
Optimization algorithms — both **classical** (SGD, Adam, L-BFGS, etc.) and **quantum-inspired** (QNG, SPSA, QAOA) — unified in one playground.


I built this repo because I _love optimization_ — not just as code, but as math, motion, and ideas.When I was working through problems in class and on my own, I realized I wanted a single place where all the classical and quantum optimization tricks could live together.Gradient Echoes is that place. A geeky playground, a toolkit, and (hopefully) a helpful reference for anyone who shares the same obsession.

Why this repo?
--------------

*   Because optimization is everywhere: training neural networks, solving physics models, simulating quantum systems.
    
*   Because the same math can look totally different when you see it in motion.
    
*   Because open-sourcing means someone else out there might learn faster or discover something cool.
    

So I’ve tried to write this in a way that’s:

*   **Beginner friendly**: you can just copy a single optimizer into your own code.
    
*   **Research friendly**: there are abstractions (Objective, Constraint, Callback, Schedule) so you can experiment like a scientist.
    
*   **Geek friendly**: lots of visualizations, tests, and room to add crazy new algorithms.
    

Features
--------

Classical optimizers:

*   Gradient Descent and SGD (with Momentum, Nesterov)
    
*   AdaGrad, RMSProp, Adam, AMSGrad
    
*   L-BFGS, Nelder–Mead
    
*   OneCycle scheduler
    

Quantum-flavored optimizers:

*   SPSA (Simultaneous Perturbation Stochastic Approximation)
    
*   Quantum SPSA
    
*   Parameter-Shift Gradient
    
*   Quantum Natural Gradient
    
*   QAOA loop (toy and backend-free for now)
    

Core abstractions:

*   Objective (function + gradient + init)
    
*   Constraint (projections for box domains, etc.)
    
*   Callback (logging, history, custom hooks)
    
*   Schedule (learning-rate schedules like OneCycle)
    

Quick Install
-------------

Clone and install:
```
git clone https://github.com/<you>/gradient-echoes.git
cd gradient-echoes
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\Activate.ps1 on Windows
pip install -e ".[dev]"
```
Optional extras:
```
pip install numpy matplotlib plotly
```

First Run
---------

Run a quick demo:
```
python -m examples.hello_gradient
```
Expected output:<br>
SGD -> x\* ≈ 3.00, f\* ≈ 0.0<br>
Adam -> x\* ≈ 3.00, f\* ≈ 0.0

Visual Showcase
---------------

Optimization is not just numbers — it’s trajectories, valleys, echoes in a landscape.Here are some visual stories from the repo:

### Optimizer Loss Curves (Quadratic)

Different optimizers racing down the same valley.Notice how momentum, adaptivity, and scaling all create distinct learning curves.

\[placeholder: loss\_curves.png\]

### Rosenbrock 3D Landscape

The infamous banana valley.Watch Adam weave through the curved canyon until it reaches the global minimum.

\[placeholder: rosenbrock\_adam\_3d.png\]

### Contour Map with Trajectory

A contour map gives intuition in 2D.Here, the optimizer’s path (red markers) shows how it zig-zags and settles inside the narrow valley.

\[placeholder: rosenbrock\_contour\_traj.png\]

Running Specific Optimizers
---------------------------

Want just one algorithm? Copy-paste it.

Example: SPSA on a toy problem:
```
from gradient_echoes.core.objective import Objective
from gradient_echoes.classical import SPSA

f = lambda x: (x - 2.0)**2
g = lambda x: 2*(x - 2.0)
obj = Objective(f, g, init=0.0)

opt = SPSA(a=0.2, c=0.1)
x_star, hist = opt.minimize(obj, steps=300)
print("x*", x_star, "f*", hist[-1]["f"])
```
Or run via helper script:
```
python scripts/run_single_optimizer.py --alg spsa --steps 300
```

Reproducing Graphs
------------------

Every figure in docs/assets/ can be regenerated from a script in scripts/:
```
python scripts/plot_loss_curves.py
python scripts/plot_rosenbrock_3d.py
python scripts/plot_rosen_contour.py
```
Dependencies: numpy, matplotlib.That way, the repo stays reproducible and tweakable.

Testing
-------

Run the tests:
```
pytest
```

They check convergence behavior, schedules, SPSA stability, etc.<br>
Not perfect convergence — just that things behave sensibly.

Roadmap
-------

*   Add RAdam, Nadam, more exotic variants
    
*   Vectorized L-BFGS implementation (NumPy / JAX-ready)
    
*   Hook into PennyLane/Qiskit for real quantum circuits
    
*   Add more visuals (animated descent, interactive Plotly dashboards)
    

Contributing
------------

Pull requests welcome!<br>
To add something new:

*   put your optimizer in gradient\_echoes/classical/ or quantum/
    
*   add an example in examples/
    
*   add a short test in tests/
    

License
-------

MIT License © 2025

Final Note
----------

This repo is my way of sharing the joy of math, code, and optimization.<br>
If you’re a student, a researcher, or just a curious geek: welcome aboard.<br>
Clone, tweak, run the plots, and maybe invent your own algorithm.

Gradient Echoes is meant to be alive — every optimizer leaves echoes in the landscape.
