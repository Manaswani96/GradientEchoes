# examples/parameter_shift_demo.py
import math
from gradient_echoes.quantum import parameter_shift_grad

def exp_cos(theta): return math.cos(theta)

for th in [0.0, 0.3, 1.0, 2.1]:
    g_ps = parameter_shift_grad(exp_cos, th)      # exact for cos with s=pi/2
    g_an = -math.sin(th)
    print(f"theta={th:.2f}  grad_ps={g_ps:.6f}  grad_analytic={g_an:.6f}  diff={abs(g_ps-g_an):.2e}")
