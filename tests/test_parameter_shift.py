# tests/test_parameter_shift.py
import math
from gradient_echoes.quantum import parameter_shift_grad

def test_parameter_shift_matches_derivative_for_cos():
    for th in [0.0, 0.4, 1.1, 2.2]:
        g_ps = parameter_shift_grad(lambda t: math.cos(t), th)
        g_an = -math.sin(th)
        assert abs(g_ps - g_an) < 1e-12  # exact for cos with s=pi/2
