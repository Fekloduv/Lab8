import numpy as np
import matplotlib.pyplot as plt

def modified_euler_method(rhs_func, initial_condition, t0, T, h0, N_x, eps):
    t = t0
    h = h0
    v = np.array(initial_condition)
    kounter = [0]
    results = []
    steps = []
    solutions = []
    coord = []

    print("{:12.6f} {:12.6f} {:12s} {:12d} {:12.6f} {:12.6f} {:12.6f}".format(
        t, h, "0", kounter[0], *v))

    def euler_Modf(t, v, h, kounter):  # Передача kounter в качестве аргумента
        v_hat = rhs_func(t, v, kounter)
        v_tilde = rhs_func(t + h, v + h * v_hat, kounter)
        return v + (h / 2) * (v_hat + v_tilde)

    while t < T and kounter[0] < N_x:
        v_First = euler_Modf(t, v, h, kounter)
        v_Second = euler_Modf(t, v, h/2, kounter)
        v_Second = euler_Modf(t + h/2, v_Second, h/2, kounter)

        R = np.linalg.norm(v_First - v_Second) / (pow(2, 2) - 1)

        if R > eps:
            h /= 2
        elif R < (eps / 64):
            h *= 2
        else:
            v = v_First
            t += h
            steps.append(h)
            solutions.append(v.copy())
            coord.append(t)

            print("{:12.6f} {:12.6f} {:12.5e} {:12d} {:12.6f} {:12.6f} {:12.6f}".format(
                t, h, R, kounter[0], *v))

        if t + h > T:
            h = T - t

        results.append((t, h, R, kounter[0], *v))

    return results, steps, solutions, coord

# Определение функции fs
def fs(t, v, kounter):
    A = np.array([[-0.4, 0.02, 0], [0, 0.8, -0.1], [0.003, 0, 1]])
    kounter[0] += 1
    return np.dot(A, v)

t0 = 1.5
T = 2.5
h0 = 0.1
N_x = 10000
eps = 0.0001
eps_count = [0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
initial_condition = [1, 1, 2]

results, steps, solutions, coord = modified_euler_method(fs, initial_condition, t0, T, h0, N_x, eps)

result_list = []
for eps in eps_count:
    print("Eps = ", eps)
    results = modified_euler_method(fs, initial_condition, t0, T, h0, N_x, eps)
    result_list.append(results)
    print(" ")
