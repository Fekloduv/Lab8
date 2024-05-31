import numpy as np
import matplotlib.pyplot as plt


def next(h, t, v, kounter):
    return v + h * fs(t + h/2, v + (h * fs(t, v, kounter))/2, kounter)


def dlt(v, v1):
    return np.linalg.norm(np.array(v) - np.array(v1), ord=2) / (pow(2, 3) - 1)


def method(t, T, h, v, kounter, eps):
    x = []
    y = []
    y1 = []
    y2 = []
    y3 = []
    kounter[0] = 0
    v2 = v
    while t < T + h / 2:
        v_0 = v
        y1.append(v_0[0])
        y2.append(v_0[1])
        y3.append(v_0[2])
        print("{:13.6f}".format(t), "{:12.6f}".format(h), "{:15.5e}".format(dlt(v, v2)), "{:12d}".format(kounter[0]),
              end=' ')
        for vi in v_0:
            print("{:12.6f}".format(vi), end=' ')
        print()
        v = next(h, t, v_0, kounter)
        v1 = next(h / 2, t, v_0, kounter)
        v2 = next(h / 2, t + h / 2, v1, kounter)
        while dlt(v, v2) > eps:
            h /= 2
            v = v1
            v1 = next(h / 2, t, v_0, kounter)
            v2 = next(h / 2, t + h / 2, v1, kounter)
        t += h
        x.append(t)
        y.append(h)
    print()
    x.pop()
    y.pop()
    y1.pop()
    y2.pop()
    y3.pop()
    return x, y, min(y), len(x), y1, y2, y3


def graphs(t, T, h, v, kounter):
    epses = [0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
    k = 0
    minhes = []
    numstepses = []
    fig, ax = plt.subplots(2, 3)
    fig.tight_layout()
    fig1, ax1 = plt.subplots(2, 3)
    fig1.tight_layout()
    for item in epses:
        ax[k // 3, k % 3].set_title('eps = ' + str(item))
        ax[k // 3, k % 3].set_xlabel('Отрезок')
        ax[k // 3, k % 3].set_ylabel('Шаг')
        ax1[k // 3, k % 3].set_title('eps = ' + str(item))
        ax1[k // 3, k % 3].set_xlabel('Отрезок')
        ax1[k // 3, k % 3].set_ylabel('Решение')
        print('eps = ', item)
        x, y, minh, numsteps, y1, y2, y3 = method(t, T, h, v, kounter, item)
        minhes.append(minh)
        numstepses.append(numsteps)
        ax[k // 3, k % 3].scatter(x, y)
        ax1[k // 3, k % 3].plot(x, y1)
        ax1[k // 3, k % 3].plot(x, y2)
        ax1[k // 3, k % 3].plot(x, y3)
        k += 1
    plt.figure()
    plt.semilogx(epses, minhes)
    plt.xlabel('Точность')
    plt.ylabel('Минимальный шаг')
    plt.title('Зависимость min шага от точности')
    plt.figure()
    plt.semilogx(epses, numstepses)
    plt.xlabel('Точность')
    plt.ylabel('Число шагов')
    plt.title('Зависимость числа шагов от точности')
    plt.show()


if __name__ == '__main__':
    func = []
    t = float(input())
    T = float(input())
    h = float(input())
    N_x = int(input())
    eps = float(input())
    n = int(input())
    for i in range(n + 3):
        func.append(input())
    b = ''
    for s in func:
        b = b + s + '\n'
    v_0 = input().split()
    for i in range(n):
        v_0[i] = float(v_0[i])
    v = v_0
    v1 = v_0
    v2 = v_0
    kounter = [0]
    exec(b)
    graphs(t, T, h, v, kounter)
