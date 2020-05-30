import math
import numpy as np
from typing import Tuple, List

from plot_util import PlotWrapper, orange


def dichotomy(a: float, b: float, eps: float, k: int, n) -> Tuple[float, float]:
    delta = (b - a) / 6
    mid = (b - a) / 2
    x1 = a + mid - delta
    x2 = a + mid + delta
    return x1, x2


def golden_ratio(a: float, b: float, eps: float, k: int, n) -> Tuple[float, float]:
    x1 = a + (3 - math.sqrt(5)) / 2 * (b - a)
    x2 = a + (math.sqrt(5) - 1) / 2 * (b - a)
    x2 = min(x2, b - eps / 2)
    x1 = max(x1, a + eps / 2)
    return x1, x2


def fibonacci_method(a: float, b: float, eps: float, k: int, n: int) -> Tuple[float, float]:
    f_den = _mem_fib(n - k + 3)
    f_upper = _mem_fib(n - k + 2)
    f_lower = _mem_fib(n - k + 1)
    x1 = a + f_lower / f_den
    x2 = a + f_upper / f_den
    x2 = min(x2, b - eps / 2)
    x1 = max(x1, a + eps / 2)
    return x1, x2


# we can use binary search here, but it doesn't change asymptotic all algorithm
def preprocess_for_fib(eps: float, a: float, b: float) -> int:
    i = 0
    max_range = (b - a) / eps
    while _mem_fib(i) < max_range:
        i += 1
    return i


fib_arr: List[int] = [1, 1]


# function for calculation fibonacci with memoization
def _mem_fib(n: int) -> int:
    if len(fib_arr) > n:
        return fib_arr[n]
    f_1 = _mem_fib(n - 1)
    f_2 = _mem_fib(n - 2)
    ans = f_1 + f_2
    if len(fib_arr) == n:
        fib_arr.append(ans)
    else:
        raise Exception("bugs in fibonacci function")

    return ans


def returnNone(eps, a, b):
    return None


def one_dim_extremum(f, a: float, b: float, eps, method=dichotomy, preprocess=returnNone, is_max: bool = False) -> \
        Tuple[float, float, int, List[float], List[float], List[float], List[float]]:
    iterations = 1
    prep = preprocess(eps, a, b)
    ases = [a]
    fases = [f(a)]
    bses = [b]
    fbses = [f(b)]
    while b - a > eps:
        iterations += 1
        x1, x2 = method(a, b, eps, iterations, prep)
        f1 = f(x1)
        f2 = f(x2)
        if f1 == f2:
            a = x1
            b = x2
        elif (f1 < f2) ^ is_max:
            b = x2
        else:
            a = x1
        ases.append(a)
        fases.append(f(a))
        bses.append(b)
        fbses.append(f(b))
    return f(a), a, iterations, ases, fases, bses, fbses


def _find_base_interval(f, x0: float, delta: float) -> Tuple[float, float]:
    f_x0 = f(x0)
    f_delta = f(x0 + delta)
    if f_x0 > f_delta:
        x1 = x0 + delta
        h = delta
    elif f_x0 < f_delta:
        x1 = x0 - delta
        h = -delta
    else:
        return x0, x0 + delta
    while f(x1) > f(x1 + h):
        x0 = x1
        x1 = x1 + h
        h *= 2
    return x0, x1 + h


def extremum_in_line(f, x0=0., delta=10e-5, eps=10e-5, method=dichotomy, preprocess=returnNone):
    a, b = _find_base_interval(f, x0, delta)
    return one_dim_extremum(f, a, b, eps, method, preprocess)


def func(x):
    return x ** 2 + 2 * x - 4


lower = -10
upper = 20


def plot_grafic_for_one_dim(algo_name, method=dichotomy, preprocess=returnNone):
    base_eps = 10e-6
    fa, a, iterations, ases, fases, bses, fbses = one_dim_extremum(func, lower, upper, base_eps, method, preprocess)
    length = len(ases)
    xs = range(length)
    plot = PlotWrapper(0, length)
    plot.add_line(xs, ases, "left bound", 'r')
    plot.add_line(xs, bses, "right bound", 'g')
    plot.x_label("iterations")
    plot.y_label("bound value")
    plot.saveToFile(algo_name + "_bounds")

    plot = PlotWrapper(0, length)
    plot.add_line(xs, fases, "left values", 'r')
    plot.add_line(xs, fbses, "right values", 'g')
    plot.x_label("iterations")
    plot.y_label("bound value")
    plot.saveToFile(algo_name + "_bound_values")

    div_length = np.subtract(bses, ases)

    plot = PlotWrapper(0, length)
    plot.add_line(xs, div_length, "interval length", orange)
    plot.x_label("iterations")
    plot.y_label("interval length")
    plot.saveToFile(algo_name + "_bound_len")

    xs = np.logspace(0, -10)

    def get_iterations(eps):
        _, _, loc_iterations, _, _, _, _ = one_dim_extremum(func, lower, upper, eps, method, preprocess)
        return loc_iterations

    ys = list(map(get_iterations, xs))

    plot = PlotWrapper(0, len(xs), logs=True)
    plot.add_line(xs, ys, "iterations", orange)
    plot.x_label("epsilon")
    plot.y_label("iterations")
    plot.saveToFile(algo_name + "_log_eps_iters")


def plot_grafics():
    plot_grafic_for_one_dim("dichotomy", dichotomy)
    plot_grafic_for_one_dim("golden_ratio", golden_ratio)
    plot_grafic_for_one_dim("mem_fib", fibonacci_method, preprocess_for_fib)


def extremum_in_direction(f, s: List[float], delta=10e-5, eps=10e-5, method=dichotomy, preprocess=returnNone):
    def one_dim_f(a: float):
        return f([i * a for i in s])

    fa, a, iterations, _, _, _, _ = extremum_in_line(one_dim_f, 0, delta, eps, method, preprocess)
    return fa, [i * a for i in s], iterations


def func_2(x):
    return x ** 2 - 200 * x + 100


def test_line():
    fa, a, iterations, _, _, _, _ = extremum_in_line(func_2)
    print(fa, a, iterations)


# min{x^2 + y^2 + 20 x - 4 y + 10} = -94 at (x, y) = (-10, 2)
def func_3(xs):
    x = xs[0]
    y = xs[1]
    return x ** 2 + y ** 2 + 20 * x - 4 * y + 10


s = [-1, 0.2]


def test_direction():
    fa, a, iterations = extremum_in_direction(func_3, s)
    print(fa, a, iterations)


if __name__ == '__main__':
    plot_grafics()
