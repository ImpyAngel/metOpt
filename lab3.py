import math
from typing import Tuple, List
import numpy as np
import pandas as pd


class Data_Saver:
    file = "texts/lab3/table.csv"
    u_0 = []
    accuracy = []
    iteration = []
    algo = []
    grad = []
    arg_min = []
    value = []

    def add(self, u_0, accuracy, iteration, algo, grad, arg_min, value):
        self.u_0.append(u_0)
        self.accuracy.append(accuracy)
        self.iteration.append(iteration)
        self.algo.append(algo)
        self.grad.append(grad)
        self.arg_min.append(arg_min)
        self.value.append(value)

    def serialize(self):
        df = pd.DataFrame({
            "u_0": self.u_0,
            "accuracy": self.accuracy,
            "iteration": self.iteration,
            "algo": self.algo,
            "grad": self.grad,
            "arg_min": self.arg_min,
            "value": self.value
        })
        df.to_csv(self.file)

def func(u: List[float]) -> float:
    x = u[0]
    y = u[1]
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


u_0 = [4, 3]


def true_grad(u) -> List[float]:
    x = u[0]
    y = u[1]
    return [2 * (-7 + x + y ** 2 + 2 * x * (-11 + x ** 2 + y)), 2 * (-11 + x ** 2 + y + 2 * y * (-7 + x + y ** 2))]


def true_hessian(u: List[float]) -> float:
    x = u[0]
    y = u[1]
    return 4 * (12 * x ** 3 + x ** 2 * (36 * y ** 2 - 82) - 2 * x * (
            2 * y + 21) + 12 * y ** 3 - 130 * y ** 2 - 26 * y + 273)


# central derivative
def calc_grad(f, u, h):
    x = u[0]
    y = u[1]
    df_dx = (f([x + h, y]) - f([x - h, y])) / (2 * h)
    df_dy = (f([x, y + h]) - f([x, y - h])) / (2 * h)
    return df_dx, df_dy


def calc_hessian(f, u, h):
    x = u[0]
    y = u[1]
    df_dx2 = (f([x + 2 * h, y]) - 2 * f([x + h, y]) + f([x, y])) / (h ** 2)
    df_dy2 = (f([x, y + 2 * h]) - 2 * f([x, y + h]) + f([x, y])) / (h ** 2)
    df_dyx = (f([x + h, y + h]) - f([x, y + h]) - f([x + h, y]) + f([x, y])) / (h ** 2)
    return df_dx2 * df_dy2 - df_dyx ** 2


def newton_method(grad, hessian, eps, x_0) -> Tuple[float, int]:
    x = x_0
    i = 0
    while np.linalg.norm(grad(x)) > eps:
        # print(x)
        i += 1
        x = np.subtract(x, np.divide(grad(x), hessian(x)))
    return x, i


def run_newton_with_form(algorithm, f, grad, hessian, eps, data_saver: Data_Saver):
    ans, iterations = newton_method(grad, hessian, eps, u_0)
    data_saver.add(u_0, eps, iterations, algorithm, grad(ans), ans, f(ans))


def run_true_newton(eps, data_saver):
    run_newton_with_form("true", func, true_grad, true_hessian, eps, data_saver)


def run_calc_newton(eps, data_saver):
    accuracy = 1e-3

    def grad(u):
        return calc_grad(func, u, accuracy)

    def hessian(u):
        return calc_hessian(func, u, accuracy)

    run_newton_with_form("calc", func, grad, hessian, eps, data_saver)


def run_all():
    data_saver = Data_Saver()
    run_true_newton(1e-1, data_saver)
    run_true_newton(1e-3, data_saver)
    run_true_newton(1e-5, data_saver)

    run_calc_newton(1e-1, data_saver)
    run_calc_newton(1e-3, data_saver)
    run_calc_newton(1e-5, data_saver)
    data_saver.serialize()


if __name__ == '__main__':
    run_all()
