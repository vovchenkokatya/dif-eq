"""Лабораторна робота 1, варіант 4. Задача 2."""
import argparse
from math import exp, log, sqrt

import matplotlib.pyplot as plt
import numpy as np


def g(y):
    return (y + 3)*(y - 2)


def maximal_interval(y0):
    if -3 <= y0 <= 2:
        return -np.inf, np.inf

    x_star = log((y0 + 3)/(y0 - 2))/5
    if y0 < -3:
        return x_star, np.inf
    return -np.inf, x_star


def exact_solution(x, y0):
    x = np.asarray(x, dtype=float)
    if y0 in (-3, 2):
        return np.full_like(x, y0)

    left, right = maximal_interval(y0)
    valid = (x > left) & (x < right)
    values = np.full_like(x, np.nan)
    # Еквівалентні формули зі знаменником без великих експонент.
    nonnegative = valid & (x >= 0)
    negative = valid & (x < 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        values[nonnegative] = -3 + 5*(y0+3)*np.exp(-5*x[nonnegative]) / (
            (y0+3)*np.exp(-5*x[nonnegative]) - (y0-2))
        values[negative] = 2 + 5*(y0-2)*np.exp(5*x[negative]) / (
            (y0+3) - (y0-2)*np.exp(5*x[negative]))
    values[~np.isfinite(values)] = np.nan
    return values


def difference(x):
    z = exp(-5*x)
    return 100*z/((z+9)*(11-z))


X_STAR = log((99998 + sqrt(99998**2 + 396))/198)/5


def build_graph():
    fig = plt.figure(figsize=(10.4, 6.8), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[0.45, 1])
    field_ax = fig.add_subplot(grid[0, :])
    curves_ax = fig.add_subplot(grid[1, 0])
    limit_ax = fig.add_subplot(grid[1, 1])

    # Векторне поле на прямій Oy, з масштабом 0.05 для довжин.
    points = np.arange(-4, 5)
    vectors = g(points)
    field_ax.axhline(0, color="#676d73", lw=0.9)
    field_ax.quiver(points, np.zeros_like(points), 0.05*vectors,
                    np.zeros_like(points), angles="xy", scale_units="xy",
                    scale=1, width=0.006, color="#087e7b")
    field_ax.scatter(points, np.zeros_like(points), color="#087e7b", s=13)
    field_ax.scatter([-3, 2], [0, 0], color="#171d24", s=35, zorder=5)
    for y, vector in zip(points, vectors):
        field_ax.annotate(str(vector), (y, 0), xytext=(0, 9),
                          textcoords="offset points", ha="center", fontsize=9)
    field_ax.set(xlim=(-4.5, 5), ylim=(-0.65, 0.8), xticks=points, yticks=[],
                 xlabel="y", title="1. Векторне поле на прямій Oy")
    field_ax.text(0.015, 0.88, r"Над вузлом вказано $g(y)$",
                  transform=field_ax.transAxes, fontsize=9)
    for side in ("top", "right", "left"):
        field_ax.spines[side].set_visible(False)
    field_ax.spines["bottom"].set_position(("data", -0.2))

    x = np.linspace(-1.1, 1.1, 2001)
    # По одному прикладу з кожної області та обидві рівноваги.
    for y0, color in [(-4, "#2457ae"), (0, "#087e7b"), (3, "#be244f")]:
        curves_ax.plot(x, exact_solution(x, y0), color=color, lw=2,
                       label=fr"$y_0={y0}$")
        curves_ax.plot(0, y0, "o", color=color, ms=4)
        left, right = maximal_interval(y0)
        if np.isfinite(left) or np.isfinite(right):
            x_asymptote = left if np.isfinite(left) else right
            curves_ax.axvline(x_asymptote, color=color, ls=":", lw=1.25)
    for y0 in (-3, 2):
        curves_ax.axhline(y0, color="#303840", ls="--", lw=1.1,
                          label=fr"$y={y0}$")
    curves_ax.axhline(-0.5, color="#85579c", ls="-.", lw=1,
                      label=r"Перегин, $y=-1/2$")
    curves_ax.set(xlim=(-1.1, 1.1), ylim=(-8, 8), xlabel="x", ylabel="y",
                  title="4. Інтегральні криві")
    curves_ax.legend(loc="upper left", fontsize=8, ncol=2)

    x = np.linspace(0, 2.2, 1000)
    limit_ax.plot(x, exact_solution(x, -2.5), color="#2457ae", lw=2,
                  label=r"$y(x,-2.5)$")
    limit_ax.plot(x, exact_solution(x, -3.5), color="#9558a2", lw=2,
                  label=r"$y(x,-3.5)$")
    limit_ax.axhline(-3, color="#303840", ls="--", lw=1, label=r"$y=-3$")
    limit_ax.axvline(X_STAR, color="#986518", ls=":", lw=1.6,
                     label=fr"Поріг $x_*={X_STAR:.5f}$")
    limit_ax.set(xlim=(0, 2.2), ylim=(-3.57, -2.43), xlabel="x", ylabel="y",
                 title="5. Зближення двох розв’язків")
    limit_ax.legend(loc="upper right", fontsize=8)
    for ax in (curves_ax, limit_ax):
        ax.grid(True, ls=":", alpha=0.5)

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", help="Зберегти графіки за вказаним шляхом.")
    args = parser.parse_args()

    print("Рівноваги: y = -3 (стійка) і y = 2 (нестійка).")
    print("Зростання: y < -3 або y > 2. Спадання: -3 < y < 2.")
    print("Перегин: y = -1/2.")
    print("Різниця менша за 0.001 при x > {:.10f}.".format(X_STAR))
    print("Наприклад, при x = 1.4 різниця ≈ {:.9f}.".format(difference(1.4)))
    figure = build_graph()
    if args.save:
        figure.savefig(args.save, dpi=240)
    else:
        plt.show()