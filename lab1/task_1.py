import argparse

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
import numpy as np
from scipy.integrate import solve_ivp


def f(x, y):
    return y**2 - 3*y - x + 2


def d2y_dx2(x, y):
    return -1 + (2*y - 3)*f(x, y)


def leave_plot(x, y):
    return 5.65 - abs(y[0])


leave_plot.terminal = True


def build_graph():
    fig, ax = plt.subplots(figsize=(10.2, 6.5))
    fig.subplots_adjust(left=0.08, right=0.72, bottom=0.12, top=0.91)
    x_min, x_max, y_min, y_max = -6.35, 6.35, -5.35, 5.35
    y = np.unique(np.r_[np.linspace(y_min, y_max, 1000), 1.5])
    x_null = y**2 - 3*y + 2
    boundary = np.clip(x_null, x_min, x_max)
    growing = ax.fill_betweenx(y, x_min, boundary, color="#d7ead9",
                               label=r"Зростання, $y'>0$")
    falling = ax.fill_betweenx(y, boundary, x_max, color="#fbe9df",
                               label=r"Спадання, $y'<0$")

    Xg, Yg = np.meshgrid(np.linspace(x_min, x_max, 600),
                         np.linspace(y_min, y_max, 500))
    curvature = d2y_dx2(Xg, Yg)
    with plt.rc_context({"hatch.color": "#a9aaa7", "hatch.linewidth": 0.45}):
        ax.contourf(Xg, Yg, curvature, levels=[-2000, 0, 2000],
                    colors="none", hatches=["//", "xx"])

    # Відрізки довжини 0.32 з центрами у вузлах (m, n/2).
    X, Y = np.meshgrid(np.arange(-6, 7), np.arange(-10, 11)/2)
    slope = f(X, Y)
    half_width = 0.16/np.sqrt(1 + slope**2)
    centers = np.column_stack((X.ravel(), Y.ravel()))
    offsets = np.column_stack((half_width.ravel(), (slope*half_width).ravel()))
    field = LineCollection(np.stack((centers-offsets, centers+offsets), axis=1),
                           colors="#65706e", linewidths=0.9, zorder=3,
                           label="Поле напрямків")
    ax.add_collection(field)

    positive = negative = None
    for k in (-1.5, -1, -0.5, 0.5, 1, 1.5):
        color = "#986518" if k > 0 else "#657c96"
        line, = ax.plot(x_null-k, y, "--", color=color, lw=1)
        if k > 0:
            positive = line
        else:
            negative = line
    positive.set_label(r"Ізокліни $k=1/2,,1,,3/2$")
    negative.set_label(r"Ізокліни $k=-1/2,,-1,,-3/2$")
    maximum, = ax.plot(x_null, y, "k-", lw=2.1, label="Максимуми, k = 0")

    inflection_handles = []
    for branch in (np.linspace(y_min, 1.499, 1200), np.linspace(1.501, y_max, 1200)):
        x_inf = branch**2 - 3*branch + 2 - 1/(2*branch-3)
        line, = ax.plot(x_inf, branch, color="#be244f", lw=2.1,
                        label="Точки перегину")
        inflection_handles.append(line)

    curves = []
    options = dict(events=leave_plot, rtol=1e-9, atol=1e-11, max_step=0.025)
    for y0, color in [(-2, "#2457ae"), (0, "#9558a2"), (1, "#087e7b")]:
        left = solve_ivp(f, (0, x_min), [y0], **options)
        right = solve_ivp(f, (0, x_max), [y0], **options)
        if not left.success or not right.success:
            raise RuntimeError(f"Не вдалося побудувати криву для y(0) = {y0}.")
        x = np.r_[left.t[::-1], right.t[1:]]
        y_curve = np.r_[left.y[0, ::-1], right.y[0, 1:]]
        line, = ax.plot(x, y_curve, color=color, lw=2.4, zorder=5,
                        label=fr"$y(0)={y0}$")
        curves.append(line)
        ax.plot(0, y0, "o", color=color, markeredgecolor="white", ms=6, zorder=6)

    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max), xlabel="x", ylabel="y",
           xticks=np.arange(-6, 7), yticks=np.arange(-5, 6),
           title=r"Задача 1.  $y'=y^2-3y-x+2$")
    ax.set_aspect("equal")
    ax.axvline(0, color="#727777", lw=0.7, zorder=1)
    ax.axhline(0, color="#727777", lw=0.7, zorder=1)
    ax.grid(True, color="#777777", alpha=0.19, lw=0.55)
    up = Patch(facecolor="white", edgecolor="#a9aaa7", hatch="//",
               label=r"Опуклість вгору, $y''<0$")
    down = Patch(facecolor="white", edgecolor="#a9aaa7", hatch="xx",
                 label=r"Опуклість вниз, $y''>0$")
    ax.legend(handles=[field, positive, negative, growing, falling, maximum,
                       up, down, inflection_handles[0], *curves],
              loc="center left", bbox_to_anchor=(1.015, 0.5), frameon=False,
              borderaxespad=0, fontsize=9, labelspacing=0.9)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", help="Зберегти графік за вказаним шляхом.")
    args = parser.parse_args()

    print("Ізокліни: x = y^2 - 3y + 2 - k.")
    print("У точках x = y^2 - 3y + 2 маємо y' = 0 і y'' = -1. Це максимуми.")
    print("Точки перегину: x = y^2 - 3y + 2 - 1/(2y-3), y ≠ 3/2.")
    figure = build_graph()
    if args.save:
        figure.savefig(args.save, dpi=240)
    else:
        plt.show()
