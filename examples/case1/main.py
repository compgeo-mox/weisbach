import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.special import lambertw
from scipy.optimize import fsolve

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("font", size=32.5)

params = {
    "text.latex.preamble": r"\usepackage{bm,amsmath,siunitx}\newcommand{\sib}[1]{[\si{#1}]}"
}
plt.rcParams.update(params)
mpl.rcParams["axes.linewidth"] = 1.5


def lambda_smooth(u, alpha, beta1, beta2, u_bar):
    if u < u_bar:
        return alpha * u
    else:
        return beta1 * np.square(u / lambertw(beta2 * u).real)


def lambda_forch(u, alpha, beta1, beta2, u_bar):
    if u < u_bar:
        return alpha * u
    else:
        return beta1 * np.square(u / lambertw(beta2 * u_bar).real)


def f_s(Re, Re_bar):
    if Re < Re_bar:
        return 64 / Re
    else:
        return 1 / np.square(0.838 * lambertw(0.629 * Re).real)


def Re_r(f_r, Re, eps, d):
    return np.sqrt(f_r / 8) * Re * eps / d


def solve_fr(f_r, Re, eps, d):
    Re_r_val = Re_r(f_r, Re, eps, d)
    sqrt_f_r = np.sqrt(f_r)

    c_1 = 1.93  # 2 1.93
    c_2 = 1.9  # 2.51 1.9
    c_3 = 0.34  # 0.305 0.34

    return 1 / sqrt_f_r + c_1 * np.log10(
        c_2 * (1 + c_3 * Re_r_val * np.exp(-11 / Re_r_val)) / (Re * sqrt_f_r)
    )


def f_r(Re, Re_bar, eps, d, f_r_val_start=0.01):
    if Re < Re_bar:
        return 64 / Re
    else:
        solve = lambda f_r: solve_fr(f_r, Re, eps, d)
        return np.atleast_1d(fsolve(solve, f_r_val_start))[0]


def main():
    # physical parameters
    mu = 0.0010016  # Pa.s at 20C
    rho = 1000  # kg/m^3
    d = 1e-2  # m
    Re_bar = 3000

    # Parameters
    alpha = 32 * mu / d**2
    beta1 = rho / (2 * d * 0.838**2)
    beta2 = 0.629 * rho * d / mu
    u_bar = Re_bar * mu / (rho * d)

    print(alpha, beta1, beta2, u_bar)

    # Plot lambda vs u
    N = 10000
    u_max = 100
    u_min = 0.1
    u = np.linspace(u_min, u_max, N)
    if False:
        lambda_val = np.array(
            [lambda_smooth(u_val, alpha, beta1, beta2, u_bar) for u_val in u]
        )

        lambda_forch_val = np.array(
            [lambda_forch(u_val, alpha, beta1, beta2, u_bar) for u_val in u]
        )

        size = 10
        fig, ax = plt.subplots(figsize=(size * 1.5, size))

        ax.plot(u, lambda_val)
        ax.plot(u, lambda_forch_val, "--")
        ax.set_xlabel("$u \sib{\meter\per\second}$")
        ax.set_ylabel("$\lambda \sib{\pascal\per\meter}$")
        plt.show()

        # Save plot
        folder = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(folder, "lambda.pdf")
        fig.savefig(filename, bbox_inches="tight")
        plt.gcf().clear()

        # os.system("pdfcrop --margins '0 -750 0 0' " + filename + " " + filename)
        os.system("pdfcrop " + filename + " " + filename)

    if True:
        # Plot f vs Re
        Re = rho * u * d / mu

        print(np.min(Re), np.max(Re))
        f_s_val = np.array([f_s(Re_val, Re_bar) for Re_val in Re])

        size = 10
        fig, ax = plt.subplots(figsize=(size * 1.75, size))

        ax.loglog(Re, f_s_val, "--", label=f"laminar")

        epss = np.array([1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2, 3.3e-2]) * d
        for eps in epss:
            f_r_val = np.array([f_r(Re_val, Re_bar, eps, d) for Re_val in Re])
            ax.loglog(Re, f_r_val, label=f"$\\varepsilon/d = {eps/d:.2e}$")

        f_range = [0.01, 0.02, 0.04, 0.06, 0.08]
        f_range_label = [f"{f:.2f}" for f in f_range]
        ax.yaxis.set(
            ticks=f_range, ticklabels=f_range_label, label_text="$f \sib{\cdot}$"
        )

        ax.set_xlabel("$Re \sib{\cdot}$")

        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.175), ncol=3)
        plt.grid(color="gray", alpha=0.5, linestyle="--", linewidth=0.7)
        plt.tight_layout()  # Adjust layout to accommodate the legend

        plt.show()

        # # Save plot
        # folder = os.path.dirname(os.path.abspath(__file__))
        # filename = os.path.join(folder, "friction.pdf")
        # fig.savefig(filename, bbox_inches="tight")
        # plt.gcf().clear()

        # # os.system("pdfcrop --margins '0 -750 0 0' " + filename + " " + filename)
        # os.system("pdfcrop " + filename + " " + filename)


if __name__ == "__main__":
    main()
