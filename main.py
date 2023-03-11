# Connor Luckett

import numpy as np
from dataclasses import dataclass


@dataclass
class NACA4Digit:
    # NPXX
    pass


@dataclass
class Riblett:
    pass


def thickness_distribution(x: np.ndarray, t: float):
    coefficients = [0.2969, -0.1260, -0.3516, -0.2843, -0.1015]
    powers = [1 / 2, 0, 1, 2, 3, 4]
    terms = np.power(x, powers)
    return 5 * t * np.sum(coefficients * terms)


def main():
    chord: float = 54
    accuracy: float = 1
    camber: float = 4  # %
    thickness: float = 15
    max_camber_position: float = 40
    m: float = camber / 100  # Maximum camber divided by 100
    p: float = max_camber_position / 10  # Position of the maximum camber divided by 10
    xx: float = thickness / 100  # Thickness divided by 100

    # Front (0 <= x <= p), Back (p <= x <= 1)
    x_front = np.arange(start=0, stop=p, step=accuracy, dtype=float)
    x_back = np.arange(start=p, stop=100, step=accuracy, dtype=float)
    x = np.concatenate((x_front, x_back))
    x_c = chord * x
    y_t = thickness_distribution(x, xx)

    y_c_front = m / p ^ 2 * (2 * p * x_front - np.power(x_front, 2))
    gradient_front = 2 * m / p ^ 2 * (p - x_front)
    y_c_back = m / (1 - p) ^ 2 * (1 - 2 * p + 2 * p * x_back - np.power(x_front, 2))
    gradient_back = 2 * m / (1 - p) ^ 2 * (p - x_back)
    thetas_front = np.arctan(gradient_front)
    thetas_back = np.arctan(gradient_back)

    thetas = np.concatenate((thetas_front + thetas_back))
    y_c = y_c_front + y_c_back

    x_u = x - y_t * np.sin(thetas)
    x_u = y_c + y_t * np.cos(thetas)
    x_l = x + y_t * np.sin(thetas)
    y_l = y_c - y_t * np.cos(thetas)


if __name__ == "__main__":
    main()
