from re import findall
from sympy import sympify, diff, var, Function, sin, Abs
from scipy.optimize import minimize_scalar
from sympy.utilities.lambdify import lambdify
from numpy import linspace, linalg, array
import matplotlib.pyplot as plt
import numpy as np


def exact_line_search(x ,function, gradient):
    try:
        def objective(alpha):
            return function(*(x - alpha * gradient))
        result = minimize_scalar(objective)
        return result.x
    except:
        return [0.1]*len(gradient)


def backtracking_line_search(x, function, gradient, alpha=0.5, beta=0.8):
    try:
        step_size = 1.0
        while function(*(x - step_size * gradient)) > function(*x) + alpha * step_size * np.dot(gradient, -gradient):
            step_size *= beta
        return step_size
    except:
        return [0.1] * len(gradient)

def get_raw_function(expr):
    try:
        # UPPERCASE X
        expr = expr.upper()

        # FIND INDEXES
        indexes = list(map(int, sorted(list(set(findall("[X](.)", expr))))))
        dim = len(indexes)

        # GENERATE VARIABLES
        variables = [f"X{i}" for i in indexes]
        for variable in variables:
            exec(f"{variable} = var('{variable}')")

        # GENERATE FUNCTION
        raw_function = sympify(expr)
        function = lambdify(variables, sympify(expr))

        # GENERATE GRADIENT VECTOR
        raw_gradient = [diff(expr, eval(f"X{i}"), 1) for i in indexes]
        gradient = [lambdify(variables, sympify(f)) for f in raw_gradient]

        # GENERATE HESSIAN MATRIX
        raw_hessian = [[diff(func, eval(f"X{i}"), 1) for i in indexes] for func in raw_gradient]
        hessian = [[lambdify(variables, sympify(f)) for f in row] for row in raw_hessian]

        return dim, raw_function, function, raw_gradient, gradient, raw_hessian, hessian
    except:
        return "Something went Wrong"


def gradient_in_point(vector, point):
    return [func(*point) for func in vector]


def hessian_in_point(matrix, point):
    return [[func(*point) for func in vector] for vector in matrix]


def is_convex(gradient, initial_point, epsilon=1e-10, max_iter=100):
    current_point = initial_point
    for _ in range(max_iter):
        # Update the current point in the direction of the negative gradient.
        new_point = current_point - epsilon * array(gradient_in_point(gradient, current_point))

        # If the new point is close to the current point, then the function has converged.
        if linalg.norm(new_point - current_point) < epsilon:
            return True

        current_point = new_point

    return False

def plot(points, function, name):
    x1_values = np.linspace(-10, 10, 50)
    x2_values = np.linspace(-10, 10, 50)
    x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)
    y_values = function(x1_grid, x2_grid)
    plt.figure(figsize=(4, 4))
    plt.contourf(x1_grid, x2_grid, y_values, cmap='viridis', levels=50)
    x_points = np.array([point[0] for point in points])
    y_points = np.array([point[1] for point in points])
    c_values = function(x_points, y_points)
    plt.plot(x_points, y_points, color='red', linestyle='-', linewidth=2, markersize=7)
    scatter = plt.scatter(x_points, y_points, c=c_values)
    plt.colorbar(scatter)
    plot_path = f"static/images/{name} 1.png"
    plt.savefig(plot_path, bbox_inches='tight')

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1_grid, x2_grid, y_values, cmap='viridis', rstride=5, cstride=5, alpha=0.8)
    plt.axis('off')
    plot_path = f"static/images/{name} 2.png"
    plt.savefig(plot_path, bbox_inches='tight')


if __name__ == "__main__":
    TESTS = {
        0: "X1^2 - X1*X2 + X2^3",
        1: "-X1^2 + X1*X2 - X2^3",
        3: "(X1-1)^2 + 5",
        4: "X1^3 - X2^2",
        5: "(X1^2 + 1)^0.5 + (X2^2 + 1)^0.5",
        6: "x0^2 - X0*X2 + X2^3 - X3",
        7: "X3**2",
        8: "sin(X1)+X2",
        9: "abs(X1)",
        10: "e^(X1+X2)",
        # 11: "X1**2 - X1*X2 + X2**3",
    }

    d, rf, f, rg, g, rh, h = get_raw_function(TESTS[0])
    print(d, rf, f, rg, g, rh, h, sep='\n\n')
    p = (-2, ) * d

    print(f"Function in {p} :\n\t {f(*p)}")
    # print(f"Function is convex : {is_convex(g, p)}")

    print('---------------------------------------------------')

    print(f"Gradient vector :\n\t {rg}")

    print(f"\nGradient in {p} :\n\t {gradient_in_point(g, p)}")

    print('---------------------------------------------------')

    print(f"Hessian matrix :\n\t {rh}")

    print(f"\nHessian in {p} :\n\t {hessian_in_point(h, p)}")
