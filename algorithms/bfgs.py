from numpy.linalg import norm
import numpy as np
from utilities import *


def bfgs_method(function, gradient_vector, hessian_matrix, dist=1,
                max_iterations=500, has_plot=False, initial_point=False):

    if not initial_point:
        x = np.random.rand(len(gradient_vector)) * dist
    else:
        x = initial_point

    iterations = 0
    temp = [x.tolist()]
    H_inv = np.linalg.inv(hessian_in_point(hessian_matrix, x))

    try:
        if has_plot:
            while iterations < max_iterations:
                gradient_val = gradient_in_point(gradient_vector, x)
                step = -np.dot(H_inv, gradient_val)
                alpha = 1.0
                while function(*(x + alpha * step)) > function(*x) + 0.5 * alpha * np.dot(gradient_val, step):
                    alpha *= 0.5
                x_new = x + alpha * step
                s = x_new - x
                y = np.array(gradient_in_point(gradient_vector, x_new)) - np.array(gradient_val)
                rho = 1.0 / (np.dot(y, s) + 1e-10)
                A = np.eye(len(x)) - rho * np.outer(s, y)
                B = np.eye(len(x)) - rho * np.outer(y, s)
                H_inv = np.dot(np.dot(A, H_inv), B) + rho * np.outer(s, s)
                x = x_new

                temp.append(x.tolist())

                if np.linalg.norm(gradient_val) < 1e-6:
                    break

                iterations += 1

            plot(temp, function, 'BFGS')
        else:
            while iterations < max_iterations:

                gradient_val = gradient_in_point(gradient_vector, x)
                step = -np.dot(H_inv, gradient_val)
                alpha = 1.0
                while function(*(x + alpha * step)) > function(*x) + 0.5 * alpha * np.dot(gradient_val, step):
                    alpha *= 0.5
                x_new = x + alpha * step
                s = x_new - x
                y = np.array(gradient_in_point(gradient_vector, x_new)) - np.array(gradient_val)
                rho = 1.0 / (np.dot(y, s) + 1e-10)
                A = np.eye(len(x)) - rho * np.outer(s, y)
                B = np.eye(len(x)) - rho * np.outer(y, s)
                H_inv = np.dot(np.dot(A, H_inv), B) + rho * np.outer(s, s)
                x = x_new

                if np.linalg.norm(gradient_val) < 1e-6:
                    break

                iterations += 1

        return iterations, x.tolist(), function(*x)
    except:
        if has_plot:
            plot(temp, function, "BFGS")
        return 'could not reach max iterations', x.tolist(), function(*x)

if __name__ == "__main__":
    TESTS = {
        # "X1^2 - X1*X2 + X2^3",
        # "-X1^2 + X1*X2 - X2^3",
        # "(X1-1)^2 + 5",
        # "100*X1^4 + 0.01*x2^4",
        # "X1^2 + X2^2",
        # "X1^3 - X2^2",
        "(X1^2 + 1)^0.5 + (X2^2 + 1)^0.5",
    }

    for test in TESTS:
        _, _, function, _, gradient, _, hessian = get_raw_function(test)
        answer = bfgs_method(function, gradient, hessian)
        print(answer)