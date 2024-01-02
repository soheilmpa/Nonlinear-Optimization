from numpy.linalg import norm
import numpy as np
from utilities import *


def newton_method(function, gradient_vector, hessian_matrix, epsilon=1e-5, method=1,
                  max_iterations=500, has_plot=False, initial_point=False, dist=1):

    if not initial_point:
        x = np.random.rand(len(gradient_vector)) * dist
    else:
        x = initial_point

    iterations = 0
    temp = [x.tolist()]

    try:
        if has_plot:

            while iterations < max_iterations:
                gradient_val = np.array(gradient_in_point(gradient_vector, x))
                hessian_val = np.array(hessian_in_point(hessian_matrix, x))
                x = x + np.linalg.solve(hessian_val, -gradient_val)
                temp.append(x.tolist())

                if np.linalg.norm(gradient_val) < epsilon:
                    break

                iterations += 1

            plot(temp, function, 'Newton')

        else:
            while iterations < max_iterations:

                gradient_val = np.array(gradient_in_point(gradient_vector, x))
                hessian_val = np.array(hessian_in_point(hessian_matrix, x))
                x = x + np.linalg.solve(hessian_val, -gradient_val)

                if np.linalg.norm(gradient_val) < epsilon:
                    break

                iterations += 1

        return iterations, x.tolist(), function(*x)

    except:
        if has_plot:
            plot(temp, function, "Newton")
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
        answer = newton_method(function, gradient, hessian)
        print(answer)