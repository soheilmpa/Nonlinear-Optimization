from numpy.linalg import norm
from utilities import *


def gradient_method(function, gradient_vector, initial_point=False, has_plot=False,
                    descent_method=1, tolerance=1e-6, stepsize=0.1, max_iterations=500, dist=1):
    if not initial_point:
        x = np.random.rand(len(gradient_vector)) * dist
    else:
        x = initial_point

    iterations = 0
    temp = [x.tolist()]

    try:
        if has_plot:
            while iterations < max_iterations:
                gradient = np.array(gradient_in_point(gradient_vector, x))

                if np.linalg.norm(gradient) < tolerance:
                    break  # Convergence check

                if descent_method == 1:
                    direction = -gradient * stepsize
                    name = "Gradient Descent (Constant)"
                elif descent_method == 2:
                    direction = -gradient * exact_line_search(x ,function, gradient)
                    name = "Gradient Descent (line search)"
                else:
                    direction = -gradient * backtracking_line_search(x ,function, gradient)
                    name = "Gradient Descent (backtracking)"

                x = x + direction
                temp.append(x.tolist())
                iterations += 1

            plot(temp, function, name)

        else:

            while iterations < max_iterations:
                gradient = np.array(gradient_in_point(gradient_vector, x))

                if np.linalg.norm(gradient) < tolerance:
                    break  # Convergence check

                if descent_method == 1:
                    direction = -gradient * stepsize
                elif descent_method == 2:
                    direction = -gradient * exact_line_search(x ,function, gradient)
                else:
                    direction = -gradient * backtracking_line_search(x ,function, gradient)

                x = x + direction
                iterations += 1

        return iterations, x.tolist(), function(*x)

    except:
        if has_plot:
            plot(temp, function, name)
        return 'could not reach max iterations', x.tolist(), function(*x)


if __name__ == "__main__":
    TESTS = {
        # "X1^2 - X1*X2 + X2^3",
        # "-X1^2 + X1*X2 - X2^3",
        # "(X1-1)^2 + 5",
        # "X1^3 + X2^3",
        "X1^10 + X2^10",
        # "(x1^2)^0.5 + (x2^2)^0.5",
        # "X1^2 + X2^2 + X3^2",
        # "(X1^2 + 1)^0.5 + (X2^2 + 1)^0.5",
    }

    for test in TESTS:
        _, _, f, _, g, _, _ = get_raw_function(test)
        answer = gradient_method(f, g, descent_method=2)
        print(answer)
