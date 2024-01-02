from flask import Flask, render_template, request
from utilities import get_raw_function
from algorithms.gradient_method import gradient_method
from algorithms.newton_method import newton_method
from algorithms.bfgs import bfgs_method

app = Flask(__name__)

dist = 8


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/calculate', methods=['POST'])
def calculate():

    function = request.form.get('textInput')
    algorithms = tuple(filter(lambda x: x!=None, [request.form.get(f'checkbox{i}') for i in range(8)]))

    answers = {}

    try:
        if algorithms and (temp := get_raw_function(function)):
            print(algorithms)
            dim, target, function, _, gradient, _, hessian = temp
            has_plot = True if dim == 2 else False

            if "Gradient Descent (Constant)" in algorithms:
                answer = gradient_method(function, gradient, has_plot=has_plot, descent_method=1, dist=dist)
                answers["Gradient Descent (Constant)"] = answer

            if "Gradient Descent (line search)" in algorithms:
                answer = gradient_method(function, gradient, has_plot=has_plot, descent_method=2, dist=dist)
                answers["Gradient Descent (line search)"] = answer

            if "Gradient Descent (backtracking)" in algorithms:
                answer = gradient_method(function, gradient, has_plot=has_plot, descent_method=3, dist=dist)
                answers["Gradient Descent (backtracking)"] = answer

            if "Newton" in algorithms:
                answer = newton_method(function, gradient, hessian, has_plot=has_plot, dist=dist)
                answers["Newton"] = answer

            if "BFGS" in algorithms:
                answer = bfgs_method(function, gradient, hessian, has_plot=has_plot, dist=dist)
                answers["BFGS"] = answer

            if "Genetic Algorithm" in algorithms:
                ...

            if "Monte Carlo" in algorithms:
                ...

            return render_template('answer.html', target=target, answers=answers, has_plot=has_plot)
        else:
            return render_template('error.html')
    except:
        return render_template('error.html')


if __name__ == '__main__':
    app.run()
