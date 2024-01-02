import numpy as np
from scipy.optimize import minimize

# Define the quadratic function
def quadratic_function(x):
    return x[0]**2 + x[1]**2

# Define the bounds for the variables
bounds = [(-5, 5), (-5, 5)]  # Replace this with the bounds for your variables

# Number of Monte Carlo samples
num_samples = 10000

# Generate random samples within the bounds
random_samples = np.random.uniform(low=bounds[0][0], high=bounds[0][1], size=(num_samples, len(bounds)))

# Function to minimize (negative of the objective function)
def objective_function(params):
    return -quadratic_function(params)

# Perform Monte Carlo optimization
results = minimize(objective_function, random_samples[0], bounds=bounds, method='L-BFGS-B')

# Display the results
print("Optimal parameters:", results.x)
print("Optimal objective value:", -results.fun)
