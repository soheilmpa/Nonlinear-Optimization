import random
from math import sqrt
from deap import base, creator, tools, algorithms


def objective_function(individual):
    x1, x2  = individual
    # return sqrt(x1**2 + 1) + sqrt(x2**2 + 1),
    return x1**20 + x2**20,
    # return x1**3 + x2**3,


# Define the problem type (minimization)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Genetic Algorithm parameters
population_size = 50
num_generations = 3
crossover_prob = 0.8
mutation_prob = 0.2

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -5, 5)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", objective_function)

def main():
    # Initialize population
    population = toolbox.population(n=population_size)

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Crossover and mutation
    algorithms.eaMuPlusLambda(population, toolbox, mu=population_size, lambda_=2*population_size,
                              cxpb=crossover_prob, mutpb=mutation_prob, ngen=num_generations,
                              stats=None, halloffame=None, verbose=True)

    # Print the best individual after optimization
    best_individual = tools.selBest(population, k=1)[0]
    print("generations:", num_generations)
    print("Best individual:", best_individual)
    print("Best fitness:", best_individual.fitness.values[0])

if __name__ == "__main__":
    main()
