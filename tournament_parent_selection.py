import numpy as np


def generate_initial_population(population_size: int, variables: int, limits: np.ndarray) -> np.ndarray:
    # Create an initial population matrix filled with zeros
    population = np.zeros((population_size, variables))
    
    # Fill the matrix with random values within the specified limits
    for i in range(population_size):
        for j in range(variables):
            lb = limits[j][0]
            up = limits[j][1]
            population[i, j] = lb + np.random.rand() * (up - lb)
    
    return population



"""
@breaf: This function is used to calculate the aptitude of the individuals in the population using any Objective Function passed as a parameter
@context: General approach to evaluate the individuals in the population
@param population: np.ndarray -> All population to be evaluated -> F.E: [[2.3, 4.5], [1.4, -0.2]]
@param objective_function: function -> Objective function to evaluate the individuals
@return: np.ndarray -> Aptitude of the individuals in the population (Vector) [9, 5]
"""
def evaluate_population(population: np.ndarray, objective_function) -> np.ndarray:
    # get the number of individuals and variables
    n_individuals, n_var = population.shape
    # create the aptitude vector
    aptitude = np.zeros(n_individuals)
    # loop over the population
    for i in range(n_individuals):
        # get the individual
        individual = population[i]
        # evaluate the individual
        aptitude[i] = objective_function(individual)
    return aptitude


# defining the langermann function
def langermann_function(individual: np.ndarray) -> float:
    # define the constants
    a = np.array([3, 5, 2, 1, 7])
    b = np.array([5, 2, 1, 4, 9])
    c = np.array([1, 2, 5, 2, 3])
    # get the variables of the individual
    x1, x2 = individual
    # calculate the result
    result = 0
    for i in range(len(a)):
        # calculate the first term - cos
        cos_term = np.cos(np.pi * ((x1 - a[i])**2 + (x2 - b[i])**2))
        # calculate the second term - exp
        exp_term = np.exp(((x1 - a[i])**2 + (x2 - b[i])**2) / np.pi)        
        # accumulate the result dividing the cosine by the exponential and multiplying by c[i]
        result += c[i] * (cos_term / exp_term)
    # return the negative result
    return -result


"""
@brief: Selects parents using tournament selection for crossover.
@args: population (np.ndarray): The current population of individuals.
@args: aptitude (np.ndarray): The aptitude (fitness) of the individuals.
@args: num_parents (int): The number of parents to select for crossover.
@return: np.ndarray: The selected parents for crossover.
"""
def tournament_selection(population: np.ndarray, aptitude: np.ndarray) -> np.ndarray:    
    num_parents, num_variables = population.shape    
    parents = np.zeros((num_parents, num_variables))    

    for i in range(num_parents):
        # Select two random individuals
        idx1 = np.random.randint(num_parents)
        idx2 = np.random.randint(num_parents)
        # Compare the aptitude of the two individuals
        if aptitude[idx1] < aptitude[idx2]:
            parents[i] = population[idx1]
        else:
            parents[i] = population[idx2]
    
    # Return the selected parents
    return parents

if __name__ == '__main__':
    # Example usage
    population_size = 5  # Size of the population
    variables = 2        # Number of variables per individual
    limits = np.array([[1, 3],   # Limits for variable 1
                    [-1, 5]]) # Limits for variable 2

    initial_population = generate_initial_population(population_size, variables, limits)
    print(f'Population_generated: \n{initial_population}\n')
    
    # evaluate the population
    aptitude = evaluate_population(initial_population, langermann_function)
    # get the index of the minimum aptitude - BEST INDIVIDUAL
    best_index = np.argmin(aptitude)
    
    print(f'Aptitude: {aptitude}')
    print(f'Best index: {best_index}')
    print(f'Best individual: {initial_population[best_index]}')
    
    # Seleccionar padres usando el torneo    
    selected_parents = tournament_selection(initial_population, aptitude)
    print(f'\nPadres seleccionados: \n{selected_parents}')
