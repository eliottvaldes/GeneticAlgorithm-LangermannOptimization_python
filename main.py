
# import libraries
import numpy as np
# import personal functions
from generate_population import generate_initial_population
from calculate_aptitude import evaluate_population, langermann_function
from tournament_parent_selection import tournament_selection
from sbx_crossover import sbx
from polynomial_mutation import apply_polinomial_mutation


"""
@brief: Function to solve the Langermann function using a genetic algorithm
@context: This function run the full GA process using the independent functions created before
@param generations: int -> Number of generations to run the GA. Default value is 200
@return: dict -> Best individual found by the GA and its aptitude
"""
def solve_langermann_function(generations: int = 200) -> dict:            
    # ALGORITHM EXECUTION
    # call function generate population
    population = generate_initial_population(population_size, variables, limits)
    
    # iterate over the number of generations
    for i in range(generations):
        # calculate aptitude_vector    
        aptitude = evaluate_population(population, langermann_function)
        
        # save the best individual in a variable use a copy
        best_individual = population[np.argmin(aptitude)].copy()
        
        # use parent tournament selection
        population = tournament_selection(population, aptitude)
        
        # apply sbx crossover
        sbx(population, limits, sbx_prob, sbx_dispersion_param)
        # apply mutation
        apply_polinomial_mutation(population, limits, mutation_probability_param, distribution_index_param)
        
        # calculate the new aptitude vector for the population after crossover
        aptitude = evaluate_population(population, langermann_function)
        # get the worst individual index = max(aptitude)
        worst_after_crossover = np.argmax(aptitude)
        
        # ELITISM SUSTITUTION
        # replace the worst individual after crossover (children) with the best individual before crossover (parent)
        population[worst_after_crossover] = best_individual
    
    # get the final aptitude vector
    aptitude = evaluate_population(population, langermann_function)
    
    # get the best individual index = min(aptitude)
    best_individual_index = np.argmin(aptitude)
    
    
    return {'individual': population[best_individual_index], 'aptitude': aptitude[best_individual_index]}




# ===============================================================
#GENERAL CONFIGURATIONS
# ===============================================================
generations = 200 #ng -> Number of generationsb
population_size = 100 #np -> Size of the population
variables = 2 #nVar -> Number of variables of each individual
limits = np.array([[0, 10], #limits var 1 -> [Lower, Upper]
                [0, 10]]) #limits var 2 -> [Lower, Upper]    

# ===============================================================
#SBX CONFIGURATIONS
# ===============================================================
sbx_prob = 0.9 #pc -> Probability of crossover
sbx_dispersion_param = 2 #nc -> Distribution index (ideal 1 or 2)    

# ===============================================================
# POLINOMIAL MUTATION CONFIGURATIONS
# ===============================================================
mutation_probability_param = 0.7 #r -> Probability of mutation
distribution_index_param = 50 #nm -> Distribution index ( ideal 20-100)        

# Show the configurations
print(f'{"*"*50}')
print('Algorithm configurations:')
print(f'Number of generations: {generations}')
print(f'Population size: {population_size}')
print(f'Number of variables: {variables}')
print(f'Limits: \n{limits}')
print(f'SBX Probability (pc): {sbx_prob}')
print(f'SBX Dispersion Parameter (nc): {sbx_dispersion_param}')
print(f'Mutation Probability: {mutation_probability_param}')
print(f'Distribution Index (nm): {distribution_index_param}')


# ===============================================================
# EXECUTION OF THE MAIN FUNCTION
# ===============================================================
print(f'\n{"*"*50}')
print('Running Algorithm...')
results = [] # list of dictionaries to store the results
for i in range(10):
    # show the current execution, the results of function and then add the results to the list
    print(f'Execution {i+1}')
    result = solve_langermann_function(generations)
    print(f'\tIndividual: {result["individual"]}, Aptitude = {result["aptitude"]}')
    results.append(result)
    

# only pass the results.aptitude of the results dictionaries
partial_results = np.array([result['aptitude'] for result in results])
# get the best, median, worst and standard deviation of the results
best = np.min(partial_results)
median = np.median(partial_results)
worst = np.max(partial_results)
std = np.std(partial_results)

print(f'\n{"*"*50}')
print(f'Statistics:')
print(f'Best: {best}')
print(f'Median: {median}')
print(f'Worst: {worst}')
print(f'Standard deviation: {std}')

    