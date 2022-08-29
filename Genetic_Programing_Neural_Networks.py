
from EDA import *
from LayerLayout import *
from Individual import *
from Pool import *
from Selections import *
# %inline matplotlib

d = {"layer_layout": np.random.choice(HIDDEN_LAYER_COUNT)*[LayerLayout()],
"optimizer":np.random.choice(MODEL_OPTIMIZER),
"loss":np.random.choice(MODEL_LOSS)}



# Here I am testing all the method of individuals
# indiv1 = Individual(**d)
# indiv2 = Individual(**d)
# # indiv2 = deepcopy(indiv1)
# # print(indiv1)
# # print(indiv2)
# indiv1.generate_model(x_train, y_train)
# print(indiv1)
# indiv1.calculate_fitness(x_test, y_test)
# print(indiv1)
# indiv1=mutate_individual(indiv1) # working
# print("The mutation happens here")
# print(indiv1)
# print("The cross over happens here")
# indiv1, indiv2 = generate_children(indiv1, indiv2)
# print(indiv1)
# print("~~~~~")
# print(indiv2)

###############################################################################
d = {"size": SIZE,
     "optim": OPTIM,
     "x_train": x_train,
     "y_train": y_train,
     "x_test": x_test,
     "y_test": y_test}

# print("Testing the population here:")
# pool1 = Pool(**d)
# print(pool1)
# pool2 = Pool(**d)
# print(pool2)
# print("Testing evaluate here:")
# pool1.evaluate()
# print(pool1)
# pool2.evaluate()
# print(pool2)
# print("Testing evolve here:")
# pool1.evolve(True, False)
# print(pool1.best_fitness_over_generations)
# print(pool1.mean_fitness_over_generations)
# pool2.evolve(True,True)

###############################################################################


def genetic_algorithm_plot(kwargs):
    """
    The function takes in a dictionary of arguments and returns a plot of the evolution of the fitness
    of the population with and without elitism
    
    :param kwargs: a dictionary of parameters for the Pool class
    """
    pop1 = Pool(**kwargs)
    pop2 = deepcopy(pop1)
    tmp1 = pop1.evolve(False, False)
    tmp2 = pop2.evolve(True, False)

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax[0].plot(range(1, len(tmp1[2])+1), tmp1[2], '-b', label='No Elitism')
    ax[0].plot(range(1, len(tmp2[2])+1), tmp2[2], '-g', label='Elitism')

    ax[1].plot(range(1, len(tmp1[1])+1), tmp1[1], '-b', label='No Elitism')
    ax[1].plot(range(1, len(tmp2[1])+1), tmp2[1], '-g', label='Elitism')

    ax[0].grid(axis='both', linestyle='--', color="#add8e6")
    ax[0].legend(loc="best")
    ax[0].set_ylabel('Distance')
    ax[0].set_title("Mean Fitness")

    ax[1].grid(axis='both', linestyle='--', color="#add8e6")
    ax[1].legend(loc="best")
    ax[1].set_ylabel('Distance')
    ax[1].set_title("Best Fitness")

    plt.title("The evolution of the Fitness by generations")
    plt.show()


# ###############################################################################
# print("\n********** Genetic Algorithm **********")

genetic_algorithm_plot(d)

# pop = Population(**d)
# print(pop)
# pop.evaluate()
# print(pop)

# ###############################################################################
# # >>>>>> Genetic Algorithm Section <<<<<<
# # print("\n********** Genetic Algorithm **********")
# #


# # d = {"size": SIZE,
# # "optim": "max",
# # "individuals": [],
# # "mean_fitness": None,
# # "crossover_probability": CROSSOVER_PROBABILITY,
# # "mutation_probability": MUTATION_PROBABILITY,
# # "x_train": x_train,
# # "y_train": y_train,
# # "x_test": x_test,
# # "y_test": y_test}

# # population = Population(**d)
# # print(population)
# # population.evaluate()
# # print(population)
# # population.evolve(GENERATIONS, True, False)
# # print(population)
# #print(population.size)


# #def main():
# #if __name__ == "__main__":

# #    # Disable Tensorflow Warning Messages
# #    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# #    # Run Program
# #    main()
