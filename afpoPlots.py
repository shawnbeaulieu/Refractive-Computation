import pickle
import matplotlib.pyplot as plt
from switch_float import switch
import constants as c
import numpy

runs = c.RUNS
gens = c.numGenerations

# plot the pareto front
# plot the avg vs generation plots

def paretoFront(population):
    pareto_front = []

    for i in population:
        i_is_dominated = False
        for j in population:
            if i != j:
                if population[j].dominates(population[i]):
                    i_is_dominated = True
                    break
        if not i_is_dominated:
            pareto_front.append(population[i])

    return pareto_front

with open('gens/gen499.dat', "rb") as f:
    for r in range(1, runs+1):
        # population of the last generation
        temp = pickle.load(f)
        pf = paretoFront(temp)
        print("pf size: "+str(len(pf)))

        plt.figure(figsize=(4,4))
        count = 0
        for i in pf:
            plt.scatter(x=-1 * i.fitnesses[0], y=-1 * i.fitnesses[1], color='blue', alpha=0.7, s=130, edgecolors='black', linewidth = 0.5)
            if count == 17:
                plt.scatter(x=-1 * i.fitnesses[0], y=-1 * i.fitnesses[1], color='magenta', alpha=1, s=140, edgecolors='black', linewidth = 0.5)
            count = count + 1
        plt.ylabel('$f_2(\mathbf{x})$'+' = NAND(f=20 Hz)', fontsize=32)
        plt.xlabel('$f_1(\mathbf{x})$'+' = NAND(f=10 Hz)', fontsize=32)
        plt.title("Pareto Front", fontsize=24)
        plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.3)
        plt.grid(which='major', color='gray', linestyle='-', linewidth=0.5)
        plt.minorticks_on()
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        plt.ylim((0, 1.0))
        plt.xlim((0, 1.0))
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(4,4))
        j = 0
        for i in pf:
            plt.scatter(x=-1 * i.fitnesses[0], y=-1 * i.fitnesses[1], color='blue', alpha=0.7, s=150, edgecolors='black', linewidth = 0.5)
            plt.annotate(str(j), (-1 * i.fitnesses[0], -1 * i.fitnesses[1]))
            print(str(j) + ": " + str(list(i.genome1)), str(i.genome2), str(i.genome3))
            j = j + 1
        plt.ylabel('$f_2(\mathbf{x})$'+' = NAND(f=20 Hz)', fontsize=24)
        plt.xlabel('$f_1(\mathbf{x})$'+' = NAND(f=10 Hz)', fontsize=24)
        plt.title("Pareto Front", fontsize=24)
        plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.show()

        temp = []
f.close()

fitness1 = numpy.zeros([gens])
fitness2 = numpy.zeros([gens])
temp = []
individuals = []
with open('avgFitness.dat', "rb") as f:
    for r in range(1, runs+1):
        for g in range(1, gens+1):
            try:
                temp = -1 * pickle.load(f)
                fitness1[g-1] = temp[0]
                fitness2[g-1] = temp[1]
            except EOFError:
                break
        temp = []
f.close()

plt.figure(figsize=(6.4,4.8))
plt.plot(list(range(1, gens+1)), fitness1, color='blue', label='$f_1(\mathbf{x})$'+' = NAND(f=10 Hz)', linewidth=3)
plt.plot(list(range(1, gens+1)), fitness2, color='red', label='$f_2(\mathbf{x})$'+' = NAND(f=20 Hz)', linewidth=3)
plt.xlabel("Generations", fontsize=32)
plt.ylabel("Average Fitness", fontsize=32)
plt.title("Evolutionary Search", fontsize=32)
plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.3)
plt.grid(which='major', color='gray', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.legend(loc='upper right', fontsize=32)
plt.minorticks_on()
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.show()