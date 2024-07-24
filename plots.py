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

pfs = []
for gen in range(1, gens):
    filename = 'gens/gen{}.dat'.format(str(gen))
    with open(filename, "rb") as f:
        temp = pickle.load(f)
        pf = paretoFront(temp)
        print(len(pf), flush=True)
        pfs.append(len(pf))
    f.close()

f = open(f'pfs.dat', 'ab')
pickle.dump(pfs, f)
f.close()

plt.figure(figsize=(4,4))
for gen in range(1, gens):
    plt.scatter(x=gen, y=pfs[gen-1], color='blue', alpha=0.5)
plt.xlabel('generations', fontsize=16)
plt.ylabel('size of pareto front', fontsize=16)
plt.title("Pareto Front", fontsize=16)
plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig("fig.png", dpi=200)
