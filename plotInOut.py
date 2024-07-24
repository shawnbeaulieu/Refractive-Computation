import constants as c
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
import sys
from os.path import exists

from switch_float import switch

runs = 5
gens = c.numGenerations
seed = int(sys.argv[1])

with open('bests{0}.dat'.format(seed), "rb") as f:
    individual = pickle.load(f)

stiffness = individual.genome1
print(stiffness)
print(np.max(stiffness))
print(np.min(stiffness))

input_1 = individual.genome2
input_2 = individual.genome3
input_3 = individual.genome4
input_4 = individual.genome5
print(input_1, input_2, input_3, input_4)

# running the best individuals
temp = []
rubish = []
#a = np.array([10, 10, 10, 10, 10, 1, 10, 10, 1, 10, 10, 10, 10, 10, 10, 10, 10, 1, 10, 10, 10, 10, 10, 1, 1, 10, 10, 1, 10, 1])
#a = np.array([1, 1, 1, 1, 1, 10, 1, 1, 10, 1, 1, 1, 1, 1, 1, 1, 1, 10, 1, 1, 1, 1, 1, 10, 10, 1, 1, 10, 1, 10])

if True:
    individual.evaluate(False)#analyze=True, seed=seed)
    exit()
else:
    alphas = np.load("nandness_Seed={0}.npy".format(seed), allow_pickle=True).item()[1]
print(max(alphas))
switch.showPacking(alphas, stiffness, input_1, input_2, input_3, input_4, input_5, input_6, input_7)
#switch.showPacking(np.round(np.random.uniform(1, 10, size=30), decimals=1), 14, 16)
#print(switch.plotInOut(a))

for r in [999]:#in range(1, runs+1):
    with open(f'bests{0}.dat'.format(seed), "rb") as f:
        # population of the last generation
        temp = pickle.load(f)
        # best individual of last generation
        best = temp[0]
        print(best.indv.genome1)
        print(best.indv.genome2)
        print(best.indv.genome3)
        switch.showPacking(best.indv.genome1, best.indv.genome2, best.indv.genome3)
        print(switch.plotInOut(best.indv.genome1, best.indv.genome2, best.indv.genome3))
        print("fitness: " + str(best.indv.fitness))
        temp = []
    f.close()
