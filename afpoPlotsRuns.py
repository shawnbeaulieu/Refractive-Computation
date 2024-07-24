import pickle
import matplotlib.pyplot as plt
from switch_float import switch
import constants as c
import numpy

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

runs = c.RUNS
gens = c.numGenerations

fitness1 = numpy.zeros([runs, gens])
fitness2 = numpy.zeros([runs, gens])
temp = []
individuals = []
for r in range(1, runs+1):
    with open(f'avgFitness{r}.dat', "rb") as f:
        for g in range(1, gens+1):
            try:
                temp = -1 * pickle.load(f)
                fitness1[r-1, g-1] = temp[0]
                fitness2[r-1, g-1] = temp[1]
            except EOFError:
                break
        temp = []
f.close()

mean_f_1 = numpy.mean(fitness1, axis=0)
std_f_1 = numpy.std(fitness1, axis=0)

mean_f_2 = numpy.mean(fitness2, axis=0)
std_f_2 = numpy.std(fitness2, axis=0)

plt.figure(figsize=(6.4,4.8))
plt.plot(list(range(1, gens+1)), mean_f_1, color='green', label='$f_1(\mathbf{x})$'+' = NAND($\omega$=10)', linewidth=3)
plt.fill_between(list(range(1, gens+1)), mean_f_1-std_f_1, mean_f_1+std_f_1, color='green', alpha=0.1, linewidth=1)
plt.plot(list(range(1, gens+1)), mean_f_2, color='red', label='$f_2(\mathbf{x})$'+' = NAND($\omega$=20)', linewidth=3)
plt.fill_between(list(range(1, gens+1)), mean_f_2-std_f_2, mean_f_2+std_f_2, color='red', alpha=0.1, linewidth=1)
plt.xlabel("Generations", fontsize=42)
plt.ylabel("Average Fitness", fontsize=42)
plt.title("Evolutionary Search", fontsize=42)
plt.grid(which='minor', color='gray', linestyle=':', linewidth=0.3)
plt.grid(which='major', color='gray', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.legend(loc='upper right', fontsize=32)
plt.minorticks_on()
plt.xticks(fontsize=42)
plt.yticks(fontsize=42)
plt.show()