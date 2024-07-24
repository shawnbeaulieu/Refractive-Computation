from afpo import AFPO
import os
import constants as c
import random

import subprocess as sub
import sys

# input the seed
seed = int(sys.argv[1])

#cleaning up  the data files
sub.call(f"rm -rf gens{seed}", shell=True)
sub.call(f"mkdir gens{seed}/", shell=True)

try:
    os.remove(f"lastGeneration{seed}.dat")
except OSError:
    pass
try:
    os.remove(f"avgFitness{seed}.dat")
except OSError:
    pass
try:
    os.remove(f"bests{seed}.dat")
except OSError:
    pass

runs = c.RUNS
for r in range(1, runs+1):
    print("*********************************************************", flush=True)
    print("seed: "+str(seed), flush=True)
    randomSeed = seed
    random.seed(seed)
    afpo = AFPO(seed, 1)
    afpo.evolve()
    
