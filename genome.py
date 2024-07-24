from switch_float import switch
import constants as c
import random
import math
import numpy as np
import sys
import os
import pickle
import uuid
import itertools

# genome is an array of real numbers between [1, 10]
# need to change the dimension of the grid on line 17

class GENOME:

    def __init__(self, num_fitnesses, id):
        self.ID = id #self.set_uuid()
        self.age = 0
        # total number of the particles in 5 by 6 grid
        self.n_col = 6
        self.n_row = 5
        self.N = self.n_col*self.n_row
        #self.idx_source = int((self.n_col+1)/2)
        #self.idx_output1 = 27#int(self.N-int((self.n_col+1)/2))
        #self.idx_output2 = 33
        #self.genome1 = np.random.choice([0.05,2,4,6,8,10], self.N, p=[0.25,0.15,0.15,0.15,0.15,0.15])
        self.genome1 = np.round(np.random.uniform(0.05, 10, size=self.N), decimals=1)
        locations = np.random.choice(list(range(self.N)), 4, replace=False)
        self.genome2 = locations[0]
        self.genome3 = locations[1]
        self.genome4 = locations[2]#int(self.N-int((self.n_col+1)/2))#locations[2]
        self.genome5 = locations[3]#int((self.n_col + 1) / 2)#locations[3]

        self.needs_eval = True
        self.fitnesses = [0.0 for x in range(num_fitnesses)]
        # determines if this individual was already evaluated
        self.needs_eval = True
        # nandness metric
        self.nandness = 0.0

    def aging(self):
        self.age = self.age + 1

    def dominates(self, other):
        # returns True if self dominates other param other, False otherwise.

        self_min_traits = self.get_minimize_vals()
        self_max_traits = self.get_maximize_vals()

        other_min_traits = other.get_minimize_vals()
        other_max_traits = other.get_maximize_vals()

        # all min traits must be at least as small as corresponding min traits
        if list(filter(lambda x: x[0] > x[1], zip(self_min_traits, other_min_traits))):
            return False

        # all max traits must be at least as large as corresponding max traits
        if list(filter(lambda x: x[0] < x[1], zip(self_max_traits, other_max_traits))):
            return False

        # any min trait smaller than other min trait
        if list(filter(lambda x: x[0] < x[1], zip(self_min_traits, other_min_traits))):
            return True

        # any max trait larger than other max trait
        if list(filter(lambda x: x[0] > x[1], zip(self_max_traits, other_max_traits))):
            return True

        # all fitness values are the same, default to return False.
        return self.ID < other.ID

    def distance(self, other):
        # checks the distance in genotype space, if they are close enough, returns true
        dist = np.abs(self.genome1 - other.genome1)
        threshold = 2
        return (dist<=threshold).all()

    def dominatesAll(self, other):
        # used for printing generation summary
        dominates = True
        for index in range(len(self.fitnesses)):
            dominates = dominates and (self.fitnesses[index] > other.fitnesses[index])
        return dominates

    def evaluate(self, analysis=False):

        if self.needs_eval:
            fabric = switch(self.genome1, 
                            self.genome2, 
                            self.genome3, 
                            self.genome4, 
                            self.genome5)
            results = fabric.evaluate()
            self.fitnesses = [-1.0*np.sum(results[:4])]
            self.nandness = self.fitnesses[0]#np.sum(self.fitnesses)
            self.needs_eval = False

        if analysis:
            fabric = switch(self.genome1,
                            self.genome2,
                            self.genome3,
                            self.genome4,
                            self.genome5)
            results = fabric.computeGateness(4)

        return self.fitnesses

    def get_minimize_vals(self):
        return [self.age]

    def get_maximize_vals(self):
        return self.fitnesses

    def mutate(self):
        self.needs_eval = True
        #mutationRate = 0.01
        type = random.random()
        if type < 0.75:
            # choose one particle randomly
            particle = np.random.randint(low=0, high=self.N)
            variation = np.random.normal(loc=0.0, scale=0.5)
            candidate = np.round(self.genome1[particle] + variation, decimals=1)
            if candidate<0.05:
                candidate = 0.05
            elif candidate>10:
                candidate = 10
            newGenome = np.array(self.genome1)
            newGenome[particle] = candidate
            while np.all(newGenome == self.genome1):
                particle = np.random.randint(low=0, high=self.N)
                variation = np.random.normal(loc=0.0, scale=0.5)
                candidate = np.round(self.genome1[particle] + variation, decimals=1)
                if candidate<0.05:
                    candidate = 0.05
                elif candidate>10:
                    candidate = 10
                newGenome[particle] = candidate

            self.genome1 = newGenome

        else:

            type2 = np.random.choice([2,3,4,5])

            not_options = [self.genome2, 
                           self.genome3,
                           self.genome4,
                           self.genome5]

            options = [l for l in range(self.N) if l not in not_options]
            new_location = np.random.choice(options)          

            if type2 == 2:
                self.genome2 = new_location
            elif type2 == 3:
                self.genome3 = new_location
            elif type2 == 4:
                self.genome4 = new_location
            elif type2 == 5:
                self.genome5 = new_location

        self.fitnesses = [0 for x in range(len(self.fitnesses))]
        #self.age = 0
        #self.set_uuid()
        
        #self.genome = candidate

    def genomePrint(self):
        print(' [fitness: ' , end = '' )
        print(self.fitnesses , end = '' )

        print(' age: ', end = '' )
        print(str(self.age)+']', end = '' )

        print()

        print(self.genome1, 
              self.genome2,
              self.genome3,
              self.genome4, 
              self.genome5)

        print()

    def genomeShow(self):
        print("fitness is: ")
        print(self.fitnesses)

    def set_uuid(self):
        self.ID = uuid.uuid1()
        return self.ID
