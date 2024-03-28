import random
import numpy as np

list = [[4,5,7,4], [8,10,15.0,47], [5,4,7]]

parents = random.sample(list, 2)

fitnesses = []
print(f'parents : {parents}')

for individual in parents:
    print(f'Individual : {individual}')
    fitnesses.append(5)        

print(f'Parents : {parents}')
best_individual = max(parents)

print(best_individual)