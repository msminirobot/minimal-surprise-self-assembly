import matplotlib.pyplot as plt
import matplotlib 
plt.rcParams.update({'figure.max_open_warning': 0})
import os 
from collections import OrderedDict
import numpy as np
import sys 
import line_measure as line 
import squares_measure as square
import pair_measure as pair 


def adaptYPos(y):
	if y >= grid[1]:
		return y - grid[1]
	elif y < 0:
		return y + grid[1]
	else:
		return y 


def adaptXPos(x):
	if x >= grid[0]:
		return x - grid[0]
	elif x < 0:
		return x + grid[0]
	else:
		return x 


# main part 
if len(sys.argv) >= 2:
	file = sys.argv[1]
else:
	file = 'agent_trajectory'

f = open(file, 'r')
pos = []
all_agents = []

# check which agents are not within a line, square or pair  
noLine = line.measure_line(file)
noSquare = square.measure_square(file)
noPair = pair.measure_pairs(file)

# read in file with agent positions 
for line in f: 	
	# store generation number 
	if line[:4] == "Gen:":
		gen = int(line[5:].rstrip())

	elif line[:5] == "Grid:":
		row = line[6:].split(', ')
		grid = [int(row[0]), int(row[1].rstrip())]

	elif line[:7] == "Agents:":
		agents = int(line[8:].rstrip())

	elif line[:3] != "500": # do nothing as long as time step not last time step 
		continue 

	elif line.split():   
		row = line.split(': ') # first split
		p = row[1].split(', ')
		pos.append([int(p[0]), int(p[1])]) #heading unimportant for loose dispersion  
		all_agents.append([[int(p[0]), int(p[1])], [int(p[2]), int(p[3])]])

disp = [ ] 
noDisp = [ ] 

for i in range(0, len(pos)):
	count = 0
	
	# 8 agents in Moore neighborhood 
	if [adaptXPos(pos[i][0]+1), pos[i][1]] in pos:
		count += 1

	if [adaptXPos(pos[i][0]-1), pos[i][1]] in pos: 
		count += 1 

	if [pos[i][0], adaptYPos(pos[i][1]+1)] in pos: 
		count += 1

	if [pos[i][0], adaptYPos(pos[i][1]-1)] in pos:
		count += 1

	if [adaptXPos(pos[i][0]+1), adaptYPos(pos[i][1]+1)] in pos: 
		count += 1
	
	if [adaptXPos(pos[i][0]+1), adaptYPos(pos[i][1]-1)] in pos:
		count += 1 
	
	if [adaptXPos(pos[i][0]-1), adaptYPos(pos[i][1]+1)] in pos:
		count += 1
	
	if [adaptXPos(pos[i][0]-1), adaptYPos(pos[i][1]-1)] in pos: 
		count += 1
    
    # check if max 1 neighbor in Moore Neighborhood 
	if count <= 1: 

		# check if part of pair 
		if not (all_agents[i] in noPair): 
			count = 8

		# check if line 
		if not (all_agents[i] in noLine):
			count = 8  

		# check if part of square 
		if not ([pos[i], 0] in noSquare):
			count = 8 


	if count <= 1: 
		disp.append(pos[i])
	else: 
		noDisp.append(pos[i])

# calculate & print diff 

f = open('eval_loose_dispersion.txt', 'w')
f.write('agents not loosely dispersed: %d\n' % len(noDisp))
f.write('percentage of agents not within structure: %f\n' % (float(len(noDisp))/float(agents)))
f.write('agents loosely dispersed: %d\n' % len(disp))
f.write('percentage of agents within structure: %f\n \n' % (float(len(disp))/float(agents)))
