import matplotlib.pyplot as plt
import matplotlib 
plt.rcParams.update({'figure.max_open_warning': 0})
import os 
from collections import OrderedDict
import numpy as np
import sys 
import math 
import line_measure as lines 


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


def measure_pairs(file):
	global grid
	global agents 
	global pairs
	global none 

	noLine = lines.measure_line(file)

	pos = []
	p_all = []
	pos_all = []

	f = open(file, 'r')

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
			# array to be modified with all agents 
			pos.append([[int(p[0]), int(p[1])], [int(p[2]), int(p[3])]])
			# array with all agents w/o headings 
			p_all.append([int(p[0]), int(p[1])])
			# array with all agents (not to be modified)
			pos_all.append([[int(p[0]), int(p[1])], [int(p[2]), int(p[3])]])

	pairs = []
	none = [] 

	while pos:
		if [[adaptXPos(pos[0][0][0]+pos[0][1][0]), adaptYPos(pos[0][0][1]+pos[0][1][1])], [-1*pos[0][1][0], -1*pos[0][1][1]]] in pos: 
			index = pos.index([[adaptXPos(pos[0][0][0]+pos[0][1][0]), adaptYPos(pos[0][0][1]+pos[0][1][1])], [-1*pos[0][1][0], -1*pos[0][1][1]]])
			
			check = 0 

			# conditions to check if more complex structure 
			if not (pos[0] in noLine and pos[index] in noLine): # both agents not part of line structure
				check = 1 
			# agents on one side of both agents
			elif [adaptXPos(pos[index][0][0]+pos[index][1][1]), adaptYPos(pos[index][0][1]+pos[index][1][0])] in p_all and [adaptXPos(pos[0][0][0]-1*pos[0][1][1]), adaptYPos(pos[0][0][1]-1*pos[0][1][0])] in p_all: 
				check = 1 
			# other side 
			elif [adaptXPos(pos[index][0][0]-pos[index][1][1]), adaptYPos(pos[index][0][1]-pos[index][1][0])] in p_all and [adaptXPos(pos[0][0][0]+pos[0][1][1]), adaptYPos(pos[0][0][1]+pos[0][1][0])] in p_all: 
				check = 1 

			if check: # agents either part of line or two neighbors at one side 
				none.append(pos[0])
				none.append(pos[index])
			else: 
				pairs.append(pos[0])
				pairs.append(pos[index])
			
			del pos[index]

		else: 
			none.append(pos[0])

		del pos[0]

	return none 



if __name__ == "__main__":

	if len(sys.argv) >= 2:
		file = sys.argv[1]
	else:
		file = 'agent_trajectory'

	measure_pairs(file)

	# calculate & print diff 
	f = open('eval_pair.txt', 'w')
	f.write('agents not within pair structure: %d\n' % len(none))
	f.write('percentage of agents not within structure: %f\n' % (float(len(none))/float(agents)))
	f.write('agents within pair structure: %d\n' % (len(pairs)))
	f.write('percentage of agents within structure: %f\n \n' % (len(pairs)/float(agents)))


