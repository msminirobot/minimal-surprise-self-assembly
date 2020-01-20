import matplotlib.pyplot as plt
import matplotlib 
plt.rcParams.update({'figure.max_open_warning': 0})
import os 
from collections import OrderedDict
import numpy as np
import sys 


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


def measure_triangles(file):
	global noLattice
	global  lattice
	global agents 
	global grid 

	pos = []

	# read in file with agent positions 
	f = open(file, 'r')

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
			pos.append([[int(p[0]), int(p[1])], 0]) #heading unimportant for triangular lattice 

	# check for line 
	lattice = [ ] 
	noLattice = [ ] 

	elements = [x[0] for x in pos] 

	# iterate through all elements in pos
	for el in elements:
		count = 0
		occupied = 0 
		indices = [ ] 
		
		# 4 outer agents  
		if [adaptXPos(el[0]+2), el[1]] in elements:
			count += 1
			indices.append(elements.index([adaptXPos(el[0]+2), el[1]]))
		if [adaptXPos(el[0]-2), el[1]] in elements: 
			count += 1 
			indices.append(elements.index([adaptXPos(el[0]-2), el[1]]))
		if [el[0], adaptYPos(el[1]+2)] in elements: 
			count += 1
			indices.append(elements.index([el[0], adaptYPos(el[1]+2)]))
		if [el[0], adaptYPos(el[1]-2)] in elements:
			count += 1
			indices.append(elements.index([el[0], adaptYPos(el[1]-2)]))
		# 4 inner agents 
		if [adaptXPos(el[0]+1), adaptYPos(el[1]+1)] in elements: 
			count += 1
			indices.append(elements.index([adaptXPos(el[0]+1), adaptYPos(el[1]+1)]))
		if [adaptXPos(el[0]+1), adaptYPos(el[1]-1)] in elements:
			count += 1 
			indices.append(elements.index([adaptXPos(el[0]+1), adaptYPos(el[1]-1)]))
		if [adaptXPos(el[0]-1), adaptYPos(el[1]+1)] in elements:
			count += 1
			indices.append(elements.index([adaptXPos(el[0]-1), adaptYPos(el[1]+1)]))
		if [adaptXPos(el[0]-1), adaptYPos(el[1]-1)] in elements: 
			count += 1
			indices.append(elements.index([adaptXPos(el[0]-1), adaptYPos(el[1]-1)]))

		# rest of agents 
		if [adaptXPos(el[0]-1), adaptYPos(el[1])] in elements:
			occupied += 1
		if [adaptXPos(el[0]+1), adaptYPos(el[1])] in elements:
			occupied += 1 
		if [adaptXPos(el[0]), adaptYPos(el[1]-1)] in elements:
			occupied += 1 
		if [adaptXPos(el[0]), adaptYPos(el[1]+1)] in elements: 
			occupied += 1
		if [adaptXPos(el[0]-1), adaptYPos(el[1]+2)] in elements: 
			occupied += 1 
		if [adaptXPos(el[0]+1), adaptYPos(el[1]+2)] in elements:  
			occupied += 1
		if [adaptXPos(el[0]-1), adaptYPos(el[1]-2)] in elements:
			occupied += 1 
		if [adaptXPos(el[0]+1), adaptYPos(el[1]-2)] in elements: 
			occupied += 1 
		if [adaptXPos(el[0]-2), adaptYPos(el[1]-1)] in elements: 
			occupied += 1
		if [adaptXPos(el[0]-2), adaptYPos(el[1]+1)] in elements: 
			occupied += 1 
		if [adaptXPos(el[0]+2), adaptYPos(el[1]-1)] in elements:
			occupied += 1
		if [adaptXPos(el[0]+2), adaptYPos(el[1]+1)] in elements:
			 occupied += 1 

		if occupied == 0 and count == 8:
			pos[elements.index(el)][1] = 1 
			for i in indices:
				pos[i][1] = 1 

	for el in pos:
		if el[1] == 1:
			lattice.append(el)
		else:
			noLattice.append(el[0])

	return noLattice

if __name__ == "__main__":
	# main part 

	if len(sys.argv) >= 2:
		file = sys.argv[1]
	else:
		file = 'agent_trajectory'

	measure_triangles(file); 

	# calculate & print diff 

	f = open('eval_triangular_lattice.txt', 'w')
	f.write('agents not within triangular lattice: %d\n' % len(noLattice))
	f.write('percentage of agents not within structure: %f\n\n' % (float(len(noLattice))/float(agents)))
	f.write('agents within trianguar lattice: %d\n' % len(lattice))
	f.write('percentage of agents within structure: %f\n \n' % (float(len(lattice))/float(agents)))
	f.close() 
