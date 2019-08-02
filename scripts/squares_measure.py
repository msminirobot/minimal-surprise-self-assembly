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


def measure_square(file):
	global agents 
	global grid 
	global square
	global noSquare 

	f = open(file, 'r')
	pos = []

	# read in file with robot positions 
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
			pos.append([[int(p[0]), int(p[1])], 0]) #heading unimportant for squares 

	# check for square 
	square = [ ] 
	noSquare = [ ] 

	# all positions of agents (without marker)
	elements = [j[0] for j in pos] 

	# iterate through all elements in pos
	for el in elements:
		count = 0
		occupied = 0 
		indices = [ ] 
		
		# 3 neighboring robots to form square patterns     
		if [adaptXPos(el[0]+2), el[1]] in elements:
			count += 1
			indices.append(elements.index([adaptXPos(el[0]+2), el[1]]))
		if [adaptXPos(el[0]+2), adaptYPos(el[1]+2)] in elements: 
			count += 1 
			indices.append(elements.index([adaptXPos(el[0]+2), adaptYPos(el[1]+2)]))
		if [el[0], adaptYPos(el[1]+2)] in elements: 
			count += 1
			indices.append(elements.index([el[0], adaptYPos(el[1]+2)]))

		# rest of agents in 5x5 area - cells have to be empty 
		if [el[0], adaptYPos(el[1]-1)] in elements:
			occupied += 1
		if [el[0], adaptYPos(el[1]+1)] in elements:
			occupied += 1 
		if [el[0], adaptYPos(el[1]+3)] in elements:
			occupied += 1 
		if [adaptXPos(el[0]+2), adaptYPos(el[1]-1)] in elements: 
			occupied += 1
		if [adaptXPos(el[0]+2), adaptYPos(el[1]+1)] in elements: 
			occupied += 1 
		if [adaptXPos(el[0]+2), adaptYPos(el[1]+3)] in elements:  
			occupied += 1
		if [adaptXPos(el[0]-1), adaptYPos(el[1]-1)] in elements:
			occupied += 1 
		if [adaptXPos(el[0]-1), adaptYPos(el[1])] in elements: 
			occupied += 1 
		if [adaptXPos(el[0]-1), adaptYPos(el[1]+1)] in elements: 
			occupied += 1
		if [adaptXPos(el[0]-1), adaptYPos(el[1]+2)] in elements: 
			occupied += 1 
		if [adaptXPos(el[0]-1), adaptYPos(el[1]+3)] in elements:
			occupied += 1
		if [adaptXPos(el[0]+1), adaptYPos(el[1]-1)] in elements:
			occupied += 1 
		if [adaptXPos(el[0]+1), adaptYPos(el[1])] in elements: 
			occupied += 1 
		if [adaptXPos(el[0]+1), adaptYPos(el[1]+1)] in elements: 
			occupied += 1
		if [adaptXPos(el[0]+1), adaptYPos(el[1]+2)] in elements: 
			occupied += 1 
		if [adaptXPos(el[0]+1), adaptYPos(el[1]+3)] in elements:
			occupied += 1
		if [adaptXPos(el[0]+3), adaptYPos(el[1]-1)] in elements:
			occupied += 1 
		if [adaptXPos(el[0]+3), adaptYPos(el[1])] in elements: 
			occupied += 1 
		if [adaptXPos(el[0]+3), adaptYPos(el[1]+1)] in elements: 
			occupied += 1
		if [adaptXPos(el[0]+3), adaptYPos(el[1]+2)] in elements: 
			occupied += 1 
		if [adaptXPos(el[0]+3), adaptYPos(el[1]+3)] in elements:
			occupied += 1
        
        # mark agents part of square 
		if occupied == 0 and count == 3:
			pos[elements.index(el)][1] = 1 
			for i in indices:
				pos[i][1] = 1 

	for el in pos:
		if el[1] == 1:
			square.append(el)
		else:
			noSquare.append(el)
    
	return noSquare 


if __name__ == "__main__":
	if len(sys.argv) >= 2:
		file = sys.argv[1]
	else:
		file = 'agent_trajectory'

	measure_square(file)

	# calculate & print diff 
	f = open('eval_squares.txt', 'w')
	f.write('agents not within squares: %d\n' % len(noSquare))
	f.write('percentage of agents not within structure: %f\n' % (float(len(noSquare))/float(agents)))
	f.write('agents within squares: %d\n' % len(square))
	f.write('percentage of agents within structure: %f\n \n' % (float(len(square))/float(agents)))
