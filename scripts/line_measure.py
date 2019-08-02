import matplotlib.pyplot as plt
import matplotlib 
plt.rcParams.update({'figure.max_open_warning': 0})
import os 
from collections import OrderedDict
import numpy as np
import sys 
import math 


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


def missing_elements_x(L):
    start, end = 0, grid[0]-1
    return sorted(set(range(start, end + 1)).difference(L))


def missing_elements_y(L):
    start, end = 0, grid[1]-1
    return sorted(set(range(start, end + 1)).difference(L))


def check_consecutive(L):
	for i in range(len(L)):
		if i == len(L)-1: 
			break 
		if L[i+1] == L[i]+1: 
			return True

	return False  


def measure_line(file): 
	global grid 
	global agents 
	global lines 
	global noLine 
	global vertical 
	global horizontal
	global horizontal_agents
	global vertical_agents
	global grid_spanning

	pos = []
	vertical = 0 
	horizontal = 0 
	vertical_agents = 0 
	horizontal_agents = 0 
	grid_spanning = 0 
	line = [ ] 
	lines = [ ]
	noLine = [ ] 
	lineflag = 0 
	
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
			# all agent positions + headings 
			pos.append([[int(p[0]), int(p[1])], [int(p[2]), int(p[3])]])

	# sort list of positions for easier processing 
	pos.sort()

	# list of all agents w/o headings 
	all_cells = [x[0] for x in pos]

	while len(pos) > 1:	
		line = [ ] 
		lineflag = 0 

		# get first element
		line.append(pos[0])
		del pos[0] 

		if line[-1][1][0] != 0: # heading in x-direction 

			# elements from x_dir will be deleted 
			xdir = [x[0] for x in pos]

			# check if agents are to the right of last added agent 
			while line and [line[-1][0][0]+1, line[-1][0][1]] in xdir:
				index = xdir.index([line[-1][0][0]+1, line[-1][0][1]])
				# check if heading is in x-direction + append to line structure  
				if pos[index][1][0] != 0: 
					line.append(pos[index])
					del pos[index]
					del xdir[index]
				else: 
					break

			# check if agents are to the left of the first agent
			if line and [adaptXPos(line[0][0][0]-1), line[0][0][1]] in xdir:
				index = xdir.index([adaptXPos(line[0][0][0]-1), line[0][0][1]])
				if pos[index][1][0] != 0:
					line.append(pos[index])
					del pos[index]
					del xdir[index]

				# check if agents are to the left of the last added agent 
				while line and [adaptXPos(line[-1][0][0]-1), line[-1][0][1]] in xdir:
					index = xdir.index([adaptXPos(line[-1][0][0]-1), line[-1][0][1]])
					if pos[index][1][0] != 0: 
						line.append(pos[index])
						del pos[index]
						del xdir[index]
					else: # no more agents to the left 
						break

			# line covers whole grid length
			if len(line) == grid[0]:

				# maximum number of neighbors = line length / 2 				
				max_neighbors = math.ceil(len(line)/2.0) 

				# top side 
				neighbor_count_top = 0  
				neighbors = [ ]

				for el in line: 
					if [el[0][0], adaptYPos(el[0][1]+1)] in all_cells: 
						neighbor_count_top += 1 
						neighbors.append(el[0][0])

				neighbors.sort()
				if check_consecutive(neighbors):
					lineflag = 1

				# check if number of agents is maximum line length / 2 
				if neighbor_count_top > max_neighbors:
					lineflag = 1 
				
				# bottom side 
				neighbor_count_bottom = 0 
				neighbors = []

				for el in line: 
					if [el[0][0], adaptYPos(el[0][1]-1)] in all_cells: 
						neighbor_count_bottom += 1 
						neighbors.append(el[0][0])
						
				neighbors.sort()

				if check_consecutive(neighbors):
					lineflag = 1

				if neighbor_count_bottom > max_neighbors:
					lineflag = 1 

				# line flag is set when max neighbors criteria is met 
				if not lineflag: 
					lines.append(line)
					horizontal = horizontal + 1
					horizontal_agents = horizontal_agents + len(line)
					if len(line) == grid[0]:
						grid_spanning = grid_spanning + 1 
				else: 
					for n in line: 
						noLine.append(n)

			# line does not span whole grid length, but is longer than two agents  
			elif len(line) > 2:
	 			# find start point of line 
				missing = [x[0][0] for x in line] # all elements in line 

				while missing: 
					# line spans over torus (step between 0 and last cell of grid)
					if 0 in missing and grid[0]-1 in missing: 

						missing.sort() # sort positions

						# get missing elements in list (sorted)
						missingX = missing_elements_x(missing) 

						if missingX: 
							minValue = adaptXPos(missingX[0]-1) # first element of line 

							missing = [x[0][0] for x in line] # unsorted list (as line is build)
							minID = missing.index(minValue) # id of element 
					else: # line does not span over torus end 
						minID = missing.index(min(missing))

					# element in heading direction in list?  (check that agent points towards line structure)
					if not adaptXPos(line[minID][0][0] + line[minID][1][0]) in missing:
						noLine.append(line[minID])
						del line[minID]
						del missing[minID]

						if not missing: # no elements in line left 
							break
					else: # first element of line found 
						break 

				if line: 
					minValue = line[minID]

				missing = [x[0][0] for x in line] # all elements in line 

				while missing: 
					if 0 in missing and grid[0]-1 in missing: 

						missing.sort()
						missingX = missing_elements_x(missing)

						if missingX: 
							maxValue = adaptXPos(missingX[-1]+1)
							missing = [x[0][0] for x in line]
							maxID = missing.index(maxValue)
					else: 
						maxID = missing.index(max(missing))

					# element in max heading in list? 
					if not adaptXPos(line[maxID][0][0] + line[maxID][1][0]) in missing:
						noLine.append(line[maxID])
						del line[maxID]
						del missing[maxID]

						if not missing:
							break
					else: 
						break 

				if line: 
					minID = line.index(minValue)

				# check if one complete line 
				if line and (line[minID][1][0] == -1 * line[maxID][1][0] or len(line)== grid[0]):
					# check how many cells next to agent are covered
					max_neighbors = math.ceil(len(line)/2.0) 

					# top side 
					neighbor_count_top = 0  
					neighbors = []
					for el in line: 
						if [el[0][0], adaptYPos(el[0][1]+1)] in all_cells: 
							neighbor_count_top += 1 
							neighbors.append(el[0][0])

					neighbors.sort()

					if check_consecutive(neighbors):
						lineflag = 1 

					if neighbor_count_top > max_neighbors:
						lineflag = 1 

					# bottom side 
					neighbor_count_bottom = 0 
					neighbors = [ ] 

					for el in line: 
						if [el[0][0], adaptYPos(el[0][1]-1)] in all_cells: 
							neighbor_count_bottom += 1 
							neighbors.append(el[0][0])

					neighbors.sort()

					if check_consecutive(neighbors):
						lineflag = 1 
							
					if neighbor_count_bottom > max_neighbors:
						lineflag = 1 

					if len(line) < 3:
						lineflag = 1 

					# line flag is set when max neighbors criteria is met 
					if not lineflag: 
						lines.append(line)
						horizontal = horizontal + 1
						horizontal_agents = horizontal_agents + len(line)
						if len(line) == grid[0]:
							grid_spanning = grid_spanning + 1 
					else: 
						for n in line: 
							noLine.append(n)
				else: 
					for n in line: 
						noLine.append(n)

			else: # if no line 
				for n in line: 
					noLine.append(n)
			
			line = [ ]

		else: # y-direction != 0 
			
			# next agent - go upward (same heading direction, y position + 1)
			while pos and pos[0][0] == [line[-1][0][0], adaptYPos(line[-1][0][1]+1)] and pos[0][1][1] != 0: 
				line.append(pos[0])
				del pos[0]

			index = -1 

			# next agent - go downwards (same heading, y position - 1)
			if [[line[0][0][0], adaptYPos(line[0][0][1]-1)], [line[0][1][0], -1*line[0][1][1]]] in pos:
				index = pos.index([[line[0][0][0], adaptYPos(line[0][0][1]-1)], [line[0][1][0], -1*line[0][1][1]]])
			elif [[line[0][0][0], adaptYPos(line[0][0][1]-1)], line[0][1]] in pos:
				index = pos.index([[line[0][0][0], adaptYPos(line[0][0][1]-1)], line[0][1]])

			if index != -1: 
				line.append(pos[index])
				del pos[index]
				index = index - 1

				while index >= 0 and pos[index][0] == [line[-1][0][0], adaptYPos(line[-1][0][1]-1)] and pos[index][1][1] != 0: 
					line.append(pos[index])
					del pos[index]
					index = index - 1 

			# post processing line elements 
			
			if len(line) == grid[1]: # line length = grid length 

				# check how many cells next to agent are covered
				max_neighbors = math.ceil(len(line)/2.0)

				neighbor_count_top = 0  
				neighbors = [] 

				# top side 
				for el in line: 
					if [adaptXPos(el[0][0]+1), el[0][1]] in all_cells: 
						neighbor_count_top += 1 
						neighbors.append(el[0][1])

				neighbors.sort()

				if neighbor_count_top > max_neighbors or check_consecutive(neighbors):
					lineflag = 1 


				# bottom side 
				neighbor_count_bottom = 0 
				neighbors = []

				for el in line: 
					if [adaptXPos(el[0][0]-1), el[0][1]] in all_cells: 
						neighbor_count_bottom += 1
						neighbors.append(el[0][1])

				neighbors.sort() 

				if neighbor_count_bottom > max_neighbors or check_consecutive(neighbors):
					lineflag = 1 

				if not lineflag: 
					lines.append(line)
					vertical = vertical + 1
					vertical_agents = vertical_agents + len(line)
					if len(line) == grid[1]:
						grid_spanning = grid_spanning + 1 
				
				else: 
					for n in line: 
						noLine.append(n)

				line = [ ] 

			# line length at least 3 
			elif len(line) > 2:
				# determine if line 
				missing = [x[0][1] for x in line]

				# cut upper part 
				while missing:
					if 0 in missing and grid[1]-1 in missing: 
						missing.sort()
						missingY = missing_elements_y(missing)

						if missingY: 
							minValue = adaptYPos(missingY[0]-1)
							missing = [x[0][1] for x in line]
							minID = missing.index(minValue)

					else: 
						minID = missing.index(min(missing))

					if not adaptYPos(line[minID][0][1] + line[minID][1][1]) in missing:
						noLine.append(line[minID])
						del line[minID]
						del missing[minID]

					else:
						break

				missing = [x[0][1] for x in line]

				if missing: 
					minValue = line[minID]

				# cut lower part 
				while missing:
					if 0 in missing and grid[1]-1 in missing: 
						missing.sort()
						missingY = missing_elements_y(missing)

						if missingY: 
							maxValue = adaptYPos(missingY[-1]+1)
							missing = [x[0][1] for x in line]
							maxID = missing.index(maxValue) 
					else: 
						maxID = missing.index(max(missing)) 

					if not adaptYPos(line[maxID][0][1] + line[maxID][1][1]) in missing:
						noLine.append(line[maxID])
						del line[maxID]
						del missing[maxID]

					else:
						break  

				if line: 
					minID = line.index(minValue)

				# check if one complete line  
				if line and (line[minID][1][1] == -1 * line[maxID][1][1]):
					# check how many cells next to agent are covered
					max_neighbors = math.ceil(len(line)/2.0) 

					# right side 
					neighbor_count_top = 0  
					neighbors = [] 

					for el in line: 
						if [adaptXPos(el[0][0]+1), el[0][1]] in all_cells: 
							neighbor_count_top += 1 
							neighbors.append(el[0][1])

					neighbors.sort()

					if neighbor_count_top > max_neighbors or check_consecutive(neighbors):
						lineflag = 1 

					# left side 
					neighbor_count_bottom = 0 
					neighbors = []

					for el in line: 
						if [adaptXPos(el[0][0]-1), el[0][1]] in all_cells: 
							neighbor_count_bottom += 1
							neighbors.append(el[0][1])

					neighbors.sort()

					if neighbor_count_bottom > max_neighbors or check_consecutive(neighbors):
						lineflag = 1 

					if len(line) < 3:
						lineflag = 1 

					if not lineflag: 
						lines.append(line)
						vertical = vertical + 1
						vertical_agents = vertical_agents + len(line)
						if len(line) == grid[1]:
							grid_spanning = grid_spanning + 1 
					else: 
						for n in line: 
							noLine.append(n)
				else: 
					for n in line: 
						noLine.append(n)
			else: # if no line 
				for n in line: 
					noLine.append(n)
			
			line = [ ]

	if len(pos) > 0:
		for el in pos:
			noLine.append(el)

		pos = [ ] 

	return noLine 



if __name__ == "__main__":

	if len(sys.argv) >= 2:
		file = sys.argv[1]
	else:
		file = 'agent_trajectory'

	measure_line(file)

	# calculate & print diff 

	f = open('eval_structure_line.txt', 'w')	
	f.write('agents not within line structure: %d\n' % len(noLine))
	f.write('percentage of agents not within structure: %f\n' % (float(len(noLine))/float(agents)))
	f.write('agents within line structure: %d\n' % (agents-len(noLine)))
	f.write('percentage of agents within structure: %f\n \n' % (float(agents-len(noLine))/float(agents)))

	f.write('horizontal lines: %d\n' % horizontal)
	if horizontal > 0: 
	    f.write('agents in horizontal lines: %f\n' % (float(horizontal_agents)/float(agents-len(noLine))))
	f.write('vertical lines: %d\n' % vertical)
	if vertical > 0: 
	    f.write('agents in vertical lines: %f\n' % (float(vertical_agents)/float(agents-len(noLine))))
	f.write('grid spanning lines: %d\n' % grid_spanning)


