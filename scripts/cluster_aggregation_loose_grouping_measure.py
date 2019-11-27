import matplotlib.pyplot as plt
import matplotlib 
plt.rcParams.update({'figure.max_open_warning': 0})
import os 
from collections import OrderedDict
import numpy as np
import sys 
import itertools 


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


def missing_elements(L):
    start, end = 0, 100-1
    return sorted(set(range(start, end + 1)).difference(L)) 


def check_neighbors(pos): 

	global count  
	global count_total
	count = 0
	count_total = 0 
	tmp = [ ] 

	# orthogonal 
	if [adaptXPos(pos[0]+1), pos[1]] in all_agents_without:
		idx = all_agents_without.index([adaptXPos(pos[0]+1), pos[1]])
		count += 1
		tmp.append(idx)

	if [adaptXPos(pos[0]-1), pos[1]] in all_agents_without: 
		idx = all_agents_without.index([adaptXPos(pos[0]-1), pos[1]])
		count += 1 
		tmp.append(idx)

	if [pos[0], adaptYPos(pos[1]+1)] in all_agents_without: 
		idx = all_agents_without.index([pos[0], adaptYPos(pos[1]+1)])
		count += 1
		tmp.append(idx)

	if [pos[0], adaptYPos(pos[1]-1)] in all_agents_without:
		idx = all_agents_without.index([pos[0], adaptYPos(pos[1]-1)])
		count += 1
		tmp.append(idx)

	# minimum of 3 out of 4 orthogonal neighbors required 
	#if count >= 3: 
	count_total += count 
	# diagonal 
	if [adaptXPos(pos[0]+1), adaptYPos(pos[1]+1)] in all_agents_without: 
		idx = all_agents_without.index([adaptXPos(pos[0]+1), adaptYPos(pos[1]+1)])
		count_total += 1
		tmp.append(idx)

	if [adaptXPos(pos[0]+1), adaptYPos(pos[1]-1)] in all_agents_without:
		idx = all_agents_without.index([adaptXPos(pos[0]+1), adaptYPos(pos[1]-1)])
		count_total += 1 
		tmp.append(idx)

	if [adaptXPos(pos[0]-1), adaptYPos(pos[1]+1)] in all_agents_without:
		idx = all_agents_without.index([adaptXPos(pos[0]-1), adaptYPos(pos[1]+1)])
		count_total += 1
		tmp.append(idx)
		
	if [adaptXPos(pos[0]-1), adaptYPos(pos[1]-1)] in all_agents_without: 
		idx = all_agents_without.index([adaptXPos(pos[0]-1), adaptYPos(pos[1]-1)])
		count_total += 1
		tmp.append(idx)

	if count >= 3: 
		count = count_total

	return tmp 


def check_connected(pos, cID): 
	check = [ ] 

	# connection vs ortohogonal neighbors 
	if [adaptXPos(pos[0]+1), pos[1]] in all_agents_without: # agent on that positon exists 
		for i in range(0, len(aggregate)):
			if i != cID: 
				if [adaptXPos(pos[0]+1), pos[1]] in aggregate[i]:
					check.append(i) 

	if [adaptXPos(pos[0]-1), pos[1]] in all_agents_without: 
		for i in range(0, len(aggregate)):
			if i != cID: 
				if [adaptXPos(pos[0]-1), pos[1]] in aggregate[i]:
					check.append(i) 

	if [pos[0], adaptYPos(pos[1]+1)] in all_agents_without: 
		for i in range(0, len(aggregate)):
			if i != cID: 
				if [pos[0], adaptYPos(pos[1]+1)] in aggregate[i]:
					check.append(i) 

	if [pos[0], adaptYPos(pos[1]-1)] in all_agents_without:
		for i in range(0, len(aggregate)):
			if i != cID: 
				if [pos[0], adaptYPos(pos[1]-1)] in aggregate[i]:
					check.append(i) 

	return check 


# main part 

if len(sys.argv) >= 2:
	file = sys.argv[1]
else:
	file = 'agent_trajectory'

f = open(file, 'r')
pos = []
all_agents = []
all_agents_without = [] 

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
		pos.append([int(p[0]), int(p[1])])  
		
		all_agents.append([[int(p[0]), int(p[1])], [int(p[2]), int(p[3])]])
		all_agents_without.append([int(p[0]), int(p[1])])

# check for line 
aggregate = [ ] 
noAggregate = [ ]
loosely_grouped = [ ]
indices = [ ]  
tmp = [ ] 

while pos: 
	# check first element of list 
	checked = [ ] 
	cluster = [ ] 

	tmp = check_neighbors(pos[0])
	idx = all_agents_without.index(pos[0])

	# minimum 6 neighbors required + agents blocking movement of considered agent 
	if count >= 6: 
		
		# append center element 
		cluster.append(all_agents_without.index(pos[0]))
		checked.append(all_agents_without.index(pos[0]))
		del pos[0]

		# append neighbors of agent to cluster 
		for el in tmp: 
			cluster.append(el)
			
			if all_agents_without[el] in pos: # don't check for a complete new cluster again 
				i = pos.index(all_agents_without[el])
				del pos[i]

		while tmp: # check neighbors of neighbors 
			current = check_neighbors(all_agents_without[tmp[0]])
			checked.append(tmp[0])
			del tmp[0]

			if count >= 6:
				tmp.extend(current)
				# remove already checked from list 
				tmp = list(set(tmp))
				checked = list(set(checked))
				tmp = list(set(tmp) - set(checked))

				for el in current: 
					cluster.append(el)
				
					if all_agents_without[el] in pos:
						i = pos.index(all_agents_without[el])
						del pos[i]

	else: 
		del pos[0]

	if cluster: 
		cluster = list(set(cluster)) # ensure unique elements 
		indices.append(cluster)

# add elements instead of indices to arrays 
for c in indices:
	cl = [ ] 
	for idx in c: 
		cl.append(all_agents_without[idx])

	aggregate.append(cl)


connected = [ ] 

# check if clusters are connected 
max_connections = len(aggregate)*(len(aggregate)-1)/2.0 # undirected connections between all elements 

# check if connected clusters 
if len(aggregate) > 1:
	# check if connected via other cluster 
	for i in range(0, len(aggregate)):

		for el in aggregate[i]: 
			ret = list(set(check_connected(el, i)))
			
			if ret: 
				for eli in ret: 
					t = [i, eli]
					t.sort()
					connected.append(t)

connected.sort()
connected = list(connected for connected,_ in itertools.groupby(connected))

# generate all possible connections between elements of set 
if connected:
	check = 1 

	while check: 
		check = 0 
		length = len(connected)
		for i in range(0, length):
			for j in range(0, length):
				tmp = set(connected[i]) - set(connected[j])
				tmp2 = set(connected[j]) - set(connected[i])
				new = tmp.union(tmp2)
				new = list(new)
				new.sort() 
				if len(new) == 2 and not new in connected: 
					connected.append(new) 
					check = 1 
	
	# "set" of lists 
	connected.sort()
	connected = list(connected for connected,_ in itertools.groupby(connected))

# check which elements aren't in groups yet 
tmp = list(itertools.chain.from_iterable(indices))
tmp.sort()
no_id = missing_elements(tmp)

# agents not in aggregate
for idx in no_id:
	noAggregate.append(all_agents[idx])

# flat list - list of unique elements; all agents within cluster 
flat_list = [item for sublist in aggregate for item in sublist]
flat_list.sort()
flat_list = list(flat_list for flat_list,_ in itertools.groupby(flat_list))


# calculate & print diff 
f = open('eval_cluster_aggregation_loosegrouping.txt', 'w')

f.write('agents not within clusters: %d\n' % len(noAggregate))
f.write('percentage of agents not within structure: %f\n \n' % (float(len(noAggregate))/float(agents)))

f.write('agents within cluster: %d\n' % len(flat_list))
f.write('percentage of agents within structure: %f\n \n' % (float(len(flat_list))/float(agents)))

f.write('quantity of clusters: %d\n' % len(aggregate))
f.write('connections: %d\n' % len(connected))
f.write('max connections: %d\n\n' % max_connections)

# EVAL 
if(max_connections > 0):
	if(len(connected) == max_connections):
	    f.write('structure: Loosely Grouped')
	else: 
		f.write('structure: Clusters')
else: 
	if(len(aggregate) == 1): 
		f.write('structure: Aggregation')
	else: 
	    f.write('structure: Clusters')




