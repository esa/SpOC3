# Programmable cubes challenge - Cube pivoting maneuver definition
# GECCO 2024 Space Optimisation Competition (SpOC)

from numba.typed import List
import numpy as np

# Moveset where all moves can be reversed by an inverse rotation around the same axis.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Moves are defined in x-y, y-z and x-z plane. 
# For a move to be valid, certain locations have to be occupied by a cube (to allow docking), 
# and some have to be free  (to allow unoccupied pivoting).

# The surrounding is numbered with indices as follows.
# For x-y: upwards is increasing y, right is increasing x
# For y-z: upwards is increasing z, right is increasing y
# For x-z: upwards is increasing z, right is increasing x
# The cube to be moved is positioned in the middle (no index)
#
# 0  1  2  3  4
# 5  6  7  8  9
# 10 11 -- 12 13
# 14 15 16 17 18
# 19 20 21 22 23

# Example
# denote by x: has to be occupied
# denote by o: has to be unoccupied
# denote by C: cube to be moved
# denote by _: occupation irrelevant
#
# first scenario: counter-clockwise move to position 12. 
#                  [7,8] have to be occupied, [12,16,17] have to be empty.
#
# _ _ _ _ _
# _ _ x x _
# _ _ C o _
# _ _ o o _
# _ _ _ _ _
#
# reverse scenario: clockwise move to position 11. 
#                   [6,7] have to be occupied, [11,15,16] have to be empty.
#
# _ _ _ _ _
# _ x x _ _
# _ o C _ _
# _ o o _ _
# _ _ _ _ _

# All move patterns (8 per rotation direction and per rotation axis)
move_patterns_occupied = [[7,8], [6,11], [15,16], [12,17], # counterclockwise rotation
                          [6,7], [11,15], [16,17], [8,12], # clockwise rotation (reverse of above)
                          [16], [7], [11], [12], # counterclockwise rotation
                          [12], [11], [16], [7]] # clockwise rotation (again reverse moves)
move_patterns_empty = [[12,16,17], [7,8,12], [6,7,11], [11,15,16], # same as in occupied
                      [11,15,16], [12,16,17], [7,8,12], [6,7,11],
                      [14,15, 10,11,6,7], [8,9,12,13,16,17], [12,6,7,8,1,2], [11,15,16,17,21,22],
                      [11,6,7,8,2,3], [12,15,16,17,20,21], [17,18,12,13,7,8], [5,6,10,11,15,16]] 

# Split between 90° and 180° moves at index:
split = 8 
number_moves = 16

# maps move to either being a result of counterclockwise or clockwise rotation
move_pattern_mapping = List([np.array([0,1,2,3,8,9,10,11]), np.array([4,5,6,7,12,13,14,15])])

# Resulting displacement of node due to move
diffs = [[1,0], [0,1], [-1,0], [0,-1], 
         [-1,0], [0,-1], [1,0], [0,1],
         [-1,-1], [1,1], [-1,1], [1,-1],
         [1,1], [-1,-1], [1,-1], [-1,1]]
displacements = []
# for x-y
for j,i in diffs:
    displacements.append([j,i,0])
# for y-z
for j,i in diffs:
    displacements.append([0,j,i])
# for x-z
for j,i in diffs:
    displacements.append([j,0,i])

# Coordinate differences from cube to its surrounding locations
zsurrs, ysurrs, xsurrs = [],[],[]
for i in [2,1,0,-1,-2]:
    for j in [-2, -1, 0, 1, 2]:
        if i != 0 or j != 0:
            zsurrs.append([j,i,0])
for i in [2,1,0,-1,-2]:
    for j in [-2, -1, 0, 1, 2]:
        if i != 0 or j != 0:
            xsurrs.append([0,j,i])
for i in [2,1,0,-1,-2]:
    for j in [-2, -1, 0, 1, 2]:
        if i != 0 or j != 0:
            ysurrs.append([j,0,i])
diffs_to_surrounding = [np.array(zsurrs), np.array(xsurrs), np.array(ysurrs)]

# Structure that contains all important information about the moveset
class MoveSetBase:
    def __init__(self, move_patterns_occupied, move_patterns_empty, move_pattern_mapping, displacements, split, number_moves, diffs_to_surrounding):
        self.move_patterns_occupied = List()
        for moves in move_patterns_occupied:
            self.move_patterns_occupied.append(np.array(moves))
        self.move_patterns_empty = List()
        for moves in move_patterns_empty:
            self.move_patterns_empty.append(np.array(moves))
        self.move_pattern_mapping = move_pattern_mapping
        self.displacements = displacements
        self.split = split
        self.number_moves = number_moves
        self.diffs_to_surrounding = diffs_to_surrounding
        
class MoveSetRev(MoveSetBase):
    def __init__(self):
        super().__init__(move_patterns_occupied, move_patterns_empty, move_pattern_mapping, displacements, split, number_moves, diffs_to_surrounding)