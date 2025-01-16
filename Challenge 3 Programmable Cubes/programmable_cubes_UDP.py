# Programmable Cubes challenge
# GECCO 2024 Space Optimisation Competition (SpOC)

from numba import njit
import numpy as np
from numba.typed import List
from CubeMoveset import MoveSetRev
import json
import matplotlib.pyplot as plt

############################################################################################################################################
##### THE UDP DEFINING AND IMPLEMENTING THE OPTIMIZATION CHALLENGE
############################################################################################################################################

class programmable_cubes_UDP:
    def __init__(self, problem):
        """
        A Pygmo compatible UDP User Defined Problem representing the Programmable Cubes challenge for SpOC 2024.
        https://esa.github.io/pygmo2/tutorials/coding_udp_simple.html explains more details on UDPs.

        Args:
            problem: Name of the problem / scenario to be used. Implemented: ISS, JWST, Enterprise.
        """
        # Variable name used for storing cube ensemble configuration after chromosome evaluation
        self.final_cube_positions = None

        # Load specifications of the problem
        with open('problems/{}.json'.format(problem), 'r') as infile:
            self.setup = json.load(infile)

        # Load target cube locations and cube types
        self.target_cube_positions = np.load('{}/Target_Config.npy'.format(self.setup['path']))
        self.target_cube_types = np.load('{}/Target_Cube_Types.npy'.format(self.setup['path']))
        # Load cube types of initial configuration
        self.initial_cube_types = np.load('{}/Initial_Cube_Types.npy'.format(self.setup['path']))
   
    def get_bounds(self):
        """
        tldr; Get bounds for the decision variables.

        The chromosome is composed of tuples representing cube ID and pivot command,
        
        i.e.,

        chromosome = [cube ID, command ID, cube ID, command ID, ..., -1]

        The chromosome always has to end with -1. It is read in from left to right, executing for each cube ID the 
        command ID that comes right afterwards.
        An early termination while reading the chromosome is possible by setting -1 instead of a valid cube ID.

        Returns:
            Tuple of lists: bounds for the decision variables.
        """
        # lb = lower bounds of the chromosome entries
        # rb = upper bounds of the chromosome entries
        lb, rb = [], []
        
        # All cube IDs are bound below by -1 (the first entry of -1 indicates the chromosome end, i.e., 
        # everything that comes after is ignored when calculating the fitness). 
        # Command IDs are bound below by 0.
        # The chromosome has to end with at least one entry of -1 (added here to ensure always a valid chromosome).
        lb += [-1, 0]*self.setup['max_cmds']
        lb += [-1]
        
        # Maximum cube ID is the number of cubes - 1.
        # Moves go from 0 to 5 (clock - and counterclockwise pivoting around x,y,z axis).
        rb += [self.setup['num_cubes']-1, 5]*self.setup['max_cmds']
        rb += [-1]
        
        return (lb, rb)

    def get_nix(self):
        """
        Get number of integer variables in the chromosome/decision vector.

        Returns:
            int: number of integer variables.
        """
        # the chromosome exists solely of integer variables.
        return self.setup['max_cmds']*2+1

    def fitness(self, chromosome, initial_configuration = None, verbose = False):
        """
        Fitness function for the UDP
        
        Args:
            chromosome: the chromosome/decision vector to be tested
            verbose: whether to provide more additional output during chromosome evaluation
        Returns:
            score: the score/fitness for this chromosome.
        """
        # By default, we start from the initial configuration provided in the problem.
        # This is also the point fitness evaluation on the optimize platform starts from.
        # During optimization, you can change the initial configuration to start,
        # e.g., from intermediate configurations.
        # Note: we do not check here whether a custom cube configuration is valid 
        # (i.e., whether all cubes are connected with each other).
        if initial_configuration is None:
            initial_configuration = np.load('{}/Initial_Config.npy'.format(self.setup['path']))

        # Create the cube ensemble with an initial cube configuration.
        cubes = ProgrammableCubes(initial_configuration)

        # Roll-out the command sequence and calculate the final fitness.
        steps_needed = cubes.apply_chromosome(chromosome, verbose)
        self.final_cube_positions = cubes.cube_position
        steps_fraction = steps_needed / self.setup['max_cmds']
        score = fitness_function(cubes, steps_fraction, self.setup['fitness_offset'], 
                                 self.setup['num_cube_types'], self.initial_cube_types, 
                                 self.target_cube_types, self.target_cube_positions)
                
        return [-score]
    
    def pretty(self, chromosome):
        self.fitness(chromosome, verbose = True)
    
    def example(self):
        pass
    
    def plot(self, which_one, cube_type_to_plot = [0], custom_config = None, custom_cube_types = None):
        '''
        Plot the cube ensemble with the default colour scheme given in /problems/*.yaml.
        Renders the plot without returning anything.

        Args:
            which_one: What to plot. Options: ['ensemble', 'target'].
                       'target' plots the target cube positions, i.e., the target shape. 
                       'ensemble' plots the configuration after applying a chromosome (starting from the initial configuration).
            cube_type_to_plot: list of cube types to plot (int from 0 to maximum number of cube types - 1).
            custom_config: If not None, this will overwrite the cube positions used for plotting.
            custom_cube_types: If not None, this will overwrite the cube types used for plotting.
        '''
        if which_one == 'ensemble':
            assert(self.final_cube_positions is not None)
            positions = self.final_cube_positions
            cube_types = self.initial_cube_types    
        if which_one == 'target':
            positions = self.target_cube_positions
            cube_types = self.target_cube_types
        if custom_config is not None:
            positions = custom_config
        if custom_cube_types is not None:
            cube_types = custom_cube_types

        cube_tensor = np.zeros((len(cube_type_to_plot), 
                                self.setup['plot_dim'], 
                                self.setup['plot_dim'], 
                                self.setup['plot_dim']))
        offset = int(np.fabs(np.min(positions)))

        for l in range(len(cube_type_to_plot)):
            for pos in positions[cube_types == cube_type_to_plot[l]]:
                i,j,k = pos
                cube_tensor[l][i+offset][j+offset][k+offset] = 1
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_facecolor('white')
        for i in range(len(cube_type_to_plot)):
            ax.voxels(cube_tensor[i], facecolor = self.setup['colours'][cube_type_to_plot[i]], edgecolors = 'k', alpha=.4)
        
        plt.tight_layout()
        plt.show()   

############################################################################################################################################
##### FITNESS FUNCTION
############################################################################################################################################

ALPHA = 0.1
BETA = 1 - ALPHA
def fitness_function(cube_ensemble, steps_fraction, offset, num_cube_types, init_cube_types, target_cube_types, target_cubes):
    '''
    Fitness function used to evaluate a chromosome.

    Measures how well the final cube configuration fits the target configuration.
    Takes also into account how fast the final cube configuration has been reached (i.e., the number of pivoting operations)!
    Note: cubes of the same type are interchangable without affecting the fitness.

    Args:
        cube_ensemble: the ProgrammableCubes Object containing the cube positions.
        steps_fraction: number of pivots required divided by number of maximum pivots allowed.
        num_cube_types: number of different cube types.
        init_cube_types: list containing cube type of each initial cube. Sorted the same way as initial cube positions.
        cube_types: list containing cube type of each target cube. Sorted the same way as target cube positions.
        target_cubes: list of target cube positions. 
    Returns:
        score: fitness of the chromosome (float)
    '''
    num_correct_cubes = 0
    num_total_cubes = len(cube_ensemble.cube_position)
    for types in range(num_cube_types):
        target_list = target_cubes[target_cube_types==types].tolist()
        final_list = cube_ensemble.cube_position[init_cube_types==types].tolist()
        overlap = [cube in final_list for cube in target_list]
        num_correct_cubes += np.sum(overlap)
    cube_fraction = num_correct_cubes / num_total_cubes
    score = BETA * cube_fraction + ALPHA * (1-steps_fraction)
    score = (score-offset)/(1-offset)
    
    return score

############################################################################################################################################
##### FUNCTIONS TO DETERMINE THE CUBES SURROUNDING A CUBE (i.e., within a radius of 3.5)
############################################################################################################################################

@njit(cache=True)
def get_distance_between_vectors(target_vector, vector_list):
    '''
    Function calculating Euclidean distance between a vector and a list of vectors.

    Args:
        target_vector: single vector
        vector_list: array of vectors
    Returns:
        list of floats containing the distance between the single vector and each vector in the list.
    '''
    return np.sqrt(np.sum((target_vector - vector_list)**2, axis=1))

@njit(cache=True)
def get_surrounding_cubes(target_cube_position, all_cube_positions):
    '''
    Function determining all surrounding cubes (within a radius of 3.5) and neighbouring cubes of a target cube.

    Neighbouring means connected by face.

    Args:
        target_cube_position: position vector of the target cube whose surrounding is to be determined.
        all_cube_positions: array containing all cube positions.
    Returns:
        neighbours: list of IDs of all neighbouring cubes of the target cube.
        surrounding: list of IDs of all cubes in the surrounding vicinity of the target cube.
    '''
    cube_distances = get_distance_between_vectors(target_cube_position, all_cube_positions)
    neighbours = np.where(cube_distances == 1)[0]
    surrounding = np.where((cube_distances>0)*(cube_distances<3.5))[0]
    return neighbours, surrounding


############################################################################################################################################
##### FUNCTION TO CHECK CONNECTEDNESS OF CUBES
############################################################################################################################################

@njit(cache=True)
def check_connectivity_of_cubes(cube_neighbours, removed_cube_ID):
    '''
    Function checking whether an ensemble of cubes is still connected if a single cube is removed.
    This is done by checking whether the neighbours of the removed cube are still part of the same connected cube ensemble.
    Moves through the ensemble neighbour by neighbour starting from one of the neighbours of the removed cube.

    Args:
        cube_neighbours: list of cube positions of the ensemble
        removed_cube_ID: ID of the removed cube.
    Results:
        True if ensemble stays connected, False if removing the cube splits the ensemble into at least two disconnected parts.
    '''
    # Get neighbours of the cube to be removed
    direct_neighbours = cube_neighbours[removed_cube_ID]
    num_neighbours = len(direct_neighbours)
    # If only one neighbour: stays connected (the cube will stay connected to its neighbour when pivoting)
    if num_neighbours == 1:
        return True
    
    # Start queueing through the ensemble neighbour by neighbour, starting from one of the neighbours of the removed cube
    queue = [direct_neighbours[0]]
    visited_nodes = []
    # Pointer for the queue
    pointer = 0
    # Encountered neighbours of the cube to be removed while traversing through the ensemble (or connectivity graph of the ensemble)
    # We start from one of its neighbours, so we initialize it to 1
    neighbours_found = 1
    while len(queue) > pointer:
        # Do not visit the cube that is to be removed during traversal
        if queue[pointer] == removed_cube_ID:
            pointer += 1
            continue

        # Cube visited
        visited_nodes.append(queue[pointer])
        
        # Add neighbours of visited cube to the queue
        new_cubes = cube_neighbours[queue[pointer]]
        # Exclude previously visited cubes
        not_visited_yet = np.array([x not in queue for x in new_cubes])
        new_cubes = new_cubes[not_visited_yet]
        queue += list(new_cubes)

        # Check if any of the cubes entering the queue are neighbours of the 
        # cube we want to remove
        for ids in new_cubes:
            if ids in direct_neighbours:
                neighbours_found  += 1
        
        # If we have encountered all neighbours of the cube to be removed,
        # then the cube ensemble is still fully connected
        if num_neighbours == neighbours_found:
            return True
    
        pointer += 1
        
    # If we end up here, at least on of the neighbours was not visited, meaning it 
    # is part of another sub-ensemble -> the cube ensemble got disconnected -> return False
    return False

############################################################################################################################################
##### FUNCTIONS FOR CHECKING WHETHER A MOVE IS VALID (i.e., the cube does not bump into other cubes while pivoting)
############################################################################################################################################
    
@njit(cache=True)
def scan_surrounding(cube_position, diff_to_env, surrounding_cubes_positions):
    '''
    Checks in a 5x5 grid around the cube that we want to move whether another cube is in these locations.
    The 5x5 grid is in the plane normal to the rotation axis (see CubeMoveSet.py for more details).

    Args:
        cube_position: position vector of the cube to be moved.
        diff_to_env: list of vectors used to calculate the coordinates of the 5x5 grid locations (see CubeMoveSet.py).
        surrounding_cubes_positions: array of positions of the surrounding cubes of the cube to be moved.
    Returns:
        List of boolean entries. Entries are True if a cube is present in this location of the 5x5 surrounding grid, False otherwise.
    '''
    output = np.full((24), False)
    for i in range(len(surrounding_cubes_positions)):
        delta = (np.sum(np.fabs(cube_position+diff_to_env - surrounding_cubes_positions[i]), axis=1) == 0)
        output = np.logical_or(output, delta)
    return output

@njit(cache=True)
def check_if_move_is_valid(move_to_use, occupance, move_to_options, move_patterns_occupied, move_patterns_empty):
    '''
    Check if move is valid.

    This is done by going through all possible moves for a counter or counter-clockwise pivot maneuver and checking if 

    1) the cube would pivot and connect with another cube
    2) the cube does not collide with other cubes while pivoting.

    This is done by checking the surrounding of the moving cube. Each pivot maneuver is represented by a certain
    move pattern, i.e., where in its surrounding cubes should be present, and where no cubes should be present (to satisfy 1 and 2).

    Args:
        move_to_use: Move operation (clockwise / counterclockwise)
        occupance: binary list indicating where cubes are present in a 5x5 grid around the cube (see CubeMoveSet.py for more details). Calculated by "scan_surrounding".
        move_to_options: all maneuvers leading to the selected move (all ways a clockwise / counterclockwise rotation can be realized).
        move_patterns_occuped: for each maneuver, which cubes have to be occupied in the surrounding.
        move_pattern_empty: for each maneuver, which cubes have to be empty in the surrounding.
    Returns:
        If a maneuver fits, True and the ID of the underlying move (used to identify where the cube moves to).
        If none fit, only returns False and None.
    '''
    for i in move_to_options[move_to_use]:
        if np.all(occupance[move_patterns_occupied[i]]) == False:
            continue
        else:
            if np.all(~occupance[move_patterns_empty[i]]) == True:
                return True, i
    return False, None

@njit(cache=True)
def update_ensemble(cube_to_move, surrounding, new_surrounding_cubes):
    '''
    Update surroundings of the cubes in the old and new surrounding of the cube that moved (and the cube that moved itself).
    Done by checking which cubes left and entered the surrounding of the moving cube, and updating only those (rest stays unchanged).

    Function can also be used to update neighbours.

    Args:
        cube_to_move: the cube that is being moved.
        surrounding: list containing the IDs of surrounding cubes for every cube.
        new_surrounding_cubes: new surrounding cubes of the cube that moved (after moving).
    Returns:
        Nothing, surrounding is updated in place.
    '''
    to_remove = surrounding[cube_to_move][np.array([x not in new_surrounding_cubes for x in surrounding[cube_to_move]])]
    to_add = new_surrounding_cubes[np.array([x not in surrounding[cube_to_move] for x in new_surrounding_cubes])]

    for cubes in to_remove:
        surrounding[cubes] = surrounding[cubes][np.nonzero(surrounding[cubes] != cube_to_move)]
    for cubes in to_add:
        surrounding[cubes] = np.append(surrounding[cubes], [cube_to_move])
    surrounding[cube_to_move] = new_surrounding_cubes
    
############################################################################################################################################
##### PUTTING EVERYTHING TOGETHER: A CLASS IMPLEMENTING THE CUBE ENSEMBLE UPDATE LOGIC
############################################################################################################################################ 
   
class ProgrammableCubes:
    def __init__(self, initial_positions):
        '''
        The programmable cube ensemble.

        An ensemble is represented by an array containing all cube positions, as well as a list containing
        the IDs of surrounding cubes of each cube (surrounding cubes are cubes that have a distance less than 3.5).

        Cube IDs are used as indices both for positions and the surroundings list.

        The moveset defines how cubes can reconfigure, i.e., it contains all possible maneuvers for pivoting cubes.

        Contains methods to:
            - reset the ensemble to a new configuration (recalculating the list of surrounding cubes).
            - check and apply an update step, i.e., pivot a cube.
            - unroll a chromosome.
        '''
        # Properties of the cube ensemble
        # 1) Positions of the cubes
        # + derivative quantities of the cube positions, pre-calculated and only updated locally
        # to speed up calculations
        # 2) The surrounding of each cube, i.e., the IDs of the cubes within a radius of 3.5 of the cube
        # 3) The neighbours of each cube, i.e., the IDs of the cubes with a distance of exactly 1 (connected to the cube by face)
        self.cube_position = None
        self.cube_surroundings = None
        self.cube_neighbours = None
        # formal description of the physically possible pivoting maneuvers a cube can do
        self.moveset = MoveSetRev()
        # Set up the cube ensemble with initial cube positions
        # reset populates cube_surroundings and cube_neighbours automatically
        self.reset(initial_positions)
        
    def reset(self, positions):
        '''
        Reset the cube ensemble to a new configuration.

        Args:
            positions: cube positions to reset the ensemble to.
        Returns:
            Changes cube_position and cube_surroundings in place.
        '''
        self.cube_position = np.array(positions)
        self.cube_surroundings = List()
        self.cube_neighbours = List()
        # Populate surroundings and neighbours for each cube
        for i in range(len(self.cube_position)):
            neighs, surrs = get_surrounding_cubes(self.cube_position[i], self.cube_position)
            self.cube_surroundings.append(surrs)
            self.cube_neighbours.append(neighs)
        
    def apply_single_update_step(self, cube_to_move, move_to_use, step = '', verbose = False):
        '''
        Function applying a single update to the cube ensemble.

        For a move to be legal, two conditions have to be met:

        1) removing the selected cube from the ensemble does not disconnect it, i.e., split it into at least two disconnected components.
        2) The cube can perform the pivoting maneuver without detaching or colliding with other cubes.

        Args:
            cube_to_move: ID of the cube that is to be moved.
            move_to_use: move to be used (0 - 5, indicating clock/counterclockwise rotation around x,z,y axis)
            verbose: if True, additional information is printed.
        Returns:
            1 if the move was legal and applied, 0 if the move was illegal and ignored.
        '''
        # Extract rotation axis and rotation direction
        rot_axis = int(move_to_use/2)
        rot_dir =  move_to_use % 2

        ## CHECK CONDITION 1
        # check if ensemble stays fully connected if the cube is removed
        # (best case, this only iterates over a few cubes -> fast; worst case, this iterates over the whole ensemble -> slower)
        not_connecting_cube = check_connectivity_of_cubes(self.cube_neighbours, cube_to_move)

        if not_connecting_cube == True:
            # Removing the cube does not split the ensemble into two! Lets goooo.
            ## CHECK CONDITION 2
            surrounding_cube_positions = self.cube_position[self.cube_surroundings[cube_to_move]]
            occupance = scan_surrounding(self.cube_position[cube_to_move], 
                                         self.moveset.diffs_to_surrounding[rot_axis], 
                                         surrounding_cube_positions)
            move_legal, which = check_if_move_is_valid(rot_dir, occupance, self.moveset.move_pattern_mapping,
                                                       self.moveset.move_patterns_occupied, 
                                                       self.moveset.move_patterns_empty)
            if move_legal == True:
                # The move is legal! I am so exciting, and I just can't hide it!
                ## UPDATE THE ENSEMBLE
                # Update position
                self.cube_position[cube_to_move] += self.moveset.displacements[rot_axis*self.moveset.number_moves+which] 
                # find new surroundings of cube
                new_neighbouring_cubes, new_surrounding_cubes = get_surrounding_cubes(self.cube_position[cube_to_move], self.cube_position)
                
                # Update surroundings and neighbours
                update_ensemble(cube_to_move, self.cube_surroundings, new_surrounding_cubes)
                update_ensemble(cube_to_move, self.cube_neighbours, new_neighbouring_cubes)
                    
                # Everything worked!
                return 1
            else:
                # Oh no...
                if verbose == True:
                    print('{} -- Cannot apply move {} to cube {}. Cube would crash into neighbouring cubes. Skipping command...'.format(step, move_to_use, cube_to_move))
                return 0
        else:
            # Oh no...
            if verbose == True:
                print('{} -- Cannot apply move {} to cube {}. Cube is connecting. Skipping command...'.format(step, move_to_use, cube_to_move))
            return 0
                
    def apply_chromosome(self, chromosome, verbose):
        '''
        Given a chromosome, execute the command sequence encoded in it until
        a maximum number of steps is reached, or the chromosome ends / terminates early.

        Chromosome evaluating ends when the first cube ID of "-1" is encountered in the chromosome.

        Args:
            chromosome: The chromosome to apply.
            verbose: Whether to print additional information while applying the chromosome.
        Returns:
            chrom_end: the number of commands issued to reach the final configuration.
        '''
        # Find end point of the chromosome
        chrom_end = int(np.where(chromosome == -1)[0][0]/2)

        # Lets roll!
        for i in range(chrom_end):
            cube_to_move = int(chromosome[2*i])
            move_command = int(chromosome[2*i+1])
            # Every move command counts, so clean up illegal moves (those that do not change the ensemble)
            self.apply_single_update_step(cube_to_move, move_command, step = i, verbose = verbose)
        return chrom_end
    
############################################################################################################################################