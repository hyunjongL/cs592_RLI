#!/usr/bin/env python
import math
import numpy as np

class Node:
    """
    Node class for dijkstra search
    """
    def __init__(self, pos, idx, cost, prev_idx):
        self.pos  = np.array(pos)
        self.idx  = idx
        self.cost = cost
        self.prev_idx = prev_idx # previous node's index

    def __str__(self):
        return str(self.pos) + "," + str(self.cost) + "," + str(self.prev_idx)


def get_grid_index(state, resolution, grid_limits, grid_dim):
    """ 
    A function to convert a state to the index of grid.

    Parameters
    ----------
    state : list
        a list of coordinate values
    resolution : float
        a value of grid resolution
    grid_limits : list
        a list of minimum and maximum configuration limits, where
        each limit is a list of float values.
        (e.g., [[min_x,_min_y,min_z],[max_x,max_y,max_z]])
    grid_dim : list
        a list of grid dimensions.

    Returns
    -------
     : Integer
        The index of the input state in grid.
    """
    ids = np.round((state-grid_limits[0])/resolution).astype(int)
    if len(state)==2:
        return ids[0]*grid_dim[1]+ids[1]
    elif len(state)==3:
        return ids[0]*grid_dim[1]*grid_dim[2]+ids[1]*grid_dim[2]+ids[2]
    return NotImplemented


def is_valid(state, grid_limits, obstacle_tree, robot_size):
    """ 
    A function to check the validity of the input state.

    Parameters
    ----------
    state : list
        a list of coordinate values
    grid_limits : list
        a list of minimum and maximum configuration limits, where
        each limit is a list of float values.
        (e.g., [[min_x,_min_y,min_z],[max_x,max_y,max_z]])
    obstacle_tree : BallTree 
        a BallTree object from Scikit-Learn. The tree includes
        the coordinates of obstacles.
    robot_size : float
        a radius of a robot. This value is used for collision checking.       

    Returns
    -------
     : boolean
        True if the input state is valid
    """
    ## from IPython import embed; embed(); sys.exit()
    
    # check workspace limits
    if any( state[i] < grid_limits[0][i] for i in range(len(grid_limits[0])) ):
        return False
    if any( state[i] > grid_limits[1][i] for i in range(len(grid_limits[1])) ):
        return False

    # check collision
    if len(np.shape(state))==1: state=state[np.newaxis,:]
    count = obstacle_tree.query_radius(state, robot_size, count_only=True)
    if count>0: return False
    return True
    
def out_of_map(candidate_pos, grid_limits):
    min_limit = np.any((candidate_pos < grid_limits[0]) == True)
    max_limit = np.any((candidate_pos > grid_limits[1]) == True)
    return min_limit or max_limit

def collides(candidate_pos, obstacle_tree):
    return obstacle_tree.query(candidate_pos.reshape(1, -1))[0] == 0

def cost_function(action):
    return 1
    return np.sqrt(np.sum(action**2))

def dijkstra_planning(start, goal, actions, resolution, grid_limits,
                          obstacle_tree, robot_size, **kwargs):
    """ 
    A function to generate a path from a start to a goal 
    using Dijkstra search-based planning algorithm.  

    Parameters
    ----------
    start : list
        a list of start coordinate values (e.g., [x,y,z]).  
    goal : list
        a list of goal coordinate values (e.g., [x,y,z]).  
    actions : list
        a list of available actions, where each action is a list
        of float values (e.g., [[vx,vy,vz],[vx,vy,vz], ..., ]).  
    resolution : float
        a value of grid resolution
    grid_limits : list
        a list of minimum and maximum configuration limits, where
        each limit is a list of float values.
        (e.g., [[min_x,_min_y,min_z],[max_x,max_y,max_z]])
    obstacle_tree : BallTree 
        a BallTree object from Scikit-Learn. The tree includes
        the coordinates of obstacles.
    robot_size : float
        a radius of a robot. This value is used for collision checking.       

    Returns
    -------
    path : list
        a list of coordinate values that is the shortest path from 
        the start to the goal.
    """
    
    grid_dim = np.round((grid_limits[1]-grid_limits[0])/resolution+1).astype(int)

    openset, closedset = dict(), dict()

    # Set a start node
    start_node = Node(start,
                      get_grid_index(start, resolution, grid_limits, grid_dim),
                      0, -1)
    openset[start_node.idx] = start_node

    # Set a goal node
    goal_node = Node(goal,
                     get_grid_index(goal, resolution, grid_limits, grid_dim),
                     0, -1)
    if start_node.idx==goal_node.idx: return []
    print("Start and goal indices: {} and {}".format(start_node.idx,
                                                         goal_node.idx))

    while True:
        # Empty openset
        if len(openset) == 0: return None

        cur_idx  = min(openset, key=lambda o: openset[o].cost)
        cur_node = openset[cur_idx]


        #------------------------------------------------------------
        # ADD YOUR CODE
        #------------------------------------------------------------
        # Break if reach to the goal
        if cur_idx == goal_node.idx :
            closedset[cur_idx] = cur_node
            break

        
        # Remove the item from the open set
        del openset[cur_idx]

        # Add it to the closed set
        closedset[cur_idx] = cur_node

        # expand nodes based on available actions
        for i, action in enumerate(actions):
            candidate_pos = cur_node.pos + action

            if out_of_map(candidate_pos, grid_limits) or collides(candidate_pos, obstacle_tree):
                # Actions that go out of the map or collide should not be considered
                continue

            candidate_cost = cur_node.cost + cost_function(action)
            candidateNode = Node(candidate_pos,
                      get_grid_index(candidate_pos, resolution, grid_limits, grid_dim),
                      candidate_cost, cur_idx)
            
            if candidateNode.idx in closedset: # Closed Node
                continue
            elif candidateNode.idx in openset:
                # An open node should be compare the cost 
                if openset[candidateNode.idx].cost > candidate_cost:
                    openset[candidateNode.idx] = candidateNode
                else:
                    continue
            else:
                # Not yet opened node should be opened
                openset[candidateNode.idx] = candidateNode
            
            # Otherwise if it is already in the open set
            # ...
            
        #------------------------------------------------------------
    
    goal_node = closedset[goal_node.idx]
    # Track the path from goal to start
    path = [goal_node.pos]
    #------------------------------------------------------------
    # ADD YOUR CODE
    #------------------------------------------------------------
    prev_idx = goal_node.prev_idx
    while prev_idx != start_node.idx:
        cur_node = closedset[prev_idx]
        path.append(cur_node.pos)
        prev_idx = cur_node.prev_idx
    
    path.append(start_node.pos)
    #------------------------------------------------------------

    return path[::-1]

                

