#!/usr/bin/env python
import math
import random
import numpy as np
import matplotlib.pyplot as plt

class SARSA:
    
    def __init__(self, env, actions, resolution, grid_limits,
                    epsilon=.3, total_steps=200000, alpha=.4, gamma=.95):
        self.epsilon = epsilon
        self.total_steps = total_steps
        self.alpha = alpha
        self.gamma = gamma

        self.env = env
        self.actions = actions
        self.resolution = resolution
        self.grid_limits = grid_limits
        self.grid_dim = np.round((grid_limits[1]-grid_limits[0])/resolution+1).astype(int)

        self.action_dim = len(actions)
        self.observation_dim = self.get_grid_index(self.grid_dim-1)+1

        self.Q = np.zeros((self.observation_dim, self.action_dim), dtype=np.float64)


    def get_grid_index(self, state):
        """ Compute a grid index given an input state"""
        ids = np.round((state-self.grid_limits[0])/self.resolution).astype(int)
    
        if len(state)==2:
            return ids[0]*self.grid_dim[1]+ids[1]

        elif len(state)==3:
            return ids[0]*self.grid_dim[1]*self.grid_dim[2]+ids[1]*self.grid_dim[2]+ids[2]

        return NotImplemented

    
    def choose_action(self, state, epsilon=None):
        """
        A function to select an action given a state

        Parameters
        ----------
        state list/array: agent's status
        epsilone float: epsilon-greedy action selection
        """
        if epsilon is None: epsilon = self.epsilon

        #------------------------------------------------------------
        # Place your code here
        # -----------------------------------------------------------
        action = None
        if np.random.uniform(0, 1) < epsilon:
            # exploration
            rand_index = random.randint(0, self.action_dim - 1)
            action = rand_index
        else:
            # exploitation
            current_grid = self.get_grid_index(state)
            
            best = None
            best_action_idx = None

            for action_idx in range(self.action_dim):
                if best is None or best < self.Q[current_grid, action_idx]:
                    best_action_idx = action_idx
                    best = self.Q[current_grid, action_idx]
            action = best_action_idx
        # -----------------------------------------------------------
        return action

    
    def update(self, state, state2, reward, action, action2):
        """
        A function to update q-values

        Parameters
        ----------
        state   list/array: current state
        state2  list/array: next state
        reward  list/array: 
        action  int:        current action
        action2 int:        next action
        """

        #------------------------------------------------------------
        # Place your code here
        # -----------------------------------------------------------
        current_grid = self.get_grid_index(state)
        next_grid = self.get_grid_index(state2)
        self.Q[current_grid, action] = self.Q[current_grid, action] + self.alpha * (reward + self.gamma * (self.Q[next_grid, action2] - self.Q[current_grid, action]))
        # -----------------------------------------------------------

        
    def learn(self):
        """ learn """
        epi, t, et, m, score, m_score = 0, 0, 0, 0, 0., 0.

        state1 = self.env.reset() 
        action1 = self.choose_action(state1)

        while t < self.total_steps:
            state2, reward, done, info = self.env.step(self.actions[action1])
            action2 = self.choose_action(state2)

            self.update(state1, state2, reward, action1, action2)

            state1, action1 = state2, action2

            t+=1
            et+=1
            score += reward

            if et>=200:
                done = True
        
            if done:
                m += 1
                m_score += score

                if epi%100==0:
                    m_score /= m
                    print("episode:", epi+1, " steps:", et, " reward:", m_score)
                    
                    m_score, m = 0., 0

                et, score = 0, 0.
                epi+=1

                state1 = self.env.reset()
                action1 = self.choose_action(state1)



    def test(self):
        """ test"""

        state = self.env.reset() 

        while True:
            self.env.render()

            action = self.choose_action(state, epsilon=0.)
            next_state, reward, done, info = self.env.step(self.actions[action])

            state = next_state

            if done:
                self.env.close()
                break

        self.env.render_value_map(self.Q, self.grid_dim[0])
