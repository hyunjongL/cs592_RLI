# -*- coding: utf-8 -*-
from cmath import inf
from distutils.log import error
import numpy as np
import pickle
import copy
import itertools


class valueIterAgent:
    def __init__(self, env, gamma=0.9):
        """
        A function to select an action index given a state

        Parameters
        ----------
        env object: Gym Environment
        gamma float: discount factor
        """

        self.env         = env
        self.gamma       = gamma
        self.grid_limits = np.array([env.observation_space.low, env.observation_space.high]).astype(int) 
        self.resolution  = 1 #resolution
        self.grid_dim    = np.round((self.grid_limits[1]-self.grid_limits[0])/self.resolution+1).astype(int)
        self.actions    = np.array([[-1,0], [0,-1], [1,0], [0,1]])
        self.action_dim = len(self.actions)
        
        self.policy    = None
        self.values    = None

        x = range(self.grid_limits[0][0], self.grid_limits[1][0]+1)
        y = range(self.grid_limits[0][1], self.grid_limits[1][1]+1)
        self.states = np.array(list(itertools.product(x,y)))
        ## self.T      = create_transition_matrix(len(env.observation_space.low), self.action_dim)
        self.observation_dim = len(self.states)

    def get_grid_pos(self, idx):
        x = idx // self.grid_dim[1]
        y = idx % self.grid_dim[1]
        # print(x, y)
        return np.array((x, y))

    def get_grid_index(self, state):
        """ 
        Compute a grid index given an input state

        Parameters
        ----------
        state list/array: agent's status
        """
        ids = np.round((state-self.grid_limits[0])/self.resolution).astype(int)

        if len(state)==2:
            return ids[0]*self.grid_dim[1]+ids[1]

        return NotImplemented
        
    def out_of_map(self, pos):
        return not (pos.tolist() in self.states.tolist())

    def collides(self, pos):
        return pos.tolist() in self.env.objects.tolist()

    def stochastic_selection(self, prob_list):
        num_candidates = len(prob_list)
        prob_list = prob_list ** 3
        prob_list /= np.sum(prob_list)
        return np.random.choice(range(num_candidates), 1, p=prob_list)
        
    def choose_action(self, state):
        """
        Return an action index given a state

        Parameters
        ----------
        state list/array: agent's status
        """
        #------------------------------------------------------------
        # Place your code here
        # -----------------------------------------------------------
        if self.policy is None:
            error("No policy yet")
            return
        actions = self.policy[self.get_grid_index(state)]
        print(state, actions)
        # action_idx = np.argmax(actions)
        action_idx = self.stochastic_selection(actions)
        print(actions, action_idx)

        return action_idx[0] # action index
        # -----------------------------------------------------------
    
    def lookahead(self, state, values):
        best_value = -float("inf")
        for action in self.actions:
            post_action_state = state + action
            new_value = self.new_values(post_action_state, values)
            best_value = max(best_value, new_value)
        return best_value
    
    def transition_probability(self, state):
        num_candidates = 0
        for action in self.actions:
            new_pos = state + action
            if self.env.isValid(new_pos, check_collision=True):
                num_candidates += 1
        if num_candidates == 0:
            return 1./self.action_dim
        return 1./num_candidates

    def available_actions(self, state):
        actions = []
        for action in self.actions:
            if self.env.isValid(state + action, check_collision=True):
                actions.append(action)
        return actions
    def new_values(self, state, values):
        best_value = -float("inf")
        reward = 0
        
        # prob = self.transition_probability(state)
        prob = 1
        for action in self.actions:
            post_action_state = state + action
            reward = self.env.get_reward(state)
            # reward = self.env.get_reward(post_action_state)

            if self.out_of_map(post_action_state):
                value = reward
            else:
                value = reward + self.gamma * values[self.get_grid_index(post_action_state)]
                # The probabilty does not have to be here as the objective is to find the action with the max value.
                # The probabilty here is meaningful when there are multiple available states upon a single action.
        
            # elif self.collides(post_action_state):
            #     reward = -1
            # elif np.array_equal(post_action_state, self.env.goal_state):
            #     reward = 1
            # else:
            #     reward = 0
            # value = self.transition_probability(action) * (reward + self.gamma * values[self.get_grid_index(post_action_state)])            
            # print(value, best_value, "MAX", max(value, best_value))
            best_value = max(value, best_value)
            
        return best_value
 
            # if self.collides(post_action_state) or self.out_of_map(post_action_state):
            #     continue

            # if np.array_equal(post_action_state, self.env.goal_state):
            #     reward = 1
            # else:
            #     reward = 0
            # value = self.transition_probability(action) * (reward + self.gamma * values[self.get_grid_index(post_action_state)])            
            # # print(value, best_value, "MAX", max(value, best_value))
            # best_value = max(value, best_value)


    def solve_mdp(self, error=1e-10, **kwargs):
        """
        Compute values via value iterations

        Parameters
        ----------
        error float: a convergence check threshold
        """
        values = np.zeros(len(self.states))
        nxt_values = np.zeros(len(self.states))
        
        max_cnt = kwargs.get('max_cnt', None)

        # update each state
        diff  = float("inf")
        count = 0
        while True:
            count += 1
            delta = 0 # To use max function, first set delta to 0 // https://www.baeldung.com/cs/ml-value-iteration-vs-policy-iteration
            
            #------------------------------------------------------------
            # Place your code here
            # -----------------------------------------------------------
            # https://www.baeldung.com/cs/ml-value-iteration-vs-policy-iteration

            for i, state in enumerate(self.states):
                v = values[i]
                # Would it be ok to directly change the values array
                # nxt_values[i] = self.lookahead(state, values)
                nxt_values[i] = self.new_values(state, values)
                # print(v, nxt_values[i], abs(v - nxt_values[i]))
                delta = max(abs(v - nxt_values[i]), delta)
            
                # if abs(v - nxt_values[i]) < diff:
                #     diff = abs(v - nxt_values[i])
            diff = min(diff, delta)
            if diff <= error:
                # ...
                break
            # -----------------------------------------------------------
            values = copy.copy(nxt_values)            
            print ("Value error: {}".format(diff))
        print ("error converged to {} after {} iter".format(diff, count))
 
        return values


    def learn(self, error=1e-9):
        """ 
        Run the value iteration to obtain a policy
        
        Parameters
        ----------
        error float: a convergence check threshold
        """
        
        values = self.solve_mdp(error)

        # generate stochastic policy
        policy = np.zeros([self.observation_dim, self.action_dim])
        #------------------------------------------------------------
        # Place your code here
        # -----------------------------------------------------------
        for s in range(self.observation_dim):
           for a in range(self.action_dim):
                self.get_grid_pos(s)
                new_pos = self.get_grid_pos(s) + self.actions[a]
                if self.collides(new_pos) or self.out_of_map(new_pos):
                    policy[s, a] = 0
                else:
                    policy[s, a] = values[self.get_grid_index(new_pos)]
                


                
        # -----------------------------------------------------------
        
        self.policy = policy
        self.values = values
        return self.policy, self.values

    def get_policy(self):
        """Get the current policy"""
        return self.policy

    def test(self):
        """test"""
        state = self.env.reset()
        path  = [self.env.state]

        self.env.render_value_map(self.values, self.grid_dim[0])
        while True:
            self.env.render()

            action = self.choose_action(state)
            print(action)
            next_state, reward, done, info = self.env.step(self.actions[action])
            state = next_state
            path.append(state)

            if done:
                self.env.close()
                break
        print(path)
        return path
    
    def save(self, filename='vi.pkl'):
        """Save the obtained state values"""
        d = {'values': self.values}
        pickle.dump(d, open(filename, "wb"))

    def load(self, filename='vi.pkl'):
        """Load the stored state values"""
        d = pickle.load(open(filename, "rb"))
        self.values = d['values']
        values = self.values
        # generate stochastic policy
        policy = np.zeros([self.observation_dim, self.action_dim])
        #------------------------------------------------------------
        # Place your code here
        # -----------------------------------------------------------
        for s in range(self.observation_dim):
            for a in range(self.action_dim):
                # print(s, self.get_grid_pos(s))#, self.get_grid_index(self.get_grid_pos(s)))
                
                self.get_grid_pos(s)
                new_pos = self.get_grid_pos(s) + self.actions[a]
                if self.collides(new_pos) or self.out_of_map(new_pos):
                    policy[s, a] = 0
                else:
                    policy[s, a] = values[self.get_grid_index(new_pos)]
                                
        # -----------------------------------------------------------
        
        self.policy = policy
    
