import numpy as np
from scipy.stats import poisson
from collections import defaultdict
import random
from tqdm import tqdm

# Constants
MAX_CARS = 20
MAX_MOVE = 5
RENTAL_REWARD = 10
MOVE_COST = 2
DISCOUNT = 0.9

RENTAL_REQUEST_FIRST_LOC = 3
RENTAL_REQUEST_SECOND_LOC = 4
RETURNS_FIRST_LOC = 3
RETURNS_SECOND_LOC = 2

    
def num_cars_requested():
    return np.random.poisson(RENTAL_REQUEST_FIRST_LOC), np.random.poisson(RENTAL_REQUEST_SECOND_LOC)

class JackCarRentalEnv:
    def __init__(self):
        self.states = [(i, j) for i in range(MAX_CARS + 1) for j in range(MAX_CARS + 1)]
        self.actions = list(range(-MAX_MOVE, MAX_MOVE + 1)) # action > 0 means move from first location to second location and vice versa

        self.state_values = defaultdict(float)
        self.initialize_state_values()

        self.policy = defaultdict(int)
        self.initialize_policy()

    def initialize_state_values(self):
        for state in self.states:
            self.state_values[state] = 0

    def initialize_policy(self):
        # TODO: Why this random initialization helps?
        for state in self.states:
            self.policy[state] = random.randint(-MAX_MOVE, MAX_MOVE)
        

    def constrain_move(self, state, action):
        if action > 0: 
            if state[0] - action < 0:
                action = state[0]
        elif action < 0:
            if state[1] + action < 0:
                action = -state[1]
        else:
            action = 0
            
        return action
        
    def constrain_return(self, state):
        if state[0] > MAX_CARS:
            state = (MAX_CARS, state[1])
        if state[1] > MAX_CARS:
            state = (state[0], MAX_CARS)
        return state

    def constrain_rental(self, state, cars_requested):
        if cars_requested[0] > state[0]:
            cars_requested = (state[0], cars_requested[1])
        if cars_requested[1] > state[1]:
            cars_requested = (cars_requested[0], state[1])
        return cars_requested
    
    def compute_reward(self, actual_cars_moved, actual_cars_requested):
        cost = MOVE_COST * abs(actual_cars_moved)
        revenue = RENTAL_REWARD * actual_cars_requested[0] + RENTAL_REWARD * actual_cars_requested[1]
        return revenue - cost
        
    def request_prob(self, state, cars_requested):
        if cars_requested[0] == state[0]:
            prob_1 = 1 - poisson.cdf(cars_requested[0], mu=RENTAL_REQUEST_FIRST_LOC)
        else:
            prob_1 = poisson.pmf(cars_requested[0], mu=RENTAL_REQUEST_FIRST_LOC)
        
        if cars_requested[1] == state[1]:
            prob_2 = 1 - poisson.cdf(cars_requested[1], mu=RENTAL_REQUEST_SECOND_LOC)
        else:
            prob_2 = poisson.pmf(cars_requested[1], mu=RENTAL_REQUEST_SECOND_LOC)
        
        return prob_1 * prob_2
    
    def return_prob(self, state, cars_returned):
        if cars_returned[0] + state[0] == MAX_CARS:
            prob_1 = 1 - poisson.cdf(cars_returned[0], mu=RETURNS_FIRST_LOC)
        else:
            prob_1 = poisson.pmf(cars_returned[0], mu=RETURNS_FIRST_LOC)
        
        if cars_returned[1] + state[1] == MAX_CARS:
            prob_2 = 1 - poisson.cdf(cars_returned[1], mu=RETURNS_SECOND_LOC)
        else:
            prob_2 = poisson.pmf(cars_returned[1], mu=RETURNS_SECOND_LOC)
        
        return prob_1 * prob_2
    
    def run_policy(self, state):
        return self.policy[state]
        
    def policy_evaluation(self):
        threshold = 0.1
        delta = float('inf')
        
        while delta >= threshold:
            delta = 0
            new_state_values = defaultdict(float)
            for state in tqdm(self.states):
                action = self.run_policy(state)
                
                v_val = 0
                
                # step 1: Move cars
                # state input: state before action
                # state output: state after action (dimension unchanged)
                actual_action = self.constrain_move(state, action)
                state_1 = (state[0] + actual_action, state[1] - actual_action)

                # step 2: Request cars
                # state input: state after action
                # state output: states before return (dimension expanded)
                state_2s = []
                for loc_1_num in range(state_1[0] + 1):
                    for loc_2_num in range(state_1[1] + 1):
                        state_2 = (state_1[0] - loc_1_num, state_1[1] - loc_2_num)
                        reward = self.compute_reward(actual_action, (loc_1_num, loc_2_num))
                        # compute the transition prob of request
                        prob_2 = self.request_prob(state_1, (loc_1_num, loc_2_num))
                        state_2s.append((state_2, (loc_1_num, loc_2_num), reward, prob_2))
                
                # step 3: Return cars
                # state input: state after request
                # state output: states after return (dimension expanded further)
                for state_2, cars_requested, reward, prob_2 in state_2s:
                    state_3s = []
                    max_ret1 = MAX_CARS - state_2[0]
                    max_ret2 = MAX_CARS - state_2[1]
                    for loc_1_num in range(0, max_ret1 + 1):
                        for loc_2_num in range(0, max_ret2 + 1):
                            state_3 = (state_2[0] + loc_1_num, state_2[1] + loc_2_num)
                            prob_3 = self.return_prob(state_2, (loc_1_num, loc_2_num))
                            v_val += prob_2 * prob_3 * (reward + DISCOUNT * self.state_values[state_3])
                
                new_state_values[state] = v_val
                delta = max(delta, abs(new_state_values[state] - self.state_values[state]))
            
            self.state_values = new_state_values
            print("delta: ", delta)
                    
        return new_state_values
        

    def policy_improvement(self):
        policy_stable = True
        for state in self.states:
            old_action = self.policy[state]
            q_vals = defaultdict(float)
            for action in self.actions:
                q_val = 0
                
                # step 1: Move cars
                # state input: state before action
                # state output: state after action (dimension unchanged)
                actual_action = self.constrain_move(state, action)
                state_1 = (state[0] + actual_action, state[1] - actual_action)

                # step 2: Request cars
                # state input: state after action
                # state output: states before return (dimension expanded)
                state_2s = []
                for loc_1_num in range(state_1[0] + 1):
                    for loc_2_num in range(state_1[1] + 1):
                        state_2 = (state_1[0] - loc_1_num, state_1[1] - loc_2_num)
                        reward = self.compute_reward(actual_action, (loc_1_num, loc_2_num))
                        # compute the transition prob of request
                        prob_2 = self.request_prob(state_1, (loc_1_num, loc_2_num))
                        state_2s.append((state_2, (loc_1_num, loc_2_num), reward, prob_2))
                
                # step 3: Return cars
                # state input: state after request
                # state output: states after return (dimension expanded further)
                for state_2, cars_requested, reward, prob_2 in state_2s:
                    state_3s = []
                    max_ret1 = MAX_CARS - state_2[0]
                    max_ret2 = MAX_CARS - state_2[1]
                    for loc_1_num in range(0, max_ret1 + 1):
                        for loc_2_num in range(0, max_ret2 + 1):
                            state_3 = (state_2[0] + loc_1_num, state_2[1] + loc_2_num)
                            prob_3 = self.return_prob(state_2, (loc_1_num, loc_2_num))
                            q_val += prob_2 * prob_3 * (reward + DISCOUNT * self.state_values[state_3])

                q_vals[action] = q_val
                
            best_action = max(q_vals, key=q_vals.get)
            if best_action != old_action and q_vals[best_action] > q_vals[old_action]:
                self.policy[state] = best_action
                policy_stable = False
            
        return policy_stable
    
    def policy_iteration(self):
        iteration = 1
        while True:
            print(f"Iteration {iteration}")
            self.policy_evaluation()
            policy_stable = self.policy_improvement()
            if policy_stable:
                break
            iteration += 1
            
                
        return self.policy, self.state_values

if __name__ == "__main__":
    env = JackCarRentalEnv()
    policy, state_values = env.policy_iteration()
    print(policy)
    print(state_values)
