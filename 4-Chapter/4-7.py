import numpy as np
from scipy.stats import poisson

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
        self.actions = list(range(-max_move, max_move + 1)) # action > 0 means move from first location to second location and vice versa
        
    def constrain_move(self, state, action):
        
        return
        
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
    
    def compute_transition_prob(self, cars_returned, cars_requested):
        prob_1_return = poisson.pmf(cars_returned[0], mu=RETURNS_FIRST_LOC)
        prob_2_return = poisson.pmf(cars_returned[1], mu=RETURNS_SECOND_LOC)
        prob_1_request = poisson.pmf(cars_requested[0], mu=RENTAL_REQUEST_FIRST_LOC)
        prob_2_request = poisson.pmf(cars_requested[1], mu=RENTAL_REQUEST_SECOND_LOC)
        
        return prob_1_return * prob_2_return * prob_1_request * prob_2_request

    def compute_reward(self, actual_cars_requested, cars_moved):
        return RENTAL_REWARD * actual_cars_requested[0] + RENTAL_REWARD * actual_cars_requested[1] - MOVE_COST * abs(actual_cars_requested[0] - cars_requested[0]) - MOVE_COST * abs(actual_cars_requested[1] - cars_requested[1])

    def transition_prob_and_reward(self, state, action):
        """ compute the transition probability and reward
        Input: 
            state: (n_car_1, n_car_2)
            action: (n_move, target_loc)
        Output:
            dict: 
                key: next_state
                value: (transition_prob, reward)
        """
        cars_returned = np.random.poisson(RETURNS_FIRST_LOC), np.random.poisson(RETURNS_SECOND_LOC)
        cars_requested = np.random.poisson(RENTAL_REQUEST_FIRST_LOC), np.random.poisson(RENTAL_REQUEST_SECOND_LOC)   
        
        # after action
        next_state_1 = (state[0] + action, state[1] - action)
        next_state_1 = self.constrain_state(next_state_1)
        
        # after return
        next_state_2 = (next_state_1[0] + cars_returned[0], next_state_1[1] + cars_returned[1])
        next_state_2 = self.constrain_return(next_state_2)
        
        # after request
        actual_cars_requested = self.constrain_rental(next_state_2, cars_requested)       
        next_state_3 = (next_state_2[0] - actual_cars_requested[0], next_state_2[1] - actual_cars_requested[1])

        
        # Implementation question, do I need to consider the return time for each individual car? Answer: No, the Poisson distribution of cars in and out took care of it.
        # However, in this case, doesn't it mean that the car rental doesn't affect the rewards? Since the return probability is deterministic and doesn't depend on the state. 
        # Nevermind, I just realized that the reward are dependent on the car rental request, which does depend on the state.
        
        
        
        
