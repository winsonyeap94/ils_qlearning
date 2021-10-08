"""
Continuing from our earlier example from 02_qlearning.py, we find that our model isn't really learning. It either 
does not reach the top, or only reaches after a long time (large number of episodes). 

As an Agent learns an environment, it moves from "exploration" to "exploitation." Right now, our model is greedy and 
exploiting for max Q values always, but these Q values are worthless right now. We need the agent to explore!

We do this by applying EPSILON!
"""

import gym
import numpy as np

env = gym.make("MountainCar-v0")

# ==========================================================================================
# Q-TABLES
# ==========================================================================================
# What's a Q Table!?
#
# The way Q-Learning works is there's a "Q" value per action possible per state. This creates a table. 
# In order to figure out all of the possible states, we can either query the environment (if it is kind enough to 
# us to tell us)...or we just simply have to engage in the environment for a while to figure it out.
# 
# In our case, we can query the enviornment to find out the possible ranges for each of these state values:
print(env.observation_space.high)
print(env.observation_space.low)

# Let's set up our Q-Table and see hot it looks like
# It's impossible to cater for every possible value, hence we need to discretise our action space
DISCRETE_OS_SIZE = [20, 20]
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
print(q_table)

# As we can see, the Q-Table is a 20x20x3 matrix, which has initialized random Q values for us.
# The 20 x 20 bit is every combination of the bucket slices of all possible states. 
# The x3 bit is for every possible action we could take.

# ==========================================================================================
# UPDATING OUR Q-VALUES (Q-LEARNING)
#
# new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
# https://pythonprogramming.net/static/images/reinforcement-learning/new-q-value-formula.png
# ==========================================================================================

# Setting up our Q-Learning Parameters
LEARNING_RATE = 0.1
DISCOUNT = 0.95 # Measure of how much we want to care about FUTURE reward rather than immediate reward
EPISODES = 25000  # How many iterations of the game we want to run

# [!] Exploration Parameters
epsilon = 1  # Not a constant, going to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon // (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# Helper function for converting our environment state to match that in our Q-Table
discrete_os_window_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_window_size
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):

    print(f"============================== EPISODE : {episode} ==============================")

    # Updating our Q-Values
    discrete_state = get_discrete_state(env.reset())

    done = False
    t = 0
    print(f"{t=} | State: {discrete_state}")

    while not done:

        t += 1

        # [!] Now we will use epsilon to encourage exploration
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        print(f"{t=} | State: {new_discrete_state} | Action: {action} | Reward: {reward} | Done: {done}")

        # Let's only show once every few episodes
        if episode % 1000 == 0:
            env.render()

        # Now, we want to update our Q-Values in the Q-Table as well (Q-Learning)
        if not done:
            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])

            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action, )]

            # And here's our equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # Update our Q-Table with the new Q-Value
            q_table[discrete_state + (action, )] = new_q

        # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0

        # Resetting the discrete_state variable
        discrete_state = new_discrete_state

        # [!] Decaying our Epsilon parameter
        if START_EPSILON_DECAYING <= episode <= END_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value

env.close()

