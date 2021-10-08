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

        # Instead of action=2, we select our action based on the Q-Table
        action = np.argmax(q_table[discrete_state])
        
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        print(f"{t=} | State: {new_discrete_state} | Action: {action} | Reward: {reward} | Done: {done}")

        # Let's only show once every few episodes
        if episode % 100 == 0:
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


    env.close()

