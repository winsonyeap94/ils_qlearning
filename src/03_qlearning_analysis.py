"""
This time, let's track and analyse our agent's behaviour.
This would help us better understand how our agent is learning and whether the rewards (logic) we are giving is 
helping the agent to learn better and faster.
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

env = gym.make("MountainCar-v0")

# ==========================================================================================
# Q-TABLES
# ==========================================================================================
# Let's set up our Q-Table and see hot it looks like
# It's impossible to cater for every possible value, hence we need to discretise our action space
DISCRETE_OS_SIZE = [20, 20]
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

# ==========================================================================================
# UPDATING OUR Q-VALUES (Q-LEARNING)
#
# new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
# https://pythonprogramming.net/static/images/reinforcement-learning/new-q-value-formula.png
# ==========================================================================================

# Setting up our Q-Learning Parameters
LEARNING_RATE = 0.1
DISCOUNT = 0.95 # Measure of how much we want to care about FUTURE reward rather than immediate reward
EPISODES = 4000  # How many iterations of the game we want to run

# Exploration Parameters
epsilon = 1  # Not a constant, going to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon // (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# [!] Tracking Parameters and Storage Variables
STATS_EVERY = 100  # How often to log our agent's stats
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

# Helper function for converting our environment state to match that in our Q-Table
discrete_os_window_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_window_size
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):

    print(f"============================== EPISODE : {episode} ==============================")

    # [!] Tracking of our episode rewards
    episode_reward = 0

    # Updating our Q-Values
    discrete_state = get_discrete_state(env.reset())

    done = False
    t = 0
    print(f"{t=} | State: {discrete_state}")

    while not done:

        t += 1

        # Now we will use epsilon to encourage exploration
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        print(f"{t=} | State: {new_discrete_state} | Action: {action} | Reward: {reward} | Done: {done}")

        # [!] Tracking of our episode rewards
        episode_reward += reward

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

        # Decaying our Epsilon parameter
        if START_EPSILON_DECAYING <= episode <= END_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value

    # [!] Consolidating an episode's rewards
    ep_rewards.append(episode_reward)
    if not episode % STATS_EVERY:
        average_reward = sum(ep_rewards[-STATS_EVERY:]) / STATS_EVERY
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
        print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')

    # [!] Exporting our results for each episode
    np.save(Path("./docs/qtable/", f"{episode}-qtable.npy"), q_table)

env.close()

# ==========================================================================================
# AGENT LEARNING ANALYSIS
# ==========================================================================================

# We can visualise how our rewards look like over time
fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
ax.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='average reward', color='navyblue')
ax.fill_between(x=aggr_ep_rewards['ep'],
                y1=aggr_ep_rewards['max'], y2=aggr_ep_rewards['min'], 
                facecolor='blue', alpha=0.5)
ax.grid()
fig.show()

# With the Q-Tables saved, we can also load them to check our agent's Q-Table growth over time
# https://pythonprogramming.net/q-learning-analysis-reinforcement-learning-python-tutorial/?completed=/q-learning-algorithm-reinforcement-learning-python-tutorial/

