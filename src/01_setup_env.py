import gym

# Initialise a Gym Environment
env = gym.make("MountainCar-v0")

# All Environments in OpenAI Gym have a pre-defined action space
# We can also create our own Gym Environment and we just have ot follow the template
print(f"Action Space: {env.action_space=}")
print(f"Number of action spaces: {env.action_space.n}")

# To start using an environment, we: 
# 1. Initialise our environment with gym.make(NAME), 
# 2. Reset our environment first using env.reset(), and then 
# 3. Take an action with env.step(ACTION).
env.reset()

done = False
while not done:
    action = 2  # Always take action=2 (go right)
    env.step(action)
    env.render()  # Visualises the environment

# When we perform a env.reset() or env.step(ACITON), we can actually get the information of the environment (state)
state = env.reset()

done = False
t = 0
print(f"{t=} | State: {state}")
while not done:
    t += 1
    action = 2
    new_state, reward, done, _ = env.step(action)
    print(f"{t=} | State: {new_state} | Reward: {reward} | Done: {done}")
    env.render()



