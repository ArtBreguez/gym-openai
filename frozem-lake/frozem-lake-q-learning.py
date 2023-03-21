import numpy as np
import gym
import random
from IPython.display import clear_output
import time

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="rgb_array")

action_space_size = env.action_space.n
state_space_size = env.observation_space.n

# Gera a tabela Q com zeros para cada estado e ação (15 estados e 4 ações)
q_table =  np.zeros((state_space_size, action_space_size))

num_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.25 #alpha
discount_rate = 0.99 #gamma

exploration_rate = 1 #epsilon
max_exploration_rate = 1 
min_exploration_rate = 0.01
exploration_decay_rate = 0.001 #epsilon decay rate

rewards_all_episodes = []

#Q-learning algorithm
for episode in range(num_episodes):
    state = env.reset()
    done = False
    rewards_current_episode = 0
    for step in range(max_steps_per_episode):

        # Exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:])
        else:
            action = env.action_space.sample()
            
        new_state, reward, terminated , info = env.step(action)
    
        # Update Q-table for Q(s,a)
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))
        
        state = new_state
        rewards_current_episode += reward
        
        # env.render()
        if done == True:
            break
        
        # Exploration rate decay
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
        rewards_all_episodes.append(rewards_current_episode)
 
env.close()
   
# Calculate and print the average reward per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
count = 1000
print("********Average reward per thousand episodes********\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000

# Print updated Q-table
print("********Q-table********\n")
print(q_table)
        
