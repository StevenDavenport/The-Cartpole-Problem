import os
import random
import gym
import numpy as np
from collections import deque
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop
from itertools import count
import random


from gym.envs.classic_control.cartpole import *
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time


class Agent():
    def __init__(self, env):
        self.env = env
        self.observation_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.memory = deque(maxlen=2000)
        self.learning_rate = 1e-4  
        self.epsilon = 1.0  
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.gamma = 0.95
        self.batch_size = 64
        self.train_start = 1000
        self.steps_taken = 0
        self.update_target = 20
        self.moving_avg_period = 25
        self.moving_avg = []
        self.episode_rewards = []
        self.model = None
        self.target_model = None
        self.score_save_limit = 500

    def build_model(self):
        model = Sequential()
        model.add(Dense(512, input_dim=self.observation_size, activation="relu", kernel_initializer='he_uniform'))
        model.add(Dense(256, activation="relu", kernel_initializer='he_uniform'))
        model.add(Dense(64, activation="relu", kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation="linear", kernel_initializer='he_uniform'))
        model.compile(loss="mse", optimizer=RMSprop(learning_rate=self.learning_rate, rho=0.95, epsilon=0.01), metrics=["accuracy"])
        return model
    
    def take_action(self, state):
        if self.model == None: 
            self.model = load_model('C:/Users/Steven/Desktop/cartpole_solution/cartpole/models/best_model.h5')
        state = np.squeeze(state).reshape(1,4)
        return np.argmax(self.model.predict(state))     # Exploit Enviroment

    def plot(self):
        plt.figure(1)
        plt.clf()        
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        self.moving_avg.append(sum(self.episode_rewards[-self.moving_avg_period:]) // self.moving_avg_period if len(self.episode_rewards) >= self.moving_avg_period else 0)
        plt.plot(self.episode_rewards)
        plt.plot(self.moving_avg)    
        plt.pause(0.001)
        clear_output(wait=True)

    def learn(self, episodes):
        self.model = self.build_model()
        self.target_model = self.build_model()

        for episode in range(episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.observation_size])
            done = False
            for step in count():
                self.env.render()
                # Decide an action
                action = None
                if random.random() <= self.epsilon:
                    action = random.randrange(self.action_size)
                else:
                    action = np.argmax(self.model.predict(state))
                # Make an action
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.observation_size])
                if not done: # or i == self.env._max_episode_steps-1:
                    reward = reward
                else:
                    reward = -100
                # Add experience to replay memory
                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                # Update Epsilon
                if len(self.memory) > self.train_start:
                    if self.epsilon > self.epsilon_min:
                        self.epsilon *= self.epsilon_decay
                # Update Target Network
                if self.steps_taken % self.update_target == 0: 
                    self.target_model.set_weights(self.model.get_weights())   
                # Check if episode is done
                if done:                   
                    print("episode: {}/{}, score: {}, e: {:.2}".format(episode, episodes, step, self.epsilon))
                    self.episode_rewards.append(step)
                    self.plot()
                    if step >= self.score_save_limit:
                        self.model.save('model' + str(step) +'.h5')
                    break
                # Create a batch from replay memory
                if len(self.memory) >= self.train_start:
                    batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
                    state_batch, next_state_batch = [], []
                        # put states and next states in a list
                    for _state, _action, _reward, _next_state, _done in batch:
                        state_batch.append(_state)
                        next_state_batch.append(_next_state)
                        # get the reward and next rewards
                    state_batch = np.array(state_batch).reshape(self.batch_size, self.observation_size)
                    next_state_batch = np.array(next_state_batch).reshape(self.batch_size, self.observation_size)
                    target = self.model.predict(state_batch)
                    target_next = self.target_model.predict(next_state_batch)
                        # get q values
                    x = 0
                    for _state, _action, _reward, _next_state, _done in batch:
                        if _done:
                            target[x][_action] = _reward
                        else:
                            target[x][_action] = _reward + self.gamma * (np.amax(target_next[x]))
                        x += 1
                    # Train the network
                    self.model.fit(state_batch, target, batch_size=self.batch_size, verbose=0)

bool_do_not_quit = True  # Boolean to quit pyglet
scores = []  # Your gaming score
a = 0  # Action
env = CartPoleEnv()
number_of_trials = 50

def key_press(k, mod):
    global bool_do_not_quit, a, restart

def run_cartPole_asAgent(agent):

    env.reset()
    env.render()
    env.viewer.window.on_key_press = key_press

    for _ in range(number_of_trials):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        t1 = time.time()  # Trial timer
        while bool_do_not_quit:
            #this is where policy function outputs action a based on the current state
            action = agent.take_action(state)
            #this is there you get the next system state after take action a
            state, reward, done, info = env.step(action)
            time.sleep(1/10)  # 10fps: Super slow for us poor little human!
            total_reward += reward
            steps += 1
            env.render()
            if done or restart:
                t1 = time.time()-t1
                scores.append(total_reward)
                print("Episode", len(scores), "| Score:", total_reward, '|', steps, "steps | %0.2fs."% t1)
                break
    env.close()
    

if __name__ == '__main__':
    env = CartPoleEnv()
    agent = Agent(env)
    #run_cartPole_asAgent(agent)  
    agent.learn(100000)
