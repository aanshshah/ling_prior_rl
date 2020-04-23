from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import input
from builtins import range
from builtins import object
from  malmo import MalmoPython
import json
import logging
import math
import os
import random
import sys
import time
from malmo import malmoutils
import numpy as np
import tensorflow as tf
import sys
from DQN import DQNSolver

sys.path = [x for x in sys.path if x != 'c:\\users\\zach_surf\\documents\\github\\pytorch-a2c-ppo-acktr-gail']
sys.path.append('../.')

# from malmo_specialized_env import MalmoEnvSpecial  
from malmo_env_env import MalmoEnvSpecial  

GAMMA = 0.99
# LEARNING_RATE = 1e-3 
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0
MEMORY_SIZE = 1000000
    
class DQNAgent(object):
    """Tabular Q-learning agent for discrete state/action spaces."""

    def __init__(self, actions,observation_space):
        self.actions = actions
        self.observation_space = observation_space
        self.dqn_solver = DQNSolver(observation_space, actions,MEMORY_SIZE,GAMMA)
        self.dqn_solver.old_network.model.set_weights(self.dqn_solver.network.model.get_weights()) 
        self.total_steps = 0
       

    def run(self,env,eval_mode=False):
        """run the agent on the world"""

        total_reward = 0
        current_r = 0
        step = 0
        
        action = None
        done = False
        state = env.reset()

        while not done:

            if self.total_steps % 500 == 0:
                self.dqn_solver.old_network.model.set_weights(self.dqn_solver.network.model.get_weights()) 

            period = 300000
            self.dqn_solver.exploration_rate = (1/(period**2))*((self.total_steps-period)**2)
            if self.total_steps > period: self.dqn_solver.exploration_rate = 0
            self.dqn_solver.exploration_rate = max(0.15,self.dqn_solver.exploration_rate) #

            if self.total_steps % 300 == 0:
                print(self.dqn_solver.exploration_rate)

            rnd = random.random()
            if not eval_mode and rnd < self.dqn_solver.exploration_rate:
                action = np.random.randint(0,self.actions)
            else:
                # print("acting")
                action = self.dqn_solver.act(state)
            next_state, current_r, done, _ = env.step(action)
            next_state = tf.convert_to_tensor(next_state,dtype=tf.float64) #np.reshape(next_state,(2,9,9)),

            if not eval_mode: 
                self.dqn_solver.remember(state, action, current_r, next_state, done)
                #if self.total_steps % 50 == 0: 
                self.dqn_solver.experience_replay()

                self.total_steps+=1
                step += 1

            total_reward += current_r
            state = next_state

        return total_reward

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python dqn_train_agent [pickaxe_stone/axe_log/shovel_clay/hoe_dirt/shears_sheep/sword_cow/sword_pig] <experiment name>")
        exit()
    
    task = sys.argv[1]
    experiment_name = task + "_" + sys.argv[2]
    eval_mode = False
    checkpoint = ""

    print("initializing environment...")
    # env = MalmoEnvSpecial("pickaxe_stone")
    # env = MalmoEnvSpecial("axe_log")
    env = MalmoEnvSpecial(task,port=9000)

    #env = MalmoEnvSpecial("sword_cow")

    # env.setup()

    
    agent = DQNAgent(actions=env.action_space.n,observation_space=env.observation_space)
    print(agent.dqn_solver.network.model.summary())
    if eval_mode:
        pass
        # agent.dqn_solver.network.model.load_weights('./checkpoints_sixth_try/checkpoint_800')
    agent.dqn_solver.network.model
    cumulative_rewards_val = []
    cumulative_rewards = []

    for e in range(100000):
        total_reward = agent.run(env,eval_mode=eval_mode)
        cumulative_rewards.append(total_reward)
        if e % 10 == 0:
            print(cumulative_rewards[-10:])
        if e % 100 == 0:
            if not eval_mode: 
                np.save(experiment_name+"_train.npy", cumulative_rewards)
        if e % 100 == 0:
            eval_reward = 0.0 
            for i in range(10):
                eval_reward += agent.run(env,eval_mode=True)
            cumulative_rewards_val.append(eval_reward/10.0)
            print("EVAL REWARDS:",cumulative_rewards_val)
            np.save(experiment_name+"_eval.npy", cumulative_rewards_val)


        if e % 100 == 0:
            if not eval_mode: agent.dqn_solver.network.model.save_weights('./'+experiment_name+'/checkpoint_'+str(e))

    print(cumulative_rewards)
