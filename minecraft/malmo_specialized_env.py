from __future__ import print_function
# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

# The "Cliff Walking" example using Q-learning.
# From pages 148-150 of:
# Richard S. Sutton and Andrews G. Barto
# Reinforcement Learning, An Introduction
# MIT Press, 1998

import malmo.minecraftbootstrap

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
import gym
import sys
import time
from malmo import malmoutils
import numpy as np
import tensorflow as tf
import math



class MalmoEnvSpecial(gym.Env):
    def checkInventoryForItem(self,obs, requested):
        for i in range(0,9):#39):
            key = 'InventorySlot_'+str(i)+'_item'
            if key in obs:
                item = obs[key]
                if item == requested:
                    return True
        return False

    def obs_to_vector(self,world_state,use_entities=True):

        state_data_raw = json.loads(world_state.observations[-1].text)
        
        if 'floor9x9' in state_data_raw:
            state_data_raw = state_data_raw['floor9x9']
        else:
            print("FAILED") 
            return np.zeros((2,self.observation_space.shape[-2],self.observation_space.shape[-1])) #self.observation_space

        state_data = [self.state_map[block] for block in state_data_raw]

        state_data =np.reshape(np.array(state_data,dtype=np.float64),(1,9,9))
        if use_entities: 
            entity_data = self.obs_to_ent_vector(world_state)
            state_data = np.concatenate((state_data,entity_data),axis=0)
        return np.expand_dims(state_data,0)

    def obs_to_ent_vector(self,world_state,relevant_entities={"diamond_pickaxe","cobblestone"}):

        entity_data = json.loads(world_state.observations[-1].text)['entities']
        # print(entity_data)
        player_data = [x for x in entity_data if x["name"]=="agent"][0]
        entities =  [x for x in entity_data if x["name"] in relevant_entities]
        
        entity_states = np.zeros((1,self.observation_space.shape[-2],self.observation_space.shape[-1]))
        # print(self.observation_space.shape)
        zero_x = entity_states.shape[2]
        zero_x = zero_x // 2 
        zero_z = entity_states.shape[1]
        zero_z = zero_z // 2 

        player_loc = int(player_data['x']),int(player_data['z'])

        # print("shifts",zero_x,zero_z)

        for e in entities:
            entity_loc = int(e['x']),int(e['z'])
            relative_x = entity_loc[0] - player_loc[0] + 1 + zero_x
            relative_z = entity_loc[1] - player_loc[1] + 1 +  zero_z
            # print("coords",relative_x,relative_z)
            entity_states[0][min(int(relative_z),8)][min(int(relative_x),8)] = self.entity_map[e["name"]]
        return entity_states #np.transpose(entity_states,(0,2,1))

    def __init__(self):
        malmoutils.fix_print()
        metadata = {'render.modes': ['human']}

        self.state_map = {"air":0,"bedrock":1,"stone":2}
        self.entity_map = {"diamond_pickaxe":1,"cobblestone":2}

        self.agent_host = MalmoPython.AgentHost()
        # Find the default mission file by looking next to the schemas folder:
        # schema_dir = None
        # try:
        #     schema_dir = "C:\\Users\\zach_surf\\Documents\\GitHub\\malmo\\MalmoPlatform\\Schemas" #os.environ['MALMO_XSD_PATH']
        # except KeyError:
        #     print("MALMO_XSD_PATH not set? Check environment.")
        #     exit(1)
        # self.mission_file =  os.path.abspath('get_stone_pickaxe.xml')
      

        # add some args
        # self.agent_host.addOptionalStringArgument('recording_dir,r',"","malmo_recording")

        # self.agent_host.addOptionalStringArgument('mission_file',
            # 'Path/to/file from which to load the mission.', self.mission_file)
        self.agent_host.addOptionalFloatArgument('alpha',
            'Learning rate of the Q-learning agent.', 0.1)
        self.agent_host.addOptionalFloatArgument('epsilon',
            'Exploration rate of the Q-learning agent.', 0.01)
        self.agent_host.addOptionalFloatArgument('gamma', 'Discount factor.', 1.0)
        self.agent_host.addOptionalFlag('load_model', 'Load initial model from model_file.')
        self.agent_host.addOptionalStringArgument('model_file', 'Path to the initial model file', '')
        self.agent_host.addOptionalFlag('debug', 'Turn on debugging.')

        malmoutils.parse_command_line(self.agent_host,["--recording_dir","malmo_recording"]) 
        # self.actionSet = ["attack 1","attack 0", "move 1", "move -1", "turn 1", "turn -1"] #,"movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]
        self.actions =["movenorth 1","movesouth 1", "movewest 1", "moveeast 1"] #,"strafe 1","strafe -1"] #,"attack 1","attack 0"]
        self.observation_space = np.zeros((2,9,9))
        self.action_space = gym.spaces.Discrete(len(self.actions))
        # agent = DQNAgent(
        #         actions=actionSet,
        #         epsilon=agent_host.getFloatArgument('epsilon'),
        #         alpha=agent_host.getFloatArgument('alpha'),
        #         gamma=agent_host.getFloatArgument('gamma'),
        #         debug = agent_host.receivedArgument("debug"),
        #         canvas = None,
        #         root = None)

        self.my_clients = MalmoPython.ClientPool()
        self.my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available

        self.max_retries = 3
        self.agentID = 0
        self.num_steps = 0
        self.max_steps = 250
        # expID = 'tabular_q_learning'

        # num_repeats = 1500000

    def step(self, action):
        world_state = self.agent_host.getWorldState()

        self.agent_host.sendCommand(self.actions[action])

        # time.sleep(0.1)
        while world_state.is_mission_running:
            world_state = self.agent_host.getWorldState()
            if world_state.number_of_observations_since_last_state > 0: # and world_state.number_of_rewards_since_last_state > 0:
                break

        # malmo_env_rewards = sum([x.getValue() for x in world_state.rewards])

        still_running = world_state.is_mission_running and self.num_steps < self.max_steps - 1



        if still_running and self.checkInventoryForItem(json.loads(world_state.observations[-1].text),"cobblestone"):
            done = True
            reward = 100
        else:
            done = not still_running
            reward = -0.1
        
        obs = self.obs_to_vector(world_state) if world_state.is_mission_running else self.observation_space
        info = {}

        self.num_steps+=1

        return obs, reward, done, info

    
    def reset(self):
            # mission_file = agent_host.getStringArgument('mission_file')
            self.num_steps = 0
            self.agent_host.sendCommand("quit")
            time.sleep(0.1)
            # with open(self.mission_file, 'r') as f:
            #     print("Loading mission from %s" % self.mission_file)
            mission_xml = self.make_env_string()

            my_mission = MalmoPython.MissionSpec(mission_xml, True)
            my_mission.allowAllDiscreteMovementCommands()
            
            # print("\nMap %d - Mission %d of %d:" % ( imap, i+1, num_repeats ))
            my_mission_record = malmoutils.get_default_recording_object(self.agent_host, "./save_%s-map%d-rep%d" % (0,0,0)) #(expID, imap, i))
        # self.agent_host.addOptionalFloatArgument('recording_dir',"malmo_recording")
                    # my_mission.drawBlock(0,204,0,"stone")
            # my_mission.drawBlock(2,204,2,"stone")

            my_mission.drawCuboid(-3,203,-3,3,203,3,"bedrock")    
            my_mission.drawCuboid(-30,203,-10,-3,227,10,"bedrock")
            my_mission.drawCuboid(3,203,-10,30,227,10,"bedrock")  
            my_mission.drawCuboid(-3,203,3,3,227,10,"bedrock")  
            my_mission.drawCuboid(-3,203,-10,3,227,-3,"bedrock")

            my_mission.drawCuboid(-2,204,-2,2,270,2,"air") 
            my_mission.drawItem(2,206,2,"diamond_pickaxe")
            my_mission.drawBlock(random.randint(0,4)-2,204,random.randint(1,3)-2,"stone")
            for retry in range(self.max_retries):
                try:
                             # <DrawItem    x="4"   y="46"  z="12" type="diamond_axe" /> 

                    # my_mission.drawItem(random.randint(0,5)-2,206,random.randint(0,5)-2,"diamond_pickaxe")
          
                    self.agent_host.startMission( my_mission, self.my_clients, my_mission_record, self.agentID, "%s-%d" % (0,0) )#(expID, i) )
                    break
                except RuntimeError as e:
                    if retry == self.max_retries - 1:
                        print("Error starting mission:",e)
                        exit(1)
                    else:
                        time.sleep(2.5)     

            print("Waiting for the mission to start", end=' ')
            world_state = self.agent_host.getWorldState()
            while not world_state.has_mission_begun:
                print(".", end="")
                time.sleep(0.1)
                world_state = self.agent_host.getWorldState()
                for error in world_state.errors:
                    print("Error:",error.text)
            while world_state.is_mission_running:
                world_state = self.agent_host.getWorldState()
                if world_state.number_of_observations_since_last_state > 0:
                    self.agent_host.sendCommand("attack 1")
                    return self.obs_to_vector(world_state)
            return None

    def setup(self,malmo_path='C:/Users/zach_surf/Documents/GitHub/malmo'):
        print("starting server...")
        current_dir = os.getcwd()
        os.chdir(malmo_path)
        malmo.minecraftbootstrap.set_malmo_xsd_path();
        malmo.minecraftbootstrap.launch_minecraft()
        os.chdir(current_dir)

    def render(self, mode='human', close=False):
        pass

    def make_env_string(self):
        base = '<?xml version="1.0" encoding="UTF-8" standalone="no" ?><Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        base+='<About><Summary>Running pickaxe-stone mission...</Summary></About>'
        base+= '<ModSettings><MsPerTick>1</MsPerTick></ModSettings>'
        base+= '<ServerSection><ServerInitialConditions><Time><StartTime>6000</StartTime><AllowPassageOfTime>false</AllowPassageOfTime>'
        base+= '</Time><Weather>clear</Weather><AllowSpawning>false</AllowSpawning></ServerInitialConditions><ServerHandlers><FlatWorldGenerator />' 
        base+= '<DrawingDecorator></DrawingDecorator>'
        base+= '<ServerQuitFromTimeUp timeLimitMs="10000000"/><ServerQuitWhenAnyAgentFinishes/></ServerHandlers></ServerSection>'
        base+= '<AgentSection mode="Survival"><Name>agent</Name><AgentStart><Placement x="-1.5" y="204" z="-1.5" pitch="50" yaw="0"/>'
        base+= '<Inventory></Inventory>'

        base+='</AgentStart>'
        base+='<AgentHandlers>'
        base+='<ObservationFromGrid> <Grid name="floor9x9"> <min x="-4" y="0" z="-4"/> <max x="4" y="0" z="4"/> </Grid> </ObservationFromGrid>'

        base+='<ObservationFromNearbyEntities><Range name="entities" xrange="5" yrange="3" zrange="5"/></ObservationFromNearbyEntities>'
        base+='<ObservationFromFullInventory/><ObservationFromFullStats/><VideoProducer want_depth="false"><Width>640</Width><Height>480</Height></VideoProducer>'
        base+='<DiscreteMovementCommands><ModifierList type="deny-list"><command>attack</command></ModifierList></DiscreteMovementCommands>'
        base+='<ContinuousMovementCommands><ModifierList type="allow-list"><command>attack</command></ModifierList>'
        base+='</ContinuousMovementCommands><MissionQuitCommands quitDescription="done"/>'
        base+='</AgentHandlers></AgentSection></Mission>'
        return base

if __name__ == "__main__":
    print("starting server...")
    current_dir = os.getcwd()
    os.chdir('C:/Users/zach_surf/Documents/GitHub/malmo')
    malmo.minecraftbootstrap.set_malmo_xsd_path();
    malmo.minecraftbootstrap.launch_minecraft()
    os.chdir(current_dir)

    print("initializing environment...")
    env = MalmoEnvSpecial()
    obs = env.reset()
    for step in range(5):
        print("\n",step)
        command = int(input())
        obs, reward, done, info = env.step(command)
        print(obs[0])
        print(reward)
        # time.sleep(1)


  



