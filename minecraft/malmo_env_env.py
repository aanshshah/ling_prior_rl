# from __future__ import print_function

# import malmo.minecraftbootstrap

# from future import standard_library
# standard_library.install_aliases()
# from builtins import input
# from builtins import range
# from builtins import object
# from  malmo import MalmoPython
import json
# import logging
import math
import os
import random
import gym
import sys
import time
# import math
# from malmo import malmoutils
import numpy as np
# import tensorflow as tf
# import math


import malmoenv
import malmoenv.bootstrap

import argparse
import time

class MalmoEnvSpecial(gym.Env):
    def checkInventoryForItem(self,obs, requested):
        for i in range(0,9):#39): #Checks primary inventory
            key = 'InventorySlot_'+str(i)+'_item'
            if key in obs:
                item = obs[key]
                if item == requested:
                    return True
        return False

    def checkBlockExists(self,obs, requested):
        return 'floor9x9' in obs and requested in obs['floor9x9']

    def obs_to_vector(self,world_state,use_entities=True,flatten=True,expand_dims=True):

        state_data_raw = json.loads(world_state) #.observations[-1].text)
        
        if 'floor9x9' in state_data_raw:
            state_data_raw = state_data_raw['floor9x9']
        else:
            print("FAILED") 
            return np.zeros((1,2,self.observation_space.shape[-2],self.observation_space.shape[-1])) #self.observation_space

        state_data = [self.state_map[block] if block in self.state_map else 0 for block in state_data_raw]

        state_data =np.reshape(np.array(state_data,dtype=np.float64),(1,9,9))
        if use_entities: 
            entity_data = self.obs_to_ent_vector(world_state,self.relevant_entities)
            if flatten:
                print(entity_data)
                state_data[np.nonzero(entity_data)] = 0
                state_data = state_data + entity_data
            else:
                state_data = np.concatenate((state_data,entity_data),axis=0)

        return np.expand_dims(state_data,0) if expand_dims else state_data
 
    def fix_player_location(self,world_state):
        if len(world_state.observations) > 0:
            entity_data = json.loads(world_state.observations[-1].text)['entities']
            player_data = [x for x in entity_data if x["name"]=="agent"][0]
            player_loc = (player_data['x'],player_data['z'])

            if abs(math.floor(player_loc[0])-player_loc[0]) != 0.5:
                new_x = round(player_loc[0]-0.5)+0.5
                self.agent_host.sendCommand("tpx {}".format(new_x))
                print("FIXED X")
            if abs(math.floor(player_loc[1])-player_loc[1]) != 0.5:
                new_z = round(player_loc[1]-0.5)+0.5
                self.agent_host.sendCommand("tpz {}".format(new_z))
                print("FIXED Z")

    def obs_to_ent_vector(self,world_state,relevant_entities):

        entity_data = json.loads(world_state)['entities']
        # print(entity_data)
        player_data = [x for x in entity_data if x["name"]=="agent"][0]
        # print(entity_data)
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
            entity_states[0][math.floor(relative_z)][math.floor(relative_x)] = self.entity_map[e["name"]]
        return entity_states #np.transpose(entity_states,(0,2,1))


    def load_mission_param(self,mission_type):
        mission_dict = {}
        if mission_type == "pickaxe_stone":
            mission_dict["state_map"] = {"air":0,"bedrock":1,"stone":2}
            mission_dict["entity_map"] = {"diamond_pickaxe":3,"cobblestone":4}
            mission_dict["relevant_entities"] = {"diamond_pickaxe","cobblestone"}
            mission_dict["goal"] = "cobblestone"
            mission_dict["step_cost"] = -0.1
            mission_dict["goal_reward"] = 100
            mission_dict["max_steps"] = 150

        elif mission_type == "axe_log":
            mission_dict["state_map"] = {"air":0,"bedrock":1,"log":2}
            mission_dict["entity_map"] = {"diamond_axe":3,"log":4}
            mission_dict["relevant_entities"] = set(mission_dict["entity_map"].keys())
            mission_dict["goal"] = "log"
            mission_dict["step_cost"] = -0.1
            mission_dict["goal_reward"] = 100
            mission_dict["max_steps"] = 150

        elif mission_type == "shovel_clay":
            mission_dict["state_map"] = {"air":0,"bedrock":1,"clay":2}
            mission_dict["entity_map"] = {"diamond_shovel":3,"clay":4}
            mission_dict["relevant_entities"] = set(mission_dict["entity_map"].keys())
            mission_dict["goal"] = "clay"
            mission_dict["step_cost"] = -0.1
            mission_dict["goal_reward"] = 100
            mission_dict["max_steps"] = 150

        elif mission_type == "hoe_farmland":
            mission_dict["state_map"] = {"air":0,"bedrock":1,"dirt":2,"farmland":6}
            mission_dict["entity_map"] = {"diamond_hoe":3,"dirt":4,"farmland":5}
            mission_dict["relevant_entities"] = set(mission_dict["entity_map"].keys())
            mission_dict["goal"] = "farmland"
            mission_dict["step_cost"] = -0.1
            mission_dict["goal_reward"] = 100
            mission_dict["max_steps"] = 150

        elif mission_type == "bucket_water":
            mission_dict["state_map"] = {"air":0,"bedrock":1,"water":2}
            mission_dict["entity_map"] = {"bucket":3,"water_bucket":4}
            mission_dict["relevant_entities"] = set(mission_dict["entity_map"].keys())
            mission_dict["goal"] = "water_bucket"
            mission_dict["step_cost"] = -0.1
            mission_dict["goal_reward"] = 100
            mission_dict["max_steps"] = 150


        elif mission_type == "sword_pig":
            mission_dict["state_map"] = {"air":0,"bedrock":1}
            mission_dict["entity_map"] = {"diamond_sword":1,"Pig":2}
            mission_dict["relevant_entities"] = set(mission_dict["entity_map"].keys())
            mission_dict["goal"] = "porkchop"
            mission_dict["step_cost"] = -0.1
            mission_dict["goal_reward"] = 100
            mission_dict["max_steps"] = 250

        elif mission_type == "sword_cow":
            mission_dict["state_map"] = {"air":0,"bedrock":1}
            mission_dict["entity_map"] = {"diamond_sword":2,"Cow":3}
            mission_dict["relevant_entities"] = set(mission_dict["entity_map"].keys())
            mission_dict["goal"] = "beef"
            mission_dict["step_cost"] = -0.1
            mission_dict["goal_reward"] = 100
            mission_dict["max_steps"] = 250

        elif mission_type == "shears_sheep":
            mission_dict["state_map"] = {"air":0,"bedrock":1}
            mission_dict["entity_map"] = {"shears":2,"Sheep":3}
            mission_dict["relevant_entities"] = set(mission_dict["entity_map"].keys())
            mission_dict["goal"] = "wool"
            mission_dict["step_cost"] = -0.1
            mission_dict["goal_reward"] = 100
            mission_dict["max_steps"] = 100

        if len(mission_dict) == 0:
            print("Invalid mission name:",mission_type)


        return mission_dict

    def build_arena(self):
        arena = ""
        arena+= '<DrawCuboid x1="{}" y1="{}" z1="{}" x2="{}" y2="{}" z2="{}" type="bedrock" />'.format(-3,203,-3,3,203,3)
        arena+= '<DrawCuboid x1="{}" y1="{}" z1="{}" x2="{}" y2="{}" z2="{}" type="bedrock" />'.format(-30,203,-10,-3,227,10)
        arena+= '<DrawCuboid x1="{}" y1="{}" z1="{}" x2="{}" y2="{}" z2="{}" type="bedrock" />'.format(3,203,-10,30,227,10)
        arena+= '<DrawCuboid x1="{}" y1="{}" z1="{}" x2="{}" y2="{}" z2="{}" type="bedrock" />'.format(-3,203,-10,3,227,-3)
        arena+= '<DrawCuboid x1="{}" y1="{}" z1="{}" x2="{}" y2="{}" z2="{}" type="bedrock" />'.format(-3,203,3,3,227,10)
        arena+= '<DrawCuboid x1="{}" y1="{}" z1="{}" x2="{}" y2="{}" z2="{}" type="air" />'.format(-2,204,-2,2,270,2)
         
        return arena

    def get_mission_xml(self,mission_type):
         arena_xml = self.build_arena()

         if mission_type == "pickaxe_stone":  

             mission_xml = self.make_env_string(self.mission_type,[arena_xml,'<DrawBlock x="{}" y="{}" z="{}" type="stone" />'.format(random.randint(0,4)-2,204,random.randint(1,3)-2),'<DrawItem x="{}" y="{}" z="{}" type="diamond_pickaxe" />'.format(2,206,2) ])
 
         elif  mission_type == "axe_log":   
             mission_xml = self.make_env_string(self.mission_type,[arena_xml,'<DrawBlock x="{}" y="{}" z="{}" type="log" />'.format(random.randint(0,4)-2,204,random.randint(1,3)-2),'<DrawItem x="{}" y="{}" z="{}" type="diamond_axe" />'.format(2,206,2) ])

         elif  mission_type == "shovel_clay":   

             mission_xml = self.make_env_string(self.mission_type,[arena_xml,'<DrawBlock x="{}" y="{}" z="{}" type="clay" />'.format(random.randint(0,4)-2,204,random.randint(1,3)-2),'<DrawItem x="{}" y="{}" z="{}" type="diamond_shovel" />'.format(2,206,2) ])

         elif  mission_type == "hoe_farmland":   
             mission_xml = self.make_env_string(self.mission_type,[arena_xml,'<DrawBlock x="{}" y="{}" z="{}" type="dirt" />'.format(random.randint(0,4)-2,204,random.randint(1,3)-2),'<DrawItem x="{}" y="{}" z="{}" type="diamond_hoe" />'.format(2,206,2) ])

         elif  mission_type == "bucket_water":   
             mission_xml = self.make_env_string(self.mission_type,[arena_xml,'<DrawBlock x="{}" y="{}" z="{}" type="water" />'.format(random.randint(0,4)-2,204,random.randint(1,3)-2),'<DrawItem x="{}" y="{}" z="{}" type="bucket" />'.format(2,206,2) ])

         elif  mission_type == "sword_pig":     
             pig_pos = (random.randint(1,3)-2,random.randint(1,3)-2)
             mission_xml = self.make_env_string(self.mission_type,[arena_xml,'<DrawEntity x="{}" y="{}" z="{}" type="Pig" />'.format(cow_pos[0],204,cow_pos[1]),'<DrawItem x="{}" y="{}" z="{}" type="diamond_sword" />'.format(2,206,2) ])

         elif  mission_type == "sword_cow":     
             cow_pos = (random.randint(1,3)-2,random.randint(1,3)-2)
             mission_xml = self.make_env_string(self.mission_type,[arena_xml,'<DrawEntity x="{}" y="{}" z="{}" type="Cow" />'.format(cow_pos[0],204,cow_pos[1]),'<DrawItem x="{}" y="{}" z="{}" type="diamond_sword" />'.format(2,206,2) ])

         elif  mission_type == "shears_sheep":     
             sheep_pos = (random.randint(1,3)-2,random.randint(1,3)-2)
             mission_xml = self.make_env_string(self.mission_type,[arena_xml,'<DrawEntity x="{}" y="{}" z="{}" type="Sheep" />'.format(sheep_pos[0],204,sheep_pos[1]),'<DrawItem x="{}" y="{}" z="{}" type="shears" />'.format(2,206,2) ])

         return mission_xml

    def __init__(self,mission_type,port):
        # malmoutils.fix_print()
        # metadata = {'render.modes': ['human']}
        self.env = malmoenv.make()
        self.mission_type = mission_type
        mission_param = self.load_mission_param(self.mission_type)
     #   print(mission_param)
        self.actions =["movenorth","movesouth", "movewest", "moveeast","attack","use"] #,"strafe 1","strafe -1"] #,"attack 1","attack 0"]
        self.observation_space = np.zeros((2,9,9))
        self.state_map = mission_param["state_map"]
        self.entity_map = mission_param["entity_map"]
        self.relevant_entities =  mission_param["relevant_entities"]
        self.goal = mission_param["goal"]
        self.step_cost =  mission_param["step_cost"]
        self.goal_reward  =  mission_param["goal_reward"]
        self.max_steps =  mission_param["max_steps"]
        self.port = port
        self.episode = 0
        mission = self.get_mission_xml(self.mission_type)
        self.env.init(mission,server='127.0.0.1',port=self.port,exp_uid="test",role=0,episode=self.episode,action_filter=self.actions) #, args.port,
      
    def step(self,action):
        _ , _ , done, info = self.env.step(action)

        if self.mission_type == "hoe_farmland":
             reached_goal = self.checkBlockExists(json.loads(info),self.goal)
        else:
             reached_goal = self.checkInventoryForItem(json.loads(info),self.goal)

        if reached_goal:
             done = True
             reward = self.goal_reward
        else:
             reward = self.step_cost
        
        obs = self.obs_to_vector(info)

        if self.num_steps >= self.max_steps:
           done=True

        self.num_steps+=1
        # print(self.num_steps)

        return obs, reward, done, info

    def reset(self,remake_mission=False):
        self.num_steps = 0
        if remake_mission:
           self.env.init(mission,server='127.0.0.1',port=self.port,exp_uid="test",role=0,episode=self.episode,action_filter=self.actions) #, args.port,
        self.episode+=1
        return self.env.reset()

    def make_env_string(self,mission_type,draw_entities=[]):
        base = '<?xml version="1.0" encoding="UTF-8" standalone="no" ?><Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        base+='<About><Summary>Running {}...</Summary></About>'.format(mission_type)
        base+= '<ModSettings><MsPerTick>1</MsPerTick></ModSettings>' #1
        base+= '<ServerSection><ServerInitialConditions><Time><StartTime>6000</StartTime><AllowPassageOfTime>false</AllowPassageOfTime>'
        base+= '</Time><Weather>clear</Weather><AllowSpawning>false</AllowSpawning></ServerInitialConditions><ServerHandlers><FlatWorldGenerator />' 
        base+= '<DrawingDecorator>'

        for entity_info in draw_entities:
            base+=entity_info

        base+='</DrawingDecorator>'
        base+= '<ServerQuitFromTimeUp timeLimitMs="10000000"/><ServerQuitWhenAnyAgentFinishes/></ServerHandlers></ServerSection>'
        base+= '<AgentSection mode="Survival"><Name>agent</Name><AgentStart><Placement x="-1.5" y="204" z="-1.5" pitch="50" yaw="0"/>' #50
        base+= '<Inventory></Inventory>'

        base+='</AgentStart>'
        base+='<AgentHandlers>'
        base+='<ObservationFromGrid> <Grid name="floor9x9"> <min x="-4" y="0" z="-4"/> <max x="4" y="0" z="4"/> </Grid> </ObservationFromGrid>'

        base+='<ObservationFromNearbyEntities><Range name="entities" xrange="5" yrange="5" zrange="5"/></ObservationFromNearbyEntities>'
        base+='<ObservationFromFullInventory/><ObservationFromFullStats/><VideoProducer want_depth="false"><Width>640</Width><Height>480</Height></VideoProducer>'
        base+='<DiscreteMovementCommands><ModifierList type="deny-list"><command>attack</command><command>use</command></ModifierList></DiscreteMovementCommands>'
        base+='<ContinuousMovementCommands><ModifierList type="allow-list"><command>attack</command><command>use</command></ModifierList>'
        base+='</ContinuousMovementCommands><MissionQuitCommands quitDescription="done"/>'
        base+='<AbsoluteMovementCommands><ModifierList type="deny-list"></ModifierList></AbsoluteMovementCommands>'
        base+='</AgentHandlers></AgentSection></Mission>'
        return base


if __name__ == "__main__":
    print("starting server...")

    if len(sys.argv) > 1 and sys.argv[1] == "RUN_SERVER":
        print("Launching on port " + sys.argv[2])
        malmoenv.bootstrap.launch_minecraft(int(sys.argv[2]))
        exit()

    print("initializing environment...")
    env = MalmoEnvSpecial("pickaxe_stone",port=9000)
    obs = env.reset()
    print("reset")
    for step in range(100):
        print("\n",step)
        command = int(input())
        obs, reward, done, info = env.step(command)
        print(obs)
        print(reward)

    def make_env_string(self,mission_type,draw_entities=[]):
        base = '<?xml version="1.0" encoding="UTF-8" standalone="no" ?><Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        base+='<About><Summary>Running {}...</Summary></About>'.format(mission_type)
        base+= '<ModSettings><MsPerTick>1</MsPerTick></ModSettings>' #1
        base+= '<ServerSection><ServerInitialConditions><Time><StartTime>6000</StartTime><AllowPassageOfTime>false</AllowPassageOfTime>'
        base+= '</Time><Weather>clear</Weather><AllowSpawning>false</AllowSpawning></ServerInitialConditions><ServerHandlers><FlatWorldGenerator />' 
        base+= '<DrawingDecorator>'

        for entity_info in draw_entities:
            base+=entity_info

        base+='</DrawingDecorator>'
        base+= '<ServerQuitFromTimeUp timeLimitMs="10000000"/><ServerQuitWhenAnyAgentFinishes/></ServerHandlers></ServerSection>'
        base+= '<AgentSection mode="Survival"><Name>agent</Name><AgentStart><Placement x="-1.5" y="204" z="-1.5" pitch="50" yaw="0"/>' #50
        base+= '<Inventory></Inventory>'

        base+='</AgentStart>'
        base+='<AgentHandlers>'
        base+='<ObservationFromGrid> <Grid name="floor9x9"> <min x="-4" y="0" z="-4"/> <max x="4" y="0" z="4"/> </Grid> </ObservationFromGrid>'

        base+='<ObservationFromNearbyEntities><Range name="entities" xrange="5" yrange="5" zrange="5"/></ObservationFromNearbyEntities>'
        base+='<ObservationFromFullInventory/><ObservationFromFullStats/><VideoProducer want_depth="false"><Width>640</Width><Height>480</Height></VideoProducer>'
        base+='<DiscreteMovementCommands><ModifierList type="deny-list"><command>attack</command><command>use</command></ModifierList></DiscreteMovementCommands>'
        base+='<ContinuousMovementCommands><ModifierList type="allow-list"><command>attack</command><command>use</command></ModifierList>'
        base+='</ContinuousMovementCommands><MissionQuitCommands quitDescription="done"/>'
        base+='<AbsoluteMovementCommands><ModifierList type="deny-list"></ModifierList></AbsoluteMovementCommands>'
        base+='</AgentHandlers></AgentSection></Mission>'
        return base

if __name__ == "__main__":
    
    if sys.argv[1] == "RUN_SERVER":
        print("Launching server on "+sys.argv[1])
        malmoenv.bootstrap.launch_minecraft(int(sys.argv[1]))
        exit()

    print("initializing environment...")
    env = MalmoEnvSpecial("shears_sheep",port=9000)
    obs = env.reset()
    print("reset")
    for step in range(100):
        print("\n",step)
        command = int(input())
        obs, reward, done, info = env.step(command)
        print(obs)
        print(reward)
        print(info)
        # time.sleep(1)


 

