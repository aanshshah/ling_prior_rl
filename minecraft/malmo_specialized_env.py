from __future__ import print_function

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
import math
from malmo import malmoutils
import numpy as np
import tensorflow as tf
import math


class MalmoEnvSpecial(gym.Env):
    def checkInventoryForItem(self,obs, requested):
        for i in range(0,9):#39): #Checks primary inventory
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

        state_data = [self.state_map[block] if block in self.state_map else 0 for block in state_data_raw]

        state_data =np.reshape(np.array(state_data,dtype=np.float64),(1,9,9))
        if use_entities: 
            entity_data = self.obs_to_ent_vector(world_state,self.relevant_entities)
            state_data = np.concatenate((state_data,entity_data),axis=0)
        return np.expand_dims(state_data,0)


    
    def fix_player_location(self,world_state):
        if len(world_state.observations) > 0:
            entity_data = json.loads(world_state.observations[-1].text)['entities']
            player_data = [x for x in entity_data if x["name"]=="agent"][0]
            player_loc = (player_data['x'],player_data['z'])

            if abs(math.floor(player_loc[0])-player_loc[0]) != 0.5:
                new_x = round(player_loc[0]-0.5)+0.5
                self.agent_host.sendCommand("tpx {}".format(new_x))
            if abs(math.floor(player_loc[1])-player_loc[1]) != 0.5:
                new_z = round(player_loc[1]-0.5)+0.5
                self.agent_host.sendCommand("tpz {}".format(new_z))





    def obs_to_ent_vector(self,world_state,relevant_entities):

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


    def load_mission_param(self,mission_type):
        mission_dict = {}
        if mission_type == "pickaxe_stone":
            mission_dict["state_map"] = {"air":0,"bedrock":1,"stone":2}
            mission_dict["entity_map"] = {"diamond_pickaxe":1,"cobblestone":2}
            mission_dict["relevant_entities"] = {"diamond_pickaxe":1,"cobblestone":2}
            mission_dict["goal"] = "cobblestone"
            mission_dict["step_cost"] = -0.1
            mission_dict["goal_reward"] = 100
            mission_dict["max_steps"] = 150

        elif mission_type == "axe_log":
            mission_dict["state_map"] = {"air":0,"bedrock":1,"log":2}
            mission_dict["entity_map"] = {"diamond_axe":1,"log":2}
            mission_dict["relevant_entities"] = set(mission_dict["entity_map"].keys())
            mission_dict["goal"] = "log"
            mission_dict["step_cost"] = -0.1
            mission_dict["goal_reward"] = 100
            mission_dict["max_steps"] = 150

        elif mission_type == "sword_pig":
            mission_dict["state_map"] = {"air":0,"bedrock":1}
            mission_dict["entity_map"] = {"diamond_sword":1,"pig":2}
            mission_dict["relevant_entities"] = set(mission_dict["entity_map"].keys())
            mission_dict["goal"] = "porkchop"
            mission_dict["step_cost"] = -0.1
            mission_dict["goal_reward"] = 100
            mission_dict["max_steps"] = 150

        elif mission_type == "sword_cow":
            mission_dict["state_map"] = {"air":0,"bedrock":1}
            mission_dict["entity_map"] = {"diamond_sword":1,"cow":2}
            mission_dict["relevant_entities"] = set(mission_dict["entity_map"].keys())
            mission_dict["goal"] = "beef"
            mission_dict["step_cost"] = -0.1
            mission_dict["goal_reward"] = 100
            mission_dict["max_steps"] = 250

        elif mission_type == "shears_sheep":
            mission_dict["state_map"] = {"air":0,"bedrock":1}
            mission_dict["entity_map"] = {"shears":1,"sheep":2}
            mission_dict["relevant_entities"] = set(mission_dict["entity_map"].keys())
            mission_dict["goal"] = "wool"
            mission_dict["step_cost"] = -0.1
            mission_dict["goal_reward"] = 100
            mission_dict["max_steps"] = 150

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

    def build_mission_env(self, mission_type):
        arena_xml = self.build_arena()

        if mission_type == "pickaxe_stone":  

            mission_xml = self.make_env_string(self.mission_type,[arena_xml])
            my_mission = MalmoPython.MissionSpec(mission_xml, True)
            my_mission.allowAllDiscreteMovementCommands() 
            my_mission.drawItem(2,206,2,"diamond_pickaxe")
            my_mission.drawBlock(random.randint(0,4)-2,204,random.randint(1,3)-2,"stone")

        elif  mission_type == "axe_log":   

            mission_xml = self.make_env_string(self.mission_type,arena_xml)
            my_mission = MalmoPython.MissionSpec(mission_xml, True)
            my_mission.allowAllDiscreteMovementCommands()  
            my_mission.drawItem(2,206,2,"diamond_axe")
            my_mission.drawBlock(random.randint(0,4)-2,204,random.randint(1,3)-2,"log")

        # elif  mission_type == "sword_pig":     

        #     pig_pos = (random.randint(1,3)-2,random.randint(1,3)-2)
        #     mission_xml = self.make_env_string(self.mission_type,[arena_xml,'<DrawEntity x="{}" y="{}" z="{}" type="Pig" />'.format(pig_pos[0],204,pig_pos[1]) ])
        #     my_mission = MalmoPython.MissionSpec(mission_xml, True)
        #     my_mission.allowAllDiscreteMovementCommands()
        #     my_mission.drawItem(2,206,2,"diamond_sword")

        # elif  mission_type == "sword_cow":     
        #     cow_pos = (random.randint(1,3)-2,random.randint(1,3)-2)
        #     mission_xml = self.make_env_string(self.mission_type,[arena_xml,'<DrawEntity x="{}" y="{}" z="{}" type="Cow" />'.format(cow_pos[0],204,cow_pos[1]) ])
        #     my_mission = MalmoPython.MissionSpec(mission_xml, True)
        #     my_mission.allowAllDiscreteMovementCommands()
        #     my_mission.drawItem(2,206,2,"diamond_sword")

        elif  mission_type == "shears_sheep":     
            sheep_pos = (random.randint(1,3)-2,random.randint(1,3)-2)
            mission_xml = self.make_env_string(self.mission_type,[arena_xml,'<DrawEntity x="{}" y="{}" z="{}" type="Sheep" />'.format(sheep_pos[0],204,sheep_pos[1]) ])
            my_mission = MalmoPython.MissionSpec(mission_xml, True)
            my_mission.allowAllDiscreteMovementCommands()
            my_mission.drawItem(2,206,2,"shears")
        #Need to fix shear op

        return my_mission

    def __init__(self,mission_type):
        malmoutils.fix_print()
        # metadata = {'render.modes': ['human']}

        self.mission_type = mission_type
        mission_param = self.load_mission_param(self.mission_type)
        print(mission_param)

        self.state_map = mission_param["state_map"]
        self.entity_map = mission_param["entity_map"]
        self.relevant_entities =  mission_param["relevant_entities"] = {"diamond_pickaxe":1,"cobblestone":2}
        self.goal = mission_param["goal"]
        self.step_cost =  mission_param["step_cost"]
        self.goal_reward  =  mission_param["goal_reward"]
        self.max_steps =  mission_param["max_steps"]


        self.agent_host = MalmoPython.AgentHost()
        self.agent_host.addOptionalFlag('debug', 'Turn on debugging.')

        malmoutils.parse_command_line(self.agent_host,["--recording_dir","malmo_recording"]) 
        self.actions =["movenorth 1","movesouth 1", "movewest 1", "moveeast 1","stay"] #,"strafe 1","strafe -1"] #,"attack 1","attack 0"]
        self.observation_space = np.zeros((2,9,9))
        self.action_space = gym.spaces.Discrete(len(self.actions))

        self.my_clients = MalmoPython.ClientPool()
        self.my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available

        self.max_retries = 3
        self.agentID = 0
        self.num_steps = 0
        # self.max_steps = 250
       
    def step(self, action):
        world_state = self.agent_host.getWorldState()
        # self.fix_player_location(world_state) #needed for mob worlds
        action_chosen = self.actions[action]
        if action_chosen != "stay": self.agent_host.sendCommand(action_chosen)
        
        if self.mission_type == "shears_sheep": 
            self.agent_host.sendCommand("attack 0")
            self.agent_host.sendCommand("use")


        # time.sleep(0.1)
        while world_state.is_mission_running:
            world_state = self.agent_host.getWorldState()
            if world_state.number_of_observations_since_last_state > 0: # and world_state.number_of_rewards_since_last_state > 0:
                break

        still_running = world_state.is_mission_running and self.num_steps < self.max_steps - 1

        if still_running and self.checkInventoryForItem(json.loads(world_state.observations[-1].text),self.goal):
            done = True
            reward = self.goal_reward
        else:
            done = not still_running
            reward = self.step_cost
        
        obs = self.obs_to_vector(world_state) if world_state.is_mission_running else self.observation_space
        info = {}

        self.num_steps+=1

        return obs, reward, done, info

    
    def reset(self):
            # mission_file = agent_host.getStringArgument('mission_file')
            self.num_steps = 0
            self.agent_host.sendCommand("quit")
            time.sleep(0.01)


         
            
            my_mission_record = malmoutils.get_default_recording_object(self.agent_host, "./save_%s-map%d-rep%d" % (0,0,0)) #(expID, imap, i))

            my_mission = self.build_mission_env(self.mission_type)
     
            for retry in range(self.max_retries):
                try:
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

    def make_env_string(self,mission_type,draw_entities=[]):
        base = '<?xml version="1.0" encoding="UTF-8" standalone="no" ?><Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        base+='<About><Summary>Running {}...</Summary></About>'.format(mission_type)
        base+= '<ModSettings><MsPerTick>1</MsPerTick></ModSettings>'
        base+= '<ServerSection><ServerInitialConditions><Time><StartTime>6000</StartTime><AllowPassageOfTime>false</AllowPassageOfTime>'
        base+= '</Time><Weather>clear</Weather><AllowSpawning>false</AllowSpawning></ServerInitialConditions><ServerHandlers><FlatWorldGenerator />' 
        base+= '<DrawingDecorator>'

        for entity_info in draw_entities:
            base+=entity_info


        base+='</DrawingDecorator>'
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
        base+='<AbsoluteMovementCommands><ModifierList type="deny-list"></ModifierList></AbsoluteMovementCommands>'
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


  



