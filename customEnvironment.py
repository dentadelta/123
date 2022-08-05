
from typing import Dict
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from shapely.geometry import box
from imageprocessing import *
import gym
from PIL import Image, ImageDraw
import numpy as np
import pydantic
import random
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings("ignore")

class PasteImage(pydantic.BaseModel):
    x: int
    y: int
    imagepath: str

class ImageEnv(gym.Env):
    metadata = {'render.modes': ['human', 'console']}
    def __init__(self,elements):
        super(ImageEnv, self).__init__()
        print('Maximum rewards', len(elements))
        self.observation_shape = (468,332)
        self.elements = elements
        self.observation_space = gym.spaces.Dict({
            'state':gym.spaces.Box(low=0, high=255, shape=(len(elements),4), dtype=np.uint8),
            'next_element_height':gym.spaces.Discrete(468),
            'next_element_width':gym.spaces.Discrete(332),
        })
        self.action_space = gym.spaces.MultiDiscrete([332,468])
        self.current_step = 0
        self.reset()

    def step(self, action):
        done = False
        element = self.next_element
        overlapped = self.isoverlapped(action,element)
        outofscreen = self.isoutofscreen(action,element)
        self.render(action,element,overlapped,outofscreen)
        if not overlapped:
            if not outofscreen:
                self.rewards += 1
            else:
                self.rewards += 0
        else:
            done = True

        if self.current_step == len(self.elements)-1:
            done = True
            observation = {'state':self.previous_element,'next_element_height':0,'next_element_width':0}
            info = {'rewards':self.rewards,'done':done, 'next_width':0,'next_height':0}
            return observation, self.rewards, done, info
        
        else:
            self.next_element = self.elements[self.current_step+1]
            observation = {'state':self.previous_element,'next_element_height':self.next_element.height,'next_element_width':self.next_element.width}
            self.current_step += 1

        info = {'rewards':self.rewards,'done':done, 'next_width':self.next_element.width,'next_height':self.next_element.height}
        return observation, self.rewards, done,info

    def render(self,action,element,overlapped,outofscreen, mode='console'):
        if mode == 'human':
            self.a4paper.paste(element,(action[0],action[1]))
            drawingbox = (action[0],action[1],action[0]+element.width,action[1]+element.height)
            draw = ImageDraw.Draw(self.a4paper)
            if not overlapped and not outofscreen:
                draw.rectangle(drawingbox, outline='blue')
            elif outofscreen:
                draw.rectangle(drawingbox, outline='red')
            elif overlapped:
                draw.rectangle(drawingbox, fill='red',width=1)
            self.render_result = self.a4paper

        elif mode == 'console':
            rendered_result = Image.new('RGB', (332,468), color='white')
            draw = ImageDraw.Draw(rendered_result)
            if len(self.previous_element) > 1:
                for bx in self.previous_element[1:]:
                    draw.rectangle((bx[0],bx[1],bx[2],bx[3]), outline='blue')
            self.render_result = rendered_result

    def reset(self):
        image = Image.new('RGB', (self.observation_shape[1],self.observation_shape[0]), color='black')
        self.a4paper = image
        self.what_machine_see = None
        self.rewards = 0
        self.previous_element = np.array([[0,0,self.observation_shape[1],self.observation_shape[0]]])
        for i in range(len(self.elements)-1):
            self.previous_element = np.concatenate((self.previous_element,np.array([[0,0,0,0]])))
        self.previous_action = None
        self.current_step = 0
        self.next_element = self.elements[0]
        observation = {'state':self.previous_element,'next_element_height':self.next_element.height,'next_element_width':self.next_element.width}
        return observation

    def close(self):
        pass

    def isoverlapped(self,action,element):
        overlapped = False
        x0,y0,x1,y1 = action[0],action[1],action[0]+element.width,action[1]+element.height
        element_box = box(x0,y0,x1,y1)
        if len(self.previous_element) > 1:
            for previous_element in self.previous_element[1:]:
                previous_box = box(previous_element[0],previous_element[1],previous_element[2],previous_element[3])
                if element_box.intersects(previous_box):
                    overlapped = True
        self.previous_element[self.current_step] = np.array([x0,y0,x1,y1])
        return overlapped

    def isoutofscreen(self,action,element):
        outofscreen = False
        x0,y0,x1,y1 = action[0],action[1],action[0]+element.width,action[1]+element.height
        if x0 < 0 or y0 < 0 or x1 > self.observation_shape[1] or y1 > self.observation_shape[0]:
            outofscreen = True
        return outofscreen

def get_elements(jsonfile):
    with open(jsonfile,'r') as jsonfile:
        jsondata = json.load(jsonfile)
        jsonfile.close()

    elements = []
    for k,v in jsondata.items():
        save_folder = v['savefolder']
        box = v['box']
        imagepath = f'/home/delta/vscode/dataextraction/{save_folder}'
        elements.append(PasteImage(x=box[0],y=box[1],imagepath=imagepath))
    return elements

def load_element(element: PasteImage):
    image = Image.open(element.imagepath)
    image = image.resize((round(image.width/5), round(image.height/5)), Image.ANTIALIAS)
    return image

class RLAgent():
    def __init__(self,jsonfile):
        elements = get_elements(jsonfile)
        elements = [load_element(element) for element in elements]
        self.elements = shuffle(elements)
        self.env = ImageEnv(elements)
        self.agent_observation = None
        self.model = PPO('MultiInputPolicy',self.env,verbose=1)


        #self.agent_check() -> Passed

    def agent_check(self):
        check_env(self.env)

    def learn(self,n_episode=30000):
        self.model.learn(n_episode)

    def play(self):
        observation = self.env.reset()
        done = False
        while not done:
            action = self.env.action_space.sample()
            observation, reward, done, info = self.env.step(action)
            observations = {
                'reward': reward,
                'done': done,
                'observation': observation,
                'info': info
            }
            self.agent_observation = observations

            if done:
                print(info)
                break
        self.env.close()
    
    def agent_reset(self):
        self.env.reset()
        self.agent_observation = None

if __name__ == '__main__':
    agent = RLAgent('/home/delta/vscode/dataextraction/croppedData/croppedData06_08_2022_04_02_28.json')
    agent.learn()
    agent.env.reset()
    for i in range(20):
        agent.play()
        agent.agent_reset()
   
