
from stable_baselines3.common.env_checker import check_env
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
    def __init__(self,observation_shape,elements):
        super(ImageEnv, self).__init__()
        print('game started:', len(elements))
        self.observation_shape = observation_shape
        self.elements = elements
        self.observation_space = gym.spaces.Box(low=0, high=255, shape= observation_shape, dtype=np.uint8)
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(observation_shape[1]),gym.spaces.Discrete(observation_shape[0])))
        self.current_step = 0
        self.reset()

    def step(self, action):
        done = False
        element = self.next_element
        overlapped = self.isoverlapped(action,element)
        outofscreen = self.isoutofscreen(action,element)
        if not overlapped:
            if not outofscreen:
                self.rewards += 1
            else:
                self.rewards += 0
        else:
            done = True
            print('overlapped, game over, see final score below:')
        observation = self.render(action,element,overlapped,outofscreen)

        if self.current_step == len(self.elements)-1:
            done = True
        
        else:
            self.next_element = self.elements[self.current_step+1]
            self.current_step += 1

        info = {'rewards':self.rewards,'done':done, 'next_element':self.next_element}
        return observation, self.rewards, done,info

    def render(self,action,element,overlapped,outofscreen):
        a4paper = Image.fromarray(self.a4paper)
        a4paper.paste(element,(action[0],action[1]))
        drawingbox = (action[0],action[1],action[0]+element.width,action[1]+element.height)
        draw = ImageDraw.Draw(a4paper)
        if not overlapped and not outofscreen:
            draw.rectangle(drawingbox, outline='blue')
        elif outofscreen:
            draw.rectangle(drawingbox, outline='red')
        elif overlapped:
            draw.rectangle(drawingbox, fill='red',width=10)
        self.a4paper = np.array(a4paper)
        return self.a4paper 

    def reset(self):
        image = Image.new('RGB', (self.observation_shape[1],self.observation_shape[0]), color='black')
        self.a4paper = np.array(image)
        self.rewards = 0
        self.previous_element = []
        self.previous_action = None
        self.current_step = 0
        self.next_element = self.elements[0]
        return self.a4paper

    def close(self):
        pass

    def isoverlapped(self,action,element):
        overlapped = False
        x0,y0,x1,y1 = action[0],action[1],action[0]+element.width,action[1]+element.height
        element_box = box(x0,y0,x1,y1)
        if self.previous_element:
            for previous_element in self.previous_element:
                if element_box.intersects(previous_element):
                    overlapped = True

        self.previous_element.append(element_box)
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

def load_element(element: PasteImage,factor=1):
    image = Image.open(element.imagepath)
    width, height = image.size
    image = image.resize((round(width/factor), round(height/factor)), Image.ANTIALIAS)
    return image

class RandomGuessReinforcementLearner():
    def __init__(self,jsonfile,observation_shape):
        elements = get_elements(jsonfile)
        elements = [load_element(element,factor=2352/observation_shape[0]) for element in elements]
        self.elements = shuffle(elements)
        self.env = ImageEnv(observation_shape,elements)
        self.agent_observation = None

        #self.agent_check() -> Passed

    def agent_check(self):
        check_env(self.env)

    def play(self):
        observation = self.env.reset()
        done = False
        while not done:
            action = self.env.action_space.sample()
            observation, reward, done, info = self.env.step(action)
            observations = {
                'reward': reward,
                'done': done,
                'next_element': info['next_element'],
                'observation': observation
            }
            self.agent_observation = observations

            if done:
                agentResult = Image.fromarray(observation)
                print('reward:',reward)
                break
        self.env.close()
        return Image.fromarray(observation)
    
    def agent_reset(self):
        self.env.reset()
        self.agent_observation = None

    
