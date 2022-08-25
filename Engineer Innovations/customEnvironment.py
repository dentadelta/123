
from typing import OrderedDict
from stable_baselines3.common.vec_env import DummyVecEnv,VecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO
from shapely.geometry import box
from imageprocessing import *
import gym
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import pydantic
import torch
from sklearn.utils import shuffle

import warnings
warnings.filterwarnings("ignore")

class ImageEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,elements):
        super(ImageEnv, self).__init__()
        self.elements = elements
        self.action_space = gym.spaces.MultiDiscrete([332,468])

        max_width = []
        max_height = []
        for element in self.elements:
            max_width.append(element.width)
            max_height.append(element.height)
        max_width = max(max_width)
        max_height = max(max_height)

        self.observation_space = gym.spaces.Dict({
            'main_screen': gym.spaces.Box(low=0, high= 255, shape=(468,332), dtype=np.uint8),
            'extra_screen': gym.spaces.Box(low=0, high= 255, shape=(max_height,max_width), dtype=np.uint8)
        })
        self.game_engine = Game_Engine(self.elements)
        self.reset()

    def step(self, action):        
        self.current_step = self.game_engine.current_frame
        observables = self.game_engine.observable_state(action)
        done = observables['done']
        reward = observables['reward']
        self.observation = {
            'main_screen': np.array(ImageOps.grayscale(observables['main_screen'])),
            'extra_screen': np.array(ImageOps.grayscale(observables['extra_screen']))
        }
        return self.observation, reward, done, {}

    def close(self):
        pass

    def reset(self):
        self.game_engine = Game_Engine(self.elements)
        self.current_step = self.game_engine.current_frame
        self.observation = OrderedDict({
            'main_screen': np.array(ImageOps.grayscale(self.game_engine.main_screen)),
            'extra_screen': np.array(ImageOps.grayscale(self.game_engine.extra_screen))
        })
        return self.observation
    def render(self,action,mode='human'):
        pass
            
class PasteImage(pydantic.BaseModel):
    x: int
    y: int
    imagepath: str

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

class Game_Engine():
    def __init__(self,elements):
        self.elements = shuffle(elements)
        max_width = []
        max_height = []
        for element in self.elements:
            max_width.append(element.width)
            max_height.append(element.height)
        max_width = max(max_width)
        max_height = max(max_height)
        
        self.current_frame = 0
        self.extra_screen = Image.new('RGB', (max_width,max_height), color='white')
        self.main_screen = Image.new('RGB', (332,468), color='white')
        self.previous_actions = []
        self.this_action = {}
        self.score = 0
        self.overlapped = False
        self.out_of_bound = False
        self.done = False

    def observable_state(self, external_action):
        self.obtain_main_screen_data(external_action)
        self.obtain_extra_screen_data()
        if self.determine_if_game_is_over() or self.score < 0 or self.overlapped:
            self.done = True
        self.step()

        return {
            'main_screen': ImageOps.grayscale(self.main_screen),
            'extra_screen': self.extra_screen,
            'reward': self.score,
            'done': self.done,
        }

    def obtain_current_action(self, external_action):
        current_element = self.elements[self.current_frame]
        action_x = external_action[0]
        action_y = external_action[1]

        self.this_action = {
            'choose_location_x_to_paste_image': action_x,
            'choose_location_y_to_paste_image': action_y,
        }
        self.this_action['current_element_width'] = current_element.width
        self.this_action['current_element_height'] = current_element.height


    def obtain_main_screen_data(self,external_action):
        self.obtain_current_action(external_action)
        overlapped = self.check_overlapped()
        out_of_bound = self.check_out_of_bound()
        fillcolor = (220,220,220)
        if overlapped:
            fillcolor = (255,0,0)
        if out_of_bound:
            fillcolor = (0,0,255)
        current_element = self.elements[self.current_frame]
        draw = ImageDraw.Draw(current_element)
        draw.rectangle([(0,0),current_element.size], fill = fillcolor)
        self.main_screen.paste(current_element,(self.this_action['choose_location_x_to_paste_image'],self.this_action['choose_location_y_to_paste_image']))
        self.overlapped = overlapped
        self.out_of_bound = out_of_bound
        

    def obtain_extra_screen_data(self):
        max_width = []
        max_height = []
        for element in self.elements:
            max_width.append(element.width)
            max_height.append(element.height)
        max_width = max(max_width)
        max_height = max(max_height)
        extra_screen = Image.new('RGB', (max_width,max_height), color='white')
        if self.current_frame != len(self.elements)-1:
            next_element = self.elements[self.current_frame+1]
            extra_screen.paste(next_element,(0,0))
            self.extra_screen = extra_screen
            awarded_score = self.determine_the_score(self.overlapped,self.out_of_bound)
            self.score = self.score + awarded_score

    def step(self):
        self.current_frame += 1
        self.previous_actions.append(self.this_action)


    def check_overlapped(self):
        current_action_x = self.this_action['choose_location_x_to_paste_image']
        current_action_y = self.this_action['choose_location_y_to_paste_image']
        current_action_width = self.this_action['current_element_width']
        current_action_height = self.this_action['current_element_height']
        current_action_box = box(current_action_x,current_action_y,current_action_x+current_action_width,current_action_y+current_action_height)

        if len(self.previous_actions) > 0:
            for i in range(len(self.previous_actions)):
                previous_action = self.previous_actions[i]
                previous_action_x = previous_action['choose_location_x_to_paste_image']
                previous_action_y = previous_action['choose_location_y_to_paste_image']
                previous_action_width = previous_action['current_element_width']
                previous_action_height = previous_action['current_element_height']
                previous_action_box = box(previous_action_x,previous_action_y,previous_action_x+previous_action_width,previous_action_y+previous_action_height)
                if current_action_box.intersects(previous_action_box):
                    return True
            return False

    def check_out_of_bound(self):
        current_action_x = self.this_action['choose_location_x_to_paste_image']
        current_action_y = self.this_action['choose_location_y_to_paste_image']
        current_action_width = self.this_action['current_element_width']
        current_action_height = self.this_action['current_element_height']
        if current_action_x < 0 or current_action_x > 332 or current_action_y < 0 or current_action_y > 468:
            return True
        return False

    
    def determine_the_score(self,overlapped,out_of_bound):
        awarded_score = 0
        if overlapped:
            awarded_score = -2
        if out_of_bound:
            awarded_score = -1
        if not overlapped and not out_of_bound:
            awarded_score = 1
        return awarded_score
    
    def determine_if_game_is_over(self):
        return self.current_frame == len(self.elements) - 1



if __name__ == '__main__':
    json_file = '/home/delta/vscode/dataextraction/croppedData/croppedData06_08_2022_04_02_28.json'
    elements = get_elements(json_file)
    elements = [load_element(element) for element in elements]
    env = ImageEnv(elements)
    callback = CheckpointCallback(save_freq=50000, save_path='./logs/',name_prefix='rl_model')
    model = PPO(env=env, policy='MultiInputPolicy',tensorboard_log="./tensorboard_log/")
    model.learn(total_timesteps=10000000, reset_num_timesteps=False, tb_log_name='PPO', callback=callback)
    model.save('rl_model')


    use_model = False
    if use_model:
        done = False
        observation = env.reset()
        model = PPO.load('rl_model', env=env)
        while not done:
            action, _ = model.predict(observation)
            observation, reward, done, info = env.step(action)
            if done:
                Image.fromarray(observation['main_screen']).show()
                break
        env.close()
        
    












            

 




    







