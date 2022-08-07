from    stable_baselines3.common.vec_env import DummyVecEnv,VecEnv

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO,DDPG,A2C
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

class TextOnImageEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super(TextOnImageEnv, self).__init__()
        json_file = '/home/delta/vscode/dataextraction/croppedData/croppedData06_08_2022_04_02_28.json'
        elements = get_elements(json_file)
        elements = [load_element(element) for element in elements]
        self.elements = elements
        self.action_space = gym.spaces.Box(low = np.array([0,0]), high= np.array([1,1]), dtype=np.float32)
        max_width = []
        max_height = []
        for element in self.elements:
            max_width.append(element.width)
            max_height.append(element.height)
        max_width = max(max_width)
        max_height = max(max_height)

        low = torch.zeros(len(elements)+1,7).numpy()
        high = torch.tensor([1660,2340,1660,2340,0,0,0]).repeat(15,1).numpy()
        self.observation_space = gym.spaces.Box(low=low, high= high, dtype=np.int16)
        self.reset()

    def step(self, action):
        self.observation[self.current_step][0] = action[0] = round(action[0]*1660)
        self.observation[self.current_step][1] = action[1] = round(action[1]*2340)

        if len(self.previous_elements) > 0:
            for previous_element in self.previous_elements:
                if box(action[0],action[1],action[0]+self.elements[self.current_step-1].width,action[1]+self.elements[self.current_step-1].height).intersects(previous_element):
                    # Overlapped
                    self.overlapped = True
                if self.overlapped:
                    self.rewards = -10 * self.current_step
                    self.observation[self.current_step][4] = -10 * self.current_step
                    self.observation[self.current_step][6] = -1
                    self.done = True
            if action[0] + self.elements[self.current_step-1].width > 1660 or action[1] + self.elements[self.current_step-1].height > 2340:
                # Out of bounds
                self.outoutbound = False
                self.rewards = -3 * self.current_step
                self.observation[self.current_step][4] = -3 * self.current_step
                self.observation[self.current_step][5] = -1
                self.done = True
            else:
                if self.current_step >5:
                    self.rewards += 10*self.current_step
                    self.observation[self.current_step][4] = 10*self.current_step
                
                if self.current_step <5:
                    self.rewards += 1*self.current_step
                    self.observation[self.current_step][4] = 1*self.current_step
                
        else:
            self.rewards += 1*self.current_step
            self.observation[self.current_step][4] = 1*self.current_step
        

        if self.current_step == len(self.elements):
            self.done = True
            self.current_step = 1
            return self.observation, self.rewards, self.done, {'current_step': self.current_step, 'total rewards': self.rewards, 'reward':  self.observation[self.current_step][4],
            'action': (action[0], action[1])}
        self.previous_elements.append(box(action[0],action[1],action[0]+self.elements[self.current_step].width,action[1]+self.elements[self.current_step].height))
        
        self.current_step += 1
        return self.observation, self.rewards, self.done, {'current_step': self.current_step-1, 'total rewards': self.rewards, 'reward':  self.observation[self.current_step-1][4],
        'action': (action[0], action[1])}

    def close(self):
        pass

    def reset(self):
        self.current_step = 1
        self.rewards = -1
        self.done = False
        self.overlapped = False
        self.outoutbound = False
        self.elements = shuffle(self.elements)
        self.previous_elements = []
        observation = np.zeros((len(self.elements) +1, 7), dtype= np.int16)
        observation[0][2] = 1660
        observation[0][3] = 2340
        for i in range(1, len(self.elements)+1):
            observation[i][2] = self.elements[i-1].width
            observation[i][3] = self.elements[i-1].height
        self.observation = observation
        return self.observation

    def render(self,mode='human'):
        image = Image.new('RGB', (1660, 2340), color = (255, 255, 255))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for i in range(len(self.previous_elements)):
            draw.rectangle((self.previous_elements[i].bounds), outline = (0,0,0), fill=(0,0,0))



        return image




















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
    return image  


# if __name__ == '__main__':
#     json_file = '/home/delta/vscode/dataextraction/croppedData/croppedData06_08_2022_04_02_28.json'
#     elements = get_elements(json_file)
#     elements = [load_element(element) for element in elements]
#     env = ImageEnv(elements)
    
#     # done = True
#     # action = env.action_space.sample()

#     # while not done:
#     #     observation, reward, done, info = env.step(action)
#     #     print(info)
#     #     action = env.action_space.sample()

#     # env.render().show()
#     # print(observation)


#     train_model = True
#     if train_model:
#         callback = CheckpointCallback(save_freq=50000, save_path='./logs/',name_prefix='rl_model')
#         model = PPO.load('logs/rl_model_6600000_steps.zip',env=env, policy='MlpPolicy',tensorboard_log="./tensorboard_log/")
#         model.learn(total_timesteps=15000000, reset_num_timesteps=False, tb_log_name='PPO', callback=callback,)
#         model.save('rl_model')


#     use_model = False
#     if use_model:
#         done = False
#         observation = env.reset()
#         model = PPO.load('logs/rl_model_6250000_steps.zip', env=env)
#         i = 0
#         images = []
#         while not done:
#             action, _ = model.predict(observation)
#             observation, reward, done, info = env.step(action)
#             print(reward)
#             images.append(env.render())
#             i += 1
#             if done:
#                 break   
#         env.close()
#         images[1].save('model/model_.gif',
#                 save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)
        
    












            

 




    







