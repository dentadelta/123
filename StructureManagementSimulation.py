# Pytorch is such an efficient matrix manipulation language

import gym
import torch
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.env_checker import check_env
import warnings
warnings.filterwarnings("ignore")

class StructureManagementEnv(gym.Env):
    def __init__(self,number_of_structures, number_of_components):
        super(StructureManagementEnv, self).__init__()
        self.number_of_structures = number_of_structures
        self.number_of_components = number_of_components
        low = torch.tensor([0]).repeat((number_of_structures,number_of_components,4)).numpy()
        high = torch.tensor([100]).repeat((number_of_structures,number_of_components,4)).numpy()
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.double)
        low = torch.tensor([0]).repeat((number_of_structures,number_of_components,2)).numpy() 
        high = torch.tensor([1]).repeat((number_of_structures,number_of_components,2)).numpy()
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.uint8)

    def step(self, action):
        this_year_condition_state_forcast = self.next_year_quantity_distribution()
        action = torch.tensor(action)
        action = F.pad(action,(1,0,0,0),value=0) 
        action = F.pad(action,(1,0,0,0),value=0)
        dot_product = torch.einsum('ijk,ijk->ijk',[action,this_year_condition_state_forcast])
        this_year_condition_state_forcast = torch.sub(this_year_condition_state_forcast,dot_product)
        sum_cs3_and_cs4 = torch.einsum('ijk->ij',[dot_product])
        sum_cs3_and_cs4 = sum_cs3_and_cs4[:, :, None]
        sum_cs3_and_cs4 = F.pad(sum_cs3_and_cs4,(1,0),value=0) 
        sum_cs3_and_cs4 = F.pad(sum_cs3_and_cs4,(0,1),value=0) 
        sum_cs3_and_cs4 = F.pad(sum_cs3_and_cs4,(0,1),value=0) 
        revised_condition_state_after_rehabs = torch.add(this_year_condition_state_forcast,sum_cs3_and_cs4)
        improvement_to_condition_state_quantity =torch.sub(self.condition_quantity,revised_condition_state_after_rehabs)
        change_to_condition_factor = improvement_to_condition_state_quantity/torch.tensor([100,100/2,100/4,100/8])
        sum_of_change_to_condition_factor = torch.sum(change_to_condition_factor,dim=2)
        rehab_unit_rate= torch.tensor([0,0,500,1000]).repeat(number_of_structures,number_of_components,1)
        rehab_cost = rehab_unit_rate*action
        benefit_rate = torch.tensor([100]).repeat(number_of_structures,number_of_components)
        benefit = benefit_rate*sum_of_change_to_condition_factor
        rehab_cost = torch.sum(rehab_cost,dim=2)
        self.expenditure = torch.sum(rehab_cost)
        benfit_cost_ratio = benefit/rehab_cost
        benfit_cost_ratio = torch.nan_to_num(benfit_cost_ratio, posinf=0.0, neginf=0.0)
        benefit_cost_ratio = torch.sum(benfit_cost_ratio).item() 
        print()
        self.BCR = round(benefit_cost_ratio,3)
        self.budget -= torch.sum(rehab_cost).item()
        self.rewards += self.BCR
        info = {
            'BCR': round(benefit_cost_ratio,3),
            'remaining_budget': '${:.2f}'.format(self.budget),
            'year alive': self.step_number,
            'accumulative BCR': round(self.rewards,3),
        }
        self.step_number += 1
        if self.budget <= 0:
            self.done = True
            return revised_condition_state_after_rehabs.numpy(), self.rewards, self.done, info
        self.budget += 1000*self.BCR*self.number_of_structures # add $100 every year  #for now....
        return revised_condition_state_after_rehabs.numpy(), self.rewards, self.done, info
    def reset(self):
        self.action = torch.tensor([0]).repeat(self.number_of_structures,self.number_of_components,2).numpy()
        self.done = False
        self.BCR = 0
        self.rewards = 0
        self.expenditure = 0
        self.budget = 10000
        self.condition_state_probabilty_moving_to_next_state()
        condition_quantity = torch.round(torch.softmax(torch.rand(number_of_structures,number_of_components,4),dim=2)*100) #sum quantity in each condition state of each component is 100
        self.condition_quantity = condition_quantity
        self.step_number = 0
        return condition_quantity.numpy()

    def render(self):
        print('original condition state:', self.condition_quantity.numpy())
        print('revised condition state after one year:', self.next_year_quantity_distribution().numpy())
        print('action:', self.action)
        print('bcr:', self.BCR)
        print('expenditure:', self.expenditure)
        print('accumulative BCR:', self.rewards)
        print('remaining budget:', self.budget)
        
    def close(self):
        pass
    def condition_state_probabilty_moving_to_next_state(self):
        cs1 = torch.softmax(torch.rand(self.number_of_structures,self.number_of_components,4),dim=-1)
        cs2 = torch.softmax(torch.rand(self.number_of_structures,self.number_of_components,3),dim=-1)
        cs2 = F.pad(cs2,(1,0,0,0),value=0)
        cs3 = torch.softmax(torch.rand(self.number_of_structures,self.number_of_components,2),dim=-1)
        cs3 = F.pad(cs3,(1,0,0,0),value=0)
        cs3 = F.pad(cs3,(1,0,0,0),value=0)
        cs4 = torch.tensor([0,0,0,1]).repeat(self.number_of_structures,self.number_of_components,1)
        conditions = torch.cat((cs4,cs3,cs2,cs1),dim=2)
        conditions = conditions.view(self.number_of_structures,self.number_of_components,-1,4)
        self.conditions = conditions
        return conditions

    def next_year_quantity_distribution(self):
        A = self.condition_quantity
        B = self.conditions
        return torch.flip(torch.einsum('ijk,ijkl->ijl',[A,B]),(2,))

if __name__ == '__main__':
    number_of_structures = 2
    number_of_components = 2
    env = StructureManagementEnv(number_of_structures, number_of_components)
    # check_env(env)
    env.reset()
    done = False
    year = 0
    while not done:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(info)
        year += 1
        if year == 20:
            done = True
            print('You reached your retirement age')

    # This game is designed to be solved by a reinforcement learning agent using DQN  (not for human). The goal is to mimize expenditure while maximise BCR

    #actions are: [[[0,0],[0,1],[1,1].....]]]  where [0,0] = not fixing a component belong to a structure, [0,1] = fix CS4 component only, [1,0] =fix CS3, [1,1] = fix both
    #fixing CS3 costs $50, fixing CS4 costs $100
    #each year you get back $1000 x BCR x number of structures

    #each structure contains multiple components
    #each components has 4 condition states
    #each condtion state has a probability of moving to the next condition state each year

    print(env.conditions)

    #CS4 has 100% chance remains at CS4
    #CS3 has xx% chance remains at CS3 and yy% chance moves to CS4 (xx+yy=1)
    #CS2 has xx% chance remains at CS2 and yy% chance moves to CS3 and zz chance to move to CS4 (xx+yy+zz=1)
    #CS2 has xx% chance remains at CS1 and yy% chance moves to CS2 and zz chance to move to CS3 and mm chance move to CS4 (xx+yy+zz+mm=1)
    #Each game the conditional proability is reset to a random density distribution


#    tensor([[[[0.0000, 0.0000, 0.0000, 1.0000],
#           [0.0000, 0.0000, 0.4952, 0.5048],
#           [0.0000, 0.3860, 0.2623, 0.3517],
#           [0.2506, 0.2292, 0.2524, 0.2677]],

#          [[0.0000, 0.0000, 0.0000, 1.0000],
#           [0.0000, 0.0000, 0.3707, 0.6293],
#           [0.0000, 0.3973, 0.2144, 0.3883],
#           [0.1927, 0.2564, 0.1954, 0.3555]]],


#         [[[0.0000, 0.0000, 0.0000, 1.0000],
#           [0.0000, 0.0000, 0.5193, 0.4807],
#           [0.0000, 0.2589, 0.2666, 0.4745],
#           [0.2604, 0.2810, 0.2179, 0.2407]],

#          [[0.0000, 0.0000, 0.0000, 1.0000],
#           [0.0000, 0.0000, 0.3554, 0.6446],
#           [0.0000, 0.3916, 0.2677, 0.3408],
#           [0.2244, 0.3456, 0.2617, 0.1683]]]])

# typical outputs each actions:

# {'BCR': 0.541, 'remaining_budget': '$6000.00', 'year alive': 0, 'accumulative BCR': 0.541}

# {'BCR': 0.599, 'remaining_budget': '$5082.00', 'year alive': 1, 'accumulative BCR': 1.14}

# {'BCR': 0.368, 'remaining_budget': '$3780.00', 'year alive': 2, 'accumulative BCR': 1.508}

# {'BCR': 0.475, 'remaining_budget': '$2516.00', 'year alive': 3, 'accumulative BCR': 1.983}

# {'BCR': 0.293, 'remaining_budget': '$966.00', 'year alive': 4, 'accumulative BCR': 2.276}

# {'BCR': 0.839, 'remaining_budget': '$-948.00', 'year alive': 5, 'accumulative BCR': 3.115}








