import gym
import torch
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.env_checker import check_env
from gym.spaces import MultiDiscrete, Box
import warnings
warnings.filterwarnings("ignore")

class StructureManagementEnv(gym.Env):
    def __init__(self,number_of_structures, number_of_components):
        super(StructureManagementEnv, self).__init__()
        self.number_of_structures = number_of_structures
        self.number_of_components = number_of_components
        low = torch.tensor([0]).repeat((number_of_structures*number_of_components*4)).numpy()
        high = torch.tensor([100]).repeat((number_of_structures*number_of_components*4)).numpy()
        self.observation_space = Box(low=low, high=high, dtype=np.double)
        low = torch.tensor([0]).repeat((number_of_structures*number_of_components*2)).numpy()
        high = torch.tensor([1]).repeat((number_of_structures*number_of_components*2)).numpy()
        self.action_space = MultiDiscrete([1 for _ in range(number_of_structures*number_of_components*2)])

    def step(self, action):
        this_year_condition_state_forcast = self.next_year_quantity_distribution()
        action = torch.tensor(action).view(self.number_of_structures,self.number_of_components,2)
        action = F.pad(action,(1,0,0,0),value=0) 
        action = F.pad(action,(1,0,0,0),value=0)
        print(action)
        dot_product = torch.einsum('ijk,ijk->ijk',[action,this_year_condition_state_forcast])
        this_year_condition_state_forcast = torch.sub(this_year_condition_state_forcast,dot_product)
        sum_cs3_and_cs4 = torch.einsum('ijk->ij',[dot_product])
        sum_cs3_and_cs4 = sum_cs3_and_cs4[:, :, None]
        sum_cs3_and_cs4 = F.pad(sum_cs3_and_cs4,(1,0),value=0) 
        sum_cs3_and_cs4 = F.pad(sum_cs3_and_cs4,(0,1),value=0) 
        sum_cs3_and_cs4 = F.pad(sum_cs3_and_cs4,(0,1),value=0) 
        revised_condition_state_after_rehabs = torch.add(this_year_condition_state_forcast,sum_cs3_and_cs4)
        improvement_to_condition_state_quantity =torch.sub(self.condition_quantity,revised_condition_state_after_rehabs)
        self.condition_quantity = revised_condition_state_after_rehabs
        change_to_condition_factor = improvement_to_condition_state_quantity/torch.tensor([100/8,100/4,100/2,100])
        sum_of_change_to_condition_factor = torch.sum(change_to_condition_factor,dim=2)
        rehab_unit_rate= torch.tensor([0,0,500,1000]).repeat(self.number_of_structures,self.number_of_components,1)
        rehab_cost = rehab_unit_rate*action
        benefit_rate = torch.tensor([100]).repeat(self.number_of_structures,self.number_of_components)
        benefit = benefit_rate*sum_of_change_to_condition_factor
        rehab_cost = torch.sum(rehab_cost,dim=2)
        self.expenditure = torch.sum(rehab_cost)
        benfit_cost_ratio = benefit/rehab_cost
        benfit_cost_ratio = torch.nan_to_num(benfit_cost_ratio, posinf=0.0, neginf=0.0)
        benefit_cost_ratio = torch.sum(benfit_cost_ratio).item() 
        self.BCR = round(benefit_cost_ratio,3)
        self.budget -= torch.sum(rehab_cost).item()
        self.budget += 1000*torch.sum(sum_of_change_to_condition_factor).item() # add $1000 every year  #for now....
        self.rewards += self.BCR
        info = {
            'BCR': round(benefit_cost_ratio,3),
            'remaining_budget': '${:.2f}'.format(self.budget),
            'year alive': self.step_number,
            'accumulative BCR': round(self.rewards,3),
            'condition_factor_changed': torch.sum(sum_of_change_to_condition_factor).item()
        }
        self.step_number += 1
        if self.budget <= 0:
            self.done = True
            return self.condition_quantity.view(-1).numpy(), self.rewards, self.done, info
        return self.condition_quantity.view(-1).numpy(), self.rewards, self.done, info
    def reset(self):
        self.ation = torch.tensor([0]).repeat(self.number_of_structures,self.number_of_components,2).numpy()
        self.done = False
        self.BCR = 0
        self.rewards = 0
        self.expenditure = 0
        self.budget = 100000
        self.condition_state_probabilty_moving_to_next_state()
        condition_quantity = torch.round(torch.softmax(torch.rand(self.number_of_structures,self.number_of_components,4),dim=2)*100) #sum quantity in each condition state of each component is 100
        self.condition_quantity = condition_quantity
        self.step_number = 0
        return condition_quantity.view(-1).numpy()

    def render(self):
        print('original condition state:', self.condition_quantity.numpy())
        print('revised condition state after one year:', self.next_year_quantity_distribution().numpy())
        print('action:', self.ation)
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
        self.condition_quantity = torch.flip(torch.einsum('ijk,ijkl->ijl',[A,B]),(2,))
        return self.condition_quantity


class ConditionStateGenerator(torch.nn.Module):
    def __init__(self,number_of_structures,number_of_components):
        super().__init__()
        self.number_of_structures = number_of_structures
        self.number_of_components = number_of_components
        cs1 = torch.softmax(torch.rand(self.number_of_structures,self.number_of_components,4),dim=-1)
        cs2 = torch.softmax(torch.rand(self.number_of_structures,self.number_of_components,3),dim=-1)
        cs2 = F.pad(cs2,(1,0,0,0),value=0)
        cs3 = torch.softmax(torch.rand(self.number_of_structures,self.number_of_components,2),dim=-1)
        cs3 = F.pad(cs3,(1,0,0,0),value=0)
        cs3 = F.pad(cs3,(1,0,0,0),value=0)
        cs4 = torch.tensor([0,0,0,1]).repeat(self.number_of_structures,self.number_of_components,1)
        conditions = torch.cat((cs4,cs3,cs2,cs1),dim=2)
        conditions = conditions.view(self.number_of_structures,self.number_of_components,-1,4)

        self.layer_1 = torch.nn.Linear(4,4, bias=False)
        self.layer_1.weight = torch.nn.Parameter(conditions.long(), requires_grad=False)

    def forward(self,x):
        x = x.long()
        x = self.layer_1(x)
        x = torch.flip(x,(2,))
        x = x.float()
        return x/1000

class ConditionStateEstimator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = torch.nn.Linear(4,4)
        self.relu = torch.nn.ReLU()
    
    def forward(self,x):
        x = self.layer_1(x)
        x = self.relu(x)
        return x

if __name__ == '__main__':
    generator = ConditionStateGenerator(2,2)
    learner = ConditionStateEstimator()
    loss_function = torch.nn.L1Loss()
    optimizer = torch.optim.Adadelta(learner.parameters(),lr=0.0001)

    component_1 = torch.tensor([20,10,0,5])
    component_2 = torch.tensor([20,10,10,5])
    structure_1 = torch.stack((component_1,component_2))
    component_3 = torch.tensor([10,10,0,5])
    component_4 = torch.tensor([10,14,0,5])
    structure_2 = torch.stack((component_3,component_4))

    structures_year_0_condition = torch.stack((structure_1,structure_2)).float()

    learner.train()
    # Generate Data:
    total_loss = 0
    for i in range(10000): #10 years L2 inspection reports
        target = generator(structures_year_0_condition)
        output = learner(structures_year_0_condition)
        loss = loss_function(output,target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % 1000 == 0:
            print(total_loss/1000)
            total_loss = 0
            learner.eval()
            print(generator(target))
            print(learner(target))
            learner.train()
        optimizer.zero_grad()

        structures_year_0_condition = target

    
        

    
    

                                        




