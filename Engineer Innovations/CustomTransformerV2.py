
### Still working on this. GO to figureout away to initialise to(device) for non-paralel layer such as linear layer....


import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from typing import Optional,Tuple
warnings.filterwarnings('ignore')

dmodel = 512
dk = 64
dv = 64
h = 8
dff = 2048

GPU_List = [0,0,0,0,0,0,0,0]    #specify how you would like to split your model over multiple GPUs


class SingleAttentionHead(nn.Module):
    def __init__(self,dmodel,dk,dv,max_length,cuda_number='cuda:0', applyMask = False):
        super(SingleAttentionHead,self).__init__()
        self.proj_key = nn.Linear(dmodel,dk).to(cuda_number)
        self.proj_query = nn.Linear(dmodel,dk).to(cuda_number)
        self.proj_value  = nn.Linear(dmodel,dv).to(cuda_number)
        self.dk = dk
        self.cuda_number = cuda_number
        self.max_length = max_length
        self.applyMask = applyMask
        
    def forward(self,query,key,value):
        query = query.to(self.cuda_number)
        key = key.to(self.cuda_number)
        value = value.to(self.cuda_number)
        
        I = torch.einsum('b i d , b j d -> b i j', q, k)
        
        if self.applyMask and y == None: #If you need a mask then this is the decoder-self attended layer
            mask = create_mask(self.max_length,self.cuda_number)
            for i in range(len(I)):
                I[i].masked_fill_(mask==0,float('-inf'))
        
        attention = F.softmax(I/(self.dk**0.5), dim=-1)
  
        head = torch.einsum('b i j , b j d -> b i d', attention, v)
        
        if self.cuda_number != 'cuda:0':
            return head.to('cuda:0')
        return head
 

class MultiheadAttention(nn.Module):
    def __init__(self,dmodel,dk,dv,max_length,applyMask = False,GPU_List = [0,0,0,0,0,0,0,0]):
        super(MultiheadAttention, self).__init__()

        self.heads = nn.ModuleList([
            SingleAttentionHead(dmodel,dk,dv,max_length,f'cuda:{GPU_List[i]}',applyMask) for i in range(int(dmodel/dk))
        ])

        self.W0 = nn.Linear(dmodel,dmodel).to('cuda:0') 


    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None):

      multi_attention_heads = 'Empty'

      for i, l in enumerate(self.heads):
        if i == 0:
          multi_attention_heads = l(query,key,value)

        else:
          multi_attention_heads = torch.cat((multi_attention_heads,l(query,key,value)), dim=2)

      multi_attention_heads = self.W0(multi_attention_heads) 

      return multi_attention_heads, 'None'
    
class CustomEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self,d_model, nhead, dim_feedforward, dropout, activation):
    super(CustomEncoderLayer,self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
    self.self_attn = MultiheadAttention(dmodel,dk,dv,dff,applyMask = False,GPU_List)
    
    
class CustomDecoderLayer(nn.TransformerDecoderLayer):
  def __init__(self,d_model, nhead, dim_feedforward, dropout, activation):
    super(CustomDecoderLayer,self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
    self.self_attn = MultiheadAttention(dmodel,dk,dv,dff,applyMask = True,GPU_List)
    
customEncoderLayer = CustomEncoderLayer(dmodel,h,dff,0.1,'relu')
customDecoderLayer = CustomDecoderLayer(dmodel,h,dff,0.1,'relu')

customTransformerEncoder = nn.TransformerEncoder(customEncoderLayer,6)
customTransformerDecoder = nn.TransformerDecoder(customDecoderLayer,6)

customTransformer = nn.Transformer(custom_encoder=customTransformerEncoder,custom_decoder=customTransformerDecoder)

print(customTransformer.parameters)
