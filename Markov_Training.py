from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import datetime
import torch.nn as nn

# Fake Data
date_array = [2010,2011,2012,2013]
data = {'s' : [1,1,1,1], 'Date':date_array,'g': [1,1,2,2], 'c': [1,1,2,2], 'c_id': [1,1,2,2], 'x1': [1,2,3,4], 'x2': [1,2,3,4], 'x3': [1,2,3,4], 'x4': [1,2,3,4]}
df = pd.DataFrame(data)

class MarkoDataSet(Dataset):
    def __init__(self, df):
        for n,g in df.groupby('s'):
            g = g.sort_values('Date', ascending=True)
            Unique_Identity = g[['g','c','c_id']]
            Unique_Identity = Unique_Identity.drop_duplicates()
            Unique_date = g['Date'].unique().tolist()
            self.Data = []
            for i in range(len(Unique_date)-1):
                current_date = Unique_date[i]
                next_date = Unique_date[i+1]
                current_date_df = g[g['Date'] == current_date]
                next_date_df = g[g['Date'] == next_date]
                for row in current_date_df.itertuples():
                    for wor in next_date_df.itertuples():
                        if row.g == wor.g and row.c == wor.c and row.c_id == wor.c_id:
                            year_diff = wor.Date - row.Date
                            self.Data.append((row.s,row.g,row.c,row.c_id,row.x1,row.x2,row.x3,row.x4,wor.x1,wor.x2,wor.x3,wor.x4,current_date,next_date, year_diff))

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, idx):
        data = self.Data[idx]
        s = data[0]
        g = data[1]
        c = data[2]
        c_id = data[3]
        inputs = data[4:8]
        outputs = data[8:12]
        current_date = data[12]
        next_date = data[13]
        inputs = torch.tensor(inputs)
        outputs = torch.tensor(outputs)
        year_diff = data[14]
        year_diff = torch.tensor(year_diff)
        return {'s':s,'g':g,'c':c,'c_id':c_id,'inputs':inputs,'targets':outputs, 'current_date':current_date, 'next_date':next_date, 'year_diff':year_diff}

class MarkoModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MarkoModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size*2)
        self.h = None
        self.c = None
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(output_size*2, output_size*2)
        self.fc3 = nn.Linear(output_size*2, output_size)


    def forward(self, x, h=None, c=None):
        x = x.view(1, -1, 5)
        if self.h is None:
            x, (h, c) = self.lstm(x)
            self.h = h.detach()
            self.c = c.detach()

        else:
            x, (h, c) = self.lstm(x, (self.h, self.c))
            self.h = h.detach()
            self.c = c.detach()
        
        x = x.view(-1,5)
        x = self.fc(x)
        x = self.dropout(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        
        return x

def data_collate_fn(batch):
    inputs = torch.stack([item['inputs'] for item in batch])
    targets = torch.stack([item['targets'] for item in batch])
    year_dif = torch.stack([item['year_diff'] for item in batch])
    return {'inputs':inputs,'targets':targets, 'year_diff':year_dif}

dataset = MarkoDataSet(df)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MarkoModel(5, 5, 5).to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=data_collate_fn)

for epoch in range(100):
    for i, data in enumerate(dataloader):
        inputs = data['inputs']
        targets = data['targets']
        year_dif = data['year_diff']
        inputs = torch.cat((inputs, year_dif.view(-1,1)), dim=1).float().to(device)
        targets = torch.cat((targets, year_dif.view(-1,1)), dim=1).float().to(device)
        output= model(inputs)
        loss = loss_function(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())

#Work in progress

# Model do learn:
# 8.891339302062988
# 8.671114921569824
# 7.886721134185791
# 7.421387672424316
# 6.730751037597656
# 6.132023334503174
# 6.0146026611328125
# 5.267750263214111
# 3.9525184631347656
# 3.0086424350738525
# 2.9119341373443604
# 1.424885869026184
# 2.486180067062378
# 1.1062493324279785
# 1.1281746625900269
# 1.9335850477218628
# 1.4420360326766968
# 0.903897225856781
# 0.6531164050102234
# 3.5372846126556396
# 1.7764055728912354
# 0.7843702435493469
# 0.7528569102287292
# 0.4303874969482422
# 0.3347478210926056
# 1.1005457639694214
# 0.5916035175323486
# 0.4191422462463379
# 0.26157012581825256
# 0.3318746089935303
# 0.32503947615623474
# 0.4281010627746582
# 0.5194684863090515
# 2.934647798538208
# 1.113714337348938
# 0.3840904235839844
# 0.36974525451660156
# 1.1217460632324219
# 0.44172558188438416
# 0.11438192427158356
# 0.18463067710399628
# 0.1910877823829651
# 0.3542122542858124
# 0.3028877377510071
# 0.19866928458213806
# 0.08264602720737457
# 0.11826739460229874
# 0.35811343789100647
# 0.5196130871772766
# 0.13315609097480774
# 0.08648752421140671
# 0.071011483669281
# 0.23445537686347961
# 0.07195112109184265
# 1.3907699584960938
# 0.062350619584321976
# 0.5543889403343201
# 0.9169471859931946
# 0.6240936517715454
# 0.6899058222770691
# 0.4952671229839325
# 0.08208713680505753
# 0.598038375377655
# 0.25926902890205383
# 0.055174823850393295
# 0.27712151408195496
# 0.06899165362119675
# 0.26878416538238525
# 0.05461367592215538
# 0.15098242461681366
# 0.5943378210067749
# 1.0991624593734741
# 0.18088746070861816
# 1.037734866142273
# 0.0626470074057579
# 0.21082143485546112
# 0.18063382804393768
# 0.20160512626171112
# 0.690079391002655
# 0.08360322564840317
# 0.501168429851532
# 0.6240193843841553
# 0.21316294372081757
# 0.20646536350250244
# 0.032597653567790985
# 0.49799638986587524
# 0.46334108710289
# 0.36437496542930603
# 0.26957157254219055
# 0.04234166070818901
# 1.047567367553711
# 0.6046843528747559
# 0.09306488186120987
# 0.09414814412593842
# 0.5804281234741211
# 0.06615949422121048
# 0.4256959855556488
# 0.08258756250143051
# 0.03984609991312027
# 0.18097341060638428

    

 



