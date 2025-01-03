#!/usr/bin/env python
# coding: utf-8

# In[82]:

on_gpu = False

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# In[83]:

dev ='cuda' if on_gpu else 'cpu'

df = pd.read_csv('../dataset/clean.csv')
df.head()


# In[84]:


X = df.drop('gdp', axis=1).values
y = df['gdp'].values


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[85]:


class factbook_data:
    def __init__(self, X, y, scale_data=True):
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            if scale_data:
                X = StandardScaler().fit_transform(X)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# In[86]:


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 2**21),
            nn.ReLU(),
            nn.Linear(2**21, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)


# In[87]:


if __name__=='__main__':
    torch.manual_seed(42)
    X,y = X, y


# In[88]:


dataset = factbook_data(X, y, scale_data=False)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)
testloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)


# In[89]:


mlp = MLP()
mlp.to(dev)

loss_function = nn.L1Loss()
optimizer = torch.optim.Adagrad(mlp.parameters(), lr=1e-4)


# In[ ]:


from datetime import datetime
start = datetime.now()
for epoch in range(0,5):
    print(f'Starting Epoch {epoch+1}')

    current_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        targets = targets.reshape((targets.shape[0], 1))

        optimizer.zero_grad()

        outputs = mlp(inputs.to(dev))

        loss = loss_function(outputs, targets.to(dev))

        loss.backward()

        optimizer.step()

        current_loss += loss.item()

        if i%10 == 0:
            print(f'Loss after mini-batch %5d: %.3f'%(i+1, current_loss/500))
            current_loss = 0.0

    print(f'Epoch {epoch+1} finished')
end = datetime.now()

print(f"Training has completed and took {(end -start).total_seconds()}s on device {dev}")


# In[ ]:


test_data = torch.from_numpy(X_test).float()
test_targets = torch.from_numpy(y_test).float()


# In[ ]:

start = datetime.now()
mlp.eval() 
stop = datetime.now()
print(f"eval took {(stop - start).total_seconds()}")


# In[ ]:


with torch.no_grad():
    outputs = mlp(test_data.to(dev))
    predicted_labels = outputs.squeeze().tolist()

predicted_labels = np.array(predicted_labels)
test_targets = np.array(test_targets)

mse = mean_squared_error(test_targets, predicted_labels)
r2 = r2_score(test_targets, predicted_labels)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

