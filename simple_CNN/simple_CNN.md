## Intro

We will build a simple convolutional neural network (CNN) to classify two Gaussian processes X vs Y.

```python
# numpy always
import numpy as np

# for generating wavy data
from math import pi

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# plotting
import matplotlib
import matplotlib.pyplot as plt

# when you generate the network it will automatically go onto the GPU
torch.set_default_tensor_type('torch.cuda.FloatTensor')
```


```python
# generate 2d white noise with given parameters
def wn2d(n = 100, mu = 0, sd = 1, l = 0.5, pts = 20):
    cf = np.zeros((pts**2, pts**2))
    np.fill_diagonal(cf, sd)
    
    if(isinstance(mu, int)):
        mf = np.tile(mu, cf.shape[0])
    else:
        mf = mu
    
    wn = np.random.multivariate_normal(mf, cf, n)
    return np.reshape(wn, (n, p, p))

# 100, 28x28 white noise field
n = 100
p = 28
ts = np.linspace(0, 1, p)

# generate mean function 
mu = np.outer(2*np.sin(2*pi*ts), 2*np.sin(2*pi*ts))
mu = np.reshape(mu, mu.size)

# Two samples. X has 0 mean. Y has wavy mean. We will try to predict which is which.
x = wn2d(n, mu = 0, pts = p)
y = wn2d(n, mu = mu, pts = p)
```


```python
fig, ax = plt.subplots(1, 2, figsize = (8, 4))

ax[0].imshow(x[0,...], origin = 'bottom', cmap = 'RdBu_r')
ax[0].set_title('X - example', fontsize = 24)
ax[1].imshow(y[0,...], origin = 'bottom', cmap = 'RdBu_r')
ax[1].set_title('Y - example', fontsize = 24)
plt.show()
```


![png](output_2_0.png)



```python
# convert x, y to "channel" format. shape = (obs, channels, width, height)
# Only 1 channel for these "images", i.e. matrix. RGB images have 3. Spectral images have a lot
# I made up the name channel format
x = np.reshape(x, (n, 1, p, p))
y = np.reshape(y, (n, 1, p, p))

# stack them on top of each other
df = np.concatenate([x, y], axis = 0)

# generate class labels
label = np.repeat([0, 1], 100)
label = np.reshape(label, (200, 1))

# convert to torch's format (from_numpy). Correct the type (float). Move to GPU (cuda)
df = torch.from_numpy(df).float().cuda()
label = torch.from_numpy(label).float().cuda()
```


```python
# in pytorch you define your network as a class 
# needs an __init__ function to define network parameters (instantiate the network)
# needs a forward function to tell pytorch how to move data through the network
class simpleCNN(nn.Module):
    def __init__(self):
        super(simpleCNN, self).__init__()
        self.model = nn.Sequential(
            # input channels = 1. output channels = 32. 3x3 kernel with 1 pixel padding
            # Use padding = (kernel_size-1)/2 so that image size doesnt change
            nn.Conv2d(1, 32, 3, padding = 1), nn.ReLU(),
            
            # maxpooling of size 2. Cuts image dimension in half (28x28 -> 14x14)
            nn.MaxPool2d(2),
            
            # input channels = 32. output channels = 16. 3x3 kernel with 1 pixel padding
            nn.Conv2d(32, 16, 3, padding = 1), nn.ReLU(),
            
            # maxpooling of size 2. Cuts image dimension in half (14x14 -> 7x7)
            nn.MaxPool2d(2),
            
            # flatten into "long" format (obs, features = channels x height x width)
            # in this case number of features = 16 x 7 x 7 
            nn.Flatten(),
            
            # output class scores
            nn.Linear(16*7*7, 1)
        )
    
    # forward says how to move data from the beginning to the end of the network
    def forward(self, x):
        return self.model(x)
```


```python
# instantiate network
scnn = simpleCNN()

# define optimizer for the network's parameters
optimizer = optim.Adam(scnn.parameters(), lr = 1e-3)

# define a loss function / minimization criteria for the optimizer
criterion = nn.MSELoss()
```


```python
epochs = 2000
for j in range(epochs):
    # reset loss value at the beginning of each training epoch
    loss = 0
    
    # reset gradients. Otherwise they accumulate!
    optimizer.zero_grad()
    
    # predict label from the data (df) using the simple NN (snn) (all observations at once)
    label_hat = scnn(df)
    
    # evaluate results
    loss = criterion(label, label_hat)
    
    # backprop the error
    loss.backward()
    
    # update the weights
    optimizer.step()
    
    # print how we're doing every so often
    if j % (epochs / 20) == 0:
        print("[iteration %04d] loss: %.6f" % (j, loss))
        
print("[iteration %04d] loss: %.6f" % (j+1, loss))
```

    [iteration 0000] loss: 0.894185
    [iteration 0100] loss: 0.001045
    [iteration 0200] loss: 0.000463
    [iteration 0300] loss: 0.000204
    [iteration 0400] loss: 0.000093
    [iteration 0500] loss: 0.000043
    [iteration 0600] loss: 0.000020
    [iteration 0700] loss: 0.000009
    [iteration 0800] loss: 0.000004
    [iteration 0900] loss: 0.000002
    [iteration 1000] loss: 0.000001
    [iteration 1100] loss: 0.000000
    [iteration 1200] loss: 0.000000
    [iteration 1300] loss: 0.000000
    [iteration 1400] loss: 0.000000
    [iteration 1500] loss: 0.000000
    [iteration 1600] loss: 0.000000
    [iteration 1700] loss: 0.000000
    [iteration 1800] loss: 0.000000
    [iteration 1900] loss: 0.000000
    [iteration 2000] loss: 0.000000



```python
# Run data through the trained NN. Move to cpu (cpu). Extract values (detach). Convert to numpy
# last 3 steps needed to plot values
label_hat = scnn(df).cpu().detach().numpy()

# bring labels back to cpu and into numpy
label = label.cpu().detach().numpy()
```


```python
# lo and behold it worked. First 100 are ~0 (X's label). Second 100 are ~1 (Y's label)
fig, ax = plt.subplots(1, 2, figsize = (12, 4))
ax[0].plot(label)
ax[0].set_title('True labels', fontsize = 24)
ax[1].plot(label_hat)
ax[1].set_title('Predicted labels', fontsize = 24)
plt.show()
```


![png](output_8_0.png)



```python

```
