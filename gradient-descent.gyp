import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

inputs = np.array([
    [800, 1200, 700], 
    [900, 1500, 800],
    [450, 800, 300],
    [650, 1050, 500]], dtype="float32")

targets = np.array([
    [3000], 
    [3500],
    [1500], 
    [2000]], dtype="float32")

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

#we create a dataset with the inputs and the targets
train_ds = TensorDataset(inputs, targets)

#we randomly initliasie a set of weights and biases
model = nn.Linear(3,1)
#print(model.weight)
#print(model.bias)

#we try to get some predictions with the weights we initialised randomly
preds = model(inputs)
#print(preds)

#when nn.Linear it generates a set of weights and biases 
weight = model.weight
bias = model.bias

#we compute the loss function
loss_fn = F.mse_loss
loss = loss_fn(preds, targets)
print("Std. loss", loss)

#when we call loss.backward PyTorch computes the gradients of the loss with respect to the weights
loss.backward()

#after we compute the weights and biases we subtract to them
#their partial derivatives, times a small amount that we call the llearning rate
with torch.no_grad():
    weight -= weight.grad * 1e-7
    bias -= bias.grad * 1e-7
    weight.grad.zero_()
    bias.grad.zero_()

preds = model(inputs)
loss = loss_fn(preds, targets)
print("Adjusted loss:", loss)

for i in range(10000):
    preds = model(inputs)
    loss = loss_fn(preds, targets)
    loss.backward()
    with torch.no_grad():
        weight -= weight.grad * 1e-7
        bias -= bias.grad * 1e-7
        weight.grad.zero_()
        bias.grad.zero_()

preds = model(inputs)
loss = loss_fn(preds, targets)
print("Final loss:", loss)




