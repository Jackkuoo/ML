import os
import sys
import math
import pandas as pd
import numpy as np

# Load the training data
X = pd.read_csv("data_X.csv", encoding='big5').values #shape: (500, 8)
X = X[:, 1:] #shape: (500, 7)
X = X.astype(float)
X = np.array(X)
print(X.shape)

Y = pd.read_csv("data_T.csv", encoding='big5').values
Y = Y[:, 1:]
Y = np.array(Y).reshape(500,)
print(Y.shape)

# scaling
maxnum = np.max(X, axis=0)
minnum = np.min(X, axis=0)
x_data = (X - minnum) / (maxnum - minnum + 1e-20)
print(x_data)

# Gradient Descent M=1
'''
# ydata = b + w*xdata
b = 0.0
w = np.ones(7)
lr = 1
epoch = 20000
b_lr = 0.0
w_lr = np.zeros(7)

for e in range(epoch):
  # Calculate the value of the loss function
  error = Y - b - np.dot(x_data, w) #shape: (500,)
  loss = np.mean(np.square(error)) # Mean Square Error

  # Calculate gradient
  b_grad = -2*np.sum(error)*1 #shape: ()
  w_grad = -2*np.dot(error, x_data) #shape: (7,)
  
  # update learning rate
  b_lr = b_lr + b_grad**2
  w_lr = w_lr + w_grad**2

  # update parameters.
  b = b - lr/np.sqrt(b_lr) * b_grad
  w = w - lr/np.sqrt(w_lr) * w_grad

  # Print "Root Mean Square Error" per 2000 epoch
  if (e+1) % 2000 == 0:
    print('epoch:{}\n Loss:{}'.format(e+1, np.sqrt(loss)))
print("w: ",w)
print("b: ",b)
'''

# Gradient Descent M=2
# ydata = b + w2*xdata + w1*xdata^2
b = 0.0
w1 = np.ones(7)
w2 = np.ones(7)
lr = 1
epoch = 20000
b_lr = 0.0
w1_lr = np.zeros(7)
w2_lr = np.zeros(7)

plt_loss =[]
plt_epoch = []

print("--- Maximum Likelihood ---")
for e in range(epoch):
  # Calculate the value of the loss function
  x_data_square = np.square(x_data) #shape: (500,7)
  error = Y - b - np.dot(x_data, w2) - np.dot(x_data_square, w1) #shape: (500,)
  loss = np.mean(np.square(error)) # Mean Square Error

  # Calculate gradient
  b_grad = -2*np.sum(error)*1 #shape: ()
  w1_grad = -2*np.dot(error, x_data_square) #shape: (7,)
  w2_grad = -2*np.dot(error, x_data) #shape: (7,)
  
  # update learning rate
  b_lr = b_lr + b_grad**2
  w1_lr = w1_lr + w1_grad**2
  w2_lr = w2_lr + w2_grad**2

  # update parameters.
  b = b - lr/np.sqrt(b_lr) * b_grad
  w1 = w1 - lr/np.sqrt(w1_lr) * w1_grad
  w2 = w2 - lr/np.sqrt(w2_lr) * w2_grad

  plt_loss.append(np.sqrt(loss))
  plt_epoch.append(e)

  # Print "Root Mean Square Error" per 1000 epoch
  if (e+1) % 2000 == 0:
    print('epoch:{}\n Loss:{}'.format(e+1, np.sqrt(loss)))


#calculate MAP theta
hypothesis = np.linspace(0, 1, 101)
theta_hat_1 = hypothesis[np.argmax(w1)]
theta_hat_2 = hypothesis[np.argmax(w2)]
print(theta_hat_1,theta_hat_2)

# Gradient Descent M=2 for Maximum a posterior approach
# ydata = b + w2*xdata + w1*xdata^2
b = 0.0
w1 = np.ones(7)
w2 = np.ones(7)
lr = 1
epoch = 20000
b_lr = 0.0
w1_lr = np.zeros(7)
w2_lr = np.zeros(7)

pltmap_loss =[]
pltmap_epoch = []
print("--- Maximum A Posterior ---")
for e in range(epoch):
  # Calculate the value of the loss function
  x_data_square = np.square(x_data) #shape: (500,7)
  error = Y - b - theta_hat_1*np.dot(x_data, w2)- theta_hat_2*np.dot(x_data_square, w1)
  loss = np.mean(np.square(error)) # Mean Square Error

  # Calculate gradient
  b_grad = -2*np.sum(error)*1 #shape: ()
  w1_grad = -2*np.dot(error, x_data_square) #shape: (7,)
  w2_grad = -2*np.dot(error, x_data) #shape: (7,)
  
  # update learning rate
  b_lr = b_lr + b_grad**2
  w1_lr = w1_lr + w1_grad**2
  w2_lr = w2_lr + w2_grad**2

  # update parameters.
  b = b - lr/np.sqrt(b_lr) * b_grad
  w1 = w1 - lr/np.sqrt(w1_lr) * w1_grad
  w2 = w2 - lr/np.sqrt(w2_lr) * w2_grad

  pltmap_loss.append(np.sqrt(loss))
  pltmap_epoch.append(e)
  # Print "Root Mean Square Error" per 2000 epoch
  if (e+1) % 2000 == 0:
    print('epoch:{}\n MAP Loss:{}'.format(e+1, np.sqrt(loss)))

