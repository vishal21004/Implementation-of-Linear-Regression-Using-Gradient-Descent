# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and select input features and target profit.
2. Standardize the input features and initialize weights, bias, and learning rate.
3. Calculate predictions, MSE loss, and update weights and bias using gradient descent.
4. Print final weights and bias, then plot loss versus iterations.

## Program:
```
/*
Program to implement the linear regression using gradient descent.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("ex3.xls")
X = data[["R&D Spend", "Administration", "Marketing Spend"]].values 
y = data["Profit"].values
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
m, n = X.shape          
w = np.zeros(n)         
b = 0.0
alpha = 0.01             
epochs = 1000
losses = []
for i in range(epochs):
    y_hat = np.dot(X, w) + b
    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)
    dw = (2/m) * np.dot(X.T, (y_hat - y))
    db = (2/m) * np.sum(y_hat - y)
    w = w - alpha * dw
    b = b - alpha * db

print("Final Weights:", w)
print("Final Bias:", b)

plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Iterations (Multiple Linear Regression)")
plt.show()
*/
```

## Output:

<img width="761" height="592" alt="image" src="https://github.com/user-attachments/assets/a6160fbe-5e40-4014-8763-2311edd6688b" />
<img width="767" height="137" alt="image" src="https://github.com/user-attachments/assets/649b5401-154f-4291-8b2b-aa4b37488a52" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.

