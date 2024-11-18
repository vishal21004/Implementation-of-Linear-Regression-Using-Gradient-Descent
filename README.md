# Implementation-of-Linear-Regression-Using-Gradient-Descent
### Date:
## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize weights randomly.
2. Compute predicted values.
3. Compute gradient of loss function.
4. Update weights using gradient descent.
## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: VISHAL M.A
RegisterNumber: 212222230177
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        
        predictions=(X).dot(theta).reshape(-1,1)
        
        errors=(predictions-y).reshape(-1,1)
        
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta

data=pd.read_csv("C:/Users/admin/Downloads/50_Startups.csv",header=None)
data.head()

X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)

theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")

```

## Output:
### DATASET:

![dataset](https://github.com/user-attachments/assets/865a3ceb-6cf2-41a1-b680-81939088bb4a)


## VALUE OF X:

![value of x](https://github.com/user-attachments/assets/13701176-454e-4aa5-9745-8a1c18d465ac)


## VALUE OF X1_SCALED:

![value of x1 scaled](https://github.com/user-attachments/assets/bcbe32e0-367e-4f91-ad46-03e09c183730)


## PREDICTED VALUE:

![predicted value](https://github.com/user-attachments/assets/b74c7a61-3894-404f-8402-49b8b774c5d5)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.

