# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries in python for Gradient Design.
2. Upload the dataset and check any null value using .isnull() function.
3. Declare the default values for linear regression.
4. Calculate the loss usinng Mean Square Error.
5. Predict the value of y.
6. Plot the graph respect to hours and scores using scatter plot function.
     
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: CHAITANYA P S
RegisterNumber:  212222230024
*/
```
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors = (predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
    
data=pd.read_csv('50_Startups.csv',header=None)
data.head()
X = (data.iloc[1:,:-2].values)
print(X)

X1=X.astype(float)
scaler = StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)

X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)

theta = linear_regression(X1_Scaled,Y1_Scaled)

new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1,new_Scaled),theta)
prediction = prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted Value:{pre}")
```   
## Output:
### df.head()
![image](https://github.com/Adhithyaram29D/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393540/ba61619c-4a51-4bc0-9e7b-e786a8667b8f)

### X and Y Values:
![image](https://github.com/Adhithyaram29D/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393540/27f66a3c-fc95-4fe8-ae41-cfb8f5d885c0)
![image](https://github.com/Adhithyaram29D/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393540/23a734d4-d18e-4e37-b36f-bb96943c9ccf)

### X and Y Scaled:
![image](https://github.com/Adhithyaram29D/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393540/4455cfbb-a384-479f-a52a-95bb075a24ef)
![image](https://github.com/Adhithyaram29D/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393540/d8d87fca-6d41-4544-8cc1-0f0423ab6b86)

### Predicted Value:
![image](https://github.com/Adhithyaram29D/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393540/1d0730d3-a94a-43fe-96cb-fe6b40d3cc4e)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
