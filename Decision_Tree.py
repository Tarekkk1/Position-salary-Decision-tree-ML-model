#Regression tamplete

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


#fitting the regression model
from sklearn.tree import DecisionTreeRegressor
regossor=DecisionTreeRegressor(random_state=(0))
regossor.fit(x, y)
# Predicting a new result with Desition tree Regression
y_pred=regossor.predict(([[6.5]]))


# Visualising the Desition tree Regression results (for higher resolution and smoother curve)
x_grid=np.arange(min(x),max(x),0.01)
x_grid=x_grid.reshape(len(x_grid),1)

plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regossor.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff (Regression MOdel)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
