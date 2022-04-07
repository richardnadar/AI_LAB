import pandas as pd import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split from sklearn.linear_model import LinearRegression from sklearn import metrics
%matplotlib inline

dataset = pd.read_csv('student_scores.csv') dataset.head()

dataset.describe()
dataset.plot(x='Hours', y='Scores', style='o') plt.title('Hours vs Percentage') plt.xlabel('Hours Studied') plt.ylabel('Percentage Score')
plt.show()

X = dataset.iloc[:, :-1].values y = dataset.iloc[:, 1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, y,test_size=0.2, random_state=0)
print('X train shape: ', X_train.shape) print('Y train shape: ', Y_train.shape) print('X test shape: ', X_test.shape) print('Y test shape: ', Y_test.shape)


regressor = LinearRegression() regressor.fit(X_train, y_train) print(regressor.intercept_) print(regressor.coef_)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}) print(df)

print('Mean Absolute Error:',metrics.mean_absolute_error (y_test, y_pred)) print('Mean Squared Error:',metrics.mean_squared_error (y_test, y_pred)) print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error (y_test, y_pred)))