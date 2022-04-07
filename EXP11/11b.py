import pandas as pd import numpy as np
import matplotlib.pyplot as plt import seaborn as sns
from sklearn.svm import SVC

%matplotlib inline
from sklearn.datasets import load_breast_cancer cancer = load_breast_cancer()
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns
= np.append(cancer['feature_names'], ['target'])) df_cancer.head()

X = df_cancer.drop(['target'], axis = 1) # We drop our "target" feature and use all the remaining features in our dataframe to train the model. X.head()

y = df_cancer['target'] y.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)

svc_model = SVC() svc_model.fit(X_train, y_train) y_predict = svc_model.predict(X_test)

cm = np.array(confusion_matrix(y_test, y_predict, labels=[1,0])) confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'],
columns=['predicted_cancer','predicted_healthy']) sns.heatmap(confusion, annot=True)

