import pandas as pd
import numpy as np
import pickle
from sklearn import datasets, linear_model, metrics
import matplotlib.pyplot as plt

#masukkan dataset
df=pd.read_csv("heart.csv")
df_baru = pd.get_dummies(df)
# Pembagian X dan y
X = df_baru.loc[:, df_baru.columns != 'HeartDisease']
y = df_baru['HeartDisease']
# membagi X dan y ke dalam bentuk train dan test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=1)
# membuat linear regression object
reg = linear_model.LinearRegression()
 
# melakukan model training
reg.fit(X_train, y_train)
 
# regression coefficients
print('Coefficients: ', reg.coef_)

# regression intercept
print('Intercept: ', reg.intercept_)
y_pred = reg.predict(X_test)
pickle.dump(reg, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(model.predict)