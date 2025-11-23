import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np 
import pandas as pd 
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import math
from keras import backend
from subprocess import check_output
import time
from keras.callbacks import Callback
from keras.models import load_model

dataset = pd.read_csv('5data.csv')
# index is dropped
dataset = dataset.drop(dataset.index[0])
# date axis is dropped using drop function
dataset = dataset.drop(['date'], axis=1)
# iloc is used for index where loc is used for label
data = dataset.iloc[:, 1:]
cl = dataset.iloc[:, 0]

# convert dataframe in numpy array
data = data.values
data = data.astype('float64')
scl = MinMaxScaler()
#Scale the data
cl = cl.values.reshape(cl.shape[0],1)
cl = scl.fit_transform(cl)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, cl,test_size=0.02372854544, shuffle=False)

X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
model = load_model('lstm_model.h5')

Xt = model.predict(X_test)

plt.plot(scl.inverse_transform(y_test.reshape(-1,1)))
plt.plot(scl.inverse_transform(Xt))
plt.legend(['TRAIN', 'TEST'], loc='upper left')
plt.show()

act = []
pred = []
for i in range (len(X_test)):
        
            Xt_i = model.predict(X_test[i].reshape(1,15,1))
            print('predicted:{0}, actual:{1}'.format(scl.inverse_transform(Xt_i),scl.inverse_transform(y_test[i].reshape(-1,1))))
            pred.append(scl.inverse_transform(Xt_i)[0][0])
            act.append(scl.inverse_transform(y_test[i].reshape(-1,1))[0][0])

# Calculate Metrics
y_true_dir = []
y_pred_dir = []

# We need at least 2 points to determine direction
for i in range(1, len(act)):
    # 1 if price went up, 0 if down
    if act[i] > act[i-1]:
        y_true_dir.append(1)
    else:
        y_true_dir.append(0)
        
    if pred[i] > act[i-1]: # Compare prediction to previous ACTUAL to see if we predicted a rise from previous actual
        y_pred_dir.append(1)
    else:
        y_pred_dir.append(0)

test_loss = mean_squared_error(y_test, Xt) # MSE on scaled data as per standard loss definition
test_rmse = math.sqrt(test_loss)

accuracy = accuracy_score(y_true_dir, y_pred_dir)
precision = precision_score(y_true_dir, y_pred_dir, zero_division=0)
recall = recall_score(y_true_dir, y_pred_dir, zero_division=0)
f1 = f1_score(y_true_dir, y_pred_dir, zero_division=0)
conf_matrix = confusion_matrix(y_true_dir, y_pred_dir)

print("\n" + "="*30)
print("EVALUATION METRICS")
print("="*30)
print(f"Test Loss (MSE): {test_loss:.6f}")
print(f"Test RMSE: {test_rmse:.6f}")
print(f"Test Accuracy: {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall: {recall:.6f}")
print(f"F1-Score: {f1:.6f}")
print("Confusion Matrix:")
print(conf_matrix)
print("="*30 + "\n")



