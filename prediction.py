#preprocessing

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df=pd.read_csv("diamonds.csv")
df=df.iloc[:,1:]

#splitting dataset into x and y
X=df.loc[:,["cut","color","clarity","depth","table"]]
y=df.loc[:,"price"]

#voulume
X["xyz"]=df.loc[:,"x"]*df.loc[:,"y"]*df.loc[:,"z"]

#label encoding the categorical columns
le=LabelEncoder()

X.loc[:,"cut"]=le.fit_transform(X.loc[:,"cut"])
X.loc[:,"clarity"]=le.fit_transform(X.loc[:,"clarity"])
X.loc[:,"color"]=le.fit_transform(X.loc[:,"color"])
#normalizing
norm=MinMaxScaler()
X.loc[:,["depth","xyz","table"]]=norm.fit_transform(X.loc[:,["depth","xyz","table"]])

#splitting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)



#training

from keras import backend as K

def R2_score(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout,Activation
import tensorflow as tf
from keras.optimizers import Adam

model = Sequential()
model.add(Dense(512,input_shape=(6,)))
model.add(Activation('relu'))
                           
model.add(Dense(256))
model.add(Activation('relu'))                            


model.add(Dense(128))   
model.add(Activation('relu'))                            


model.add(Dense(1))

model.compile(loss='mean_absolute_percentage_error', optimizer=Adam(0.006),metrics=[R2_score])

#fitting the dataset to X_train and y_train takes around 2 mins to train the data with validation of X_test and y_test
history=model.fit(X_train,y_train,epochs=100,batch_size=100,validation_data=(X_test, y_test))

y_pred=model.predict(X_test)

print("r2_score of test set:")
print(r2_score(y_pred,y_test))


#plotting epochs with r2_score and loss
import matplotlib.pyplot as plt

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['R2_score'])
plt.plot(history.history['val_R2_score'])
plt.title('model R2 score')
plt.ylabel('R2_score')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss_percentage')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()