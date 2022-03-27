# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 16:45:20 2022

@author: FrankNGUEN
https://www.youtube.com/watch?v=hPhnqTtidnA
https://miai.vn/2020/11/06/khoa-hoc-mi-python-bai-4-python-voi-keras-phan-1/
"""
# import thu vien
from numpy import  loadtxt
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
#------------------------------------------------------------------------------
# load txt data
dataset = loadtxt('dataset/pima-indians-diabetes.data.txt', delimiter=',')
#print(dataset) # print dataset
#X,y la 1 tap day du
X = dataset[:,0:8]     # Inputs,  tu 0 --> 7
y = dataset[:,8]       # Outputs, chi 8
#Chia X into X_train_validation, X_test
#Chia y into y_train_validation, y_test
# Du lieu danh co test 20%, train + validate = 80%
X_train_val, X_test, y_train_val, y_test = train_test_split(X,y,test_size=0.2)
#Chia X_train_validation into X_train, X_val
X_train, X_val, y_train, y_val           = train_test_split(X_train_val,
                                                            y_train_val,test_size=0.2) 
#------------------------------------------------------------------------------
#training model: 
#just do one time for my_model.h5 file
#Noron 16 --> 8 --> 1
model = Sequential()
model.add(Dense(16, input_dim=8, activation='relu'))      #16 - so unit o lop Dense #relu: ham phi khu tuyen tinh
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
#complie
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#train model: feed data (X_train, y_train) into model
#1 lan duyet - 1 epoch
model.fit(X_train,y_train,epochs=100,batch_size=8, validation_data=(X_val, y_val))
#luu model
model.save("model/nouron_model.h5")    # train xong, luu lai. Load file de dung
#------------------------------------------------------------------------------

