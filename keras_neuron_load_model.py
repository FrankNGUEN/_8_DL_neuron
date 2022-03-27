# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 18:56:19 2022

@author: FrankNGUEN
https://www.youtube.com/watch?v=hPhnqTtidnA
https://miai.vn/2020/11/06/khoa-hoc-mi-python-bai-4-python-voi-keras-phan-1/
"""
# import thu vien
import numpy as np
from numpy import  loadtxt
from sklearn.model_selection import train_test_split
from keras.models import load_model
#-----------------------------------------------------------------------------
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
#-----------------------------------------------------------------------------
#load model
model      = load_model("model/nouron_model.h5")
#evaluate model
loss, acc  = model.evaluate(X_test,y_test)
print("Loss = ", loss)
print("Accuracy = ", acc)
#-----------------------------------------------------------------------------
X_new = X_test[10]                     #Lay gia tri thuc trong data
y_new = y_test[10]                     #Gia tri output dung
X_new = np.expand_dims(X_new, axis=0)  #Chuyen chieu tensor

y_predict = model.predict(X_new)

result    = "Tieu duong (1)"
if y_predict <= 0.5:
    result = "Khong tieu duong (0)"

print("gia tri du doan = ", result)
print("gia tri dung    = ", y_new)
