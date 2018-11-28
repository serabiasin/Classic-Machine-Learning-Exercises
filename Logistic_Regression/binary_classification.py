import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf



data=np.loadtxt('ex2data1.txt', delimiter=',')


X = np.c_[np.ones((data.shape[0],1)), data[:,0:2]]
y = np.c_[data[:,2]]


def sigmoid(z):
    with tf.Session() as sess:
      angka=sess.run(tf.sigmoid(z))
    return angka

def costFunction(theta, X, y):
    m = y.size
     #output hypothesis 1xn
    h = sigmoid(np.dot(X,theta))

    J =(-1)*1/m*(np.dot(np.log(h).T,y)+np.dot(np.log(1-h).T,(1-y)))

    return(J[0])

def gradient(theta, X, y):
    m = y.size
#     print("Before ",theta)
#     print("Reshaped :",theta.reshape(-1,1))
    #reshape (-1,1) mengubah dimensi -1 = ukuran baris tidak diketahui dan parameter 1 = memastikan bawa cuma ada satu kolom saja
    #karena hypothesis akan di operasikan dengan matrix y jadi dimensi harus sama
    h = sigmoid(np.dot(X,theta.reshape(-1,1)))


    grad =(1/m)*X.T.dot(h-y)
#     print("Sebelum flatten : ",grad)
    return(grad.flatten()) #mengubah menjadi matrix 1xn

def predict(theta_optimized,XSample,threshold=0.5):
  print("Sigmoid Score",sigmoid(np.dot(theta_optimized,XSample)))
  hasil=sigmoid(np.dot(XSample,theta_optimized))>=threshold

  return (hasil.astype('int'))

initial_theta = np.zeros(X.shape[1]) # theta = banyak feature+1
print(initial_theta)
cost = costFunction(initial_theta, X, y)
grad = gradient(initial_theta, X, y)

from scipy.optimize import minimize
#minimilasisasi theta menggunakan function minimize
optimizing=minimize(costFunction, initial_theta, args=(X,y), method=None, jac=gradient, options={'maxiter':700})

# print('Cost: \n', cost)
# print('Grad: \n', grad)

#memanggil data X yang telah di optimisasi
print(predict(optimizing.x.T,np.array([1 ,70 ,85])))
