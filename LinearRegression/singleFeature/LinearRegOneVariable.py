import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#KASUS
#FEATURE : NILAI MEMBACA
#OUTPUT : NILAI MATEMATIKA

def showGraph(xFeature,y):
    plt.scatter(xFeature[:,1], y, s=30, c='r', marker='x', linewidths=1)
    plt.xlabel("Nilai Membaca")
    plt.ylabel("Nilai Matematika")
    plt.show()

def costFunction(x,y,tetha=[[0],[0]]):
    m=y.size
    h=x.dot(tetha)
    J=1/(2*m)*np.sum(np.square(h-y))
    return (J)

def gradientDescent(x,y,alpha,theta=[[0],[0]],iteration=200):
    JHistory=np.zeros(iteration)
    m=y.size
    for indeks in range(iteration):
        h=x.dot(theta)
        theta=theta-alpha*(1/m)*(x.T.dot(h-y))
        JHistory[indeks]=costFunction(x,y,theta)

    return (JHistory,theta)

learning_rate=0.01

data_mentah=pd.read_csv('student.csv',sep=',')
xDirty=data_mentah['Reading'].values #belum digabung dengan theta0
yDirty=data_mentah['Math'].values


X=xDirty[0:100]
X=X/100
Xnol=np.ones(X.size)

xFeature=np.array([Xnol,X]).T
Y=yDirty[0:100]

JCost,theta=gradientDescent(xFeature,Y,learning_rate)
print(theta.shape)
print(theta.item(0,99),theta.item(1,99))

#get theta from gradient descent
theta0=theta.item(0,99)
theta1=theta.item(1,99)

scoreReadingTest=74
#predict the score
print(((theta0*1)+(theta1*scoreReadingTest)/100))

# plt.plot(JCost)
# plt.ylabel('Cost J')
# plt.xlabel('Iterations');
# plt.show()
# print("Jika nilai membaca : ",75,"Maka nilai matematika",theta.T.dot([1,75]))
