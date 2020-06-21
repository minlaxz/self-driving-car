import numpy as np
import matplotlib.pyplot as plt

def draw_line(x1,x2):
    ln = plt.plot(x1,x2,'-')
    plt.pause(0.0001)
    ln[0].remove()
    
def sigmoid(score):
    return 1/(1 + np.exp(-score))

def calculate_error(parameters , points , y):
    m = points.shape[0]
    p = sigmoid(points*parameters)
    #print(np.log(p).shape)
    cross_entropy = - (1 / m) * (np.log(p).T * y + np.log(1-p).T *(1 - y))
    return cross_entropy

def gradient_descent(parameters,points,y,alpha):
    m = points.shape[0]
    for i in range(1000):
        p = sigmoid(points*parameters)
        gradient = (points.T * (p - y))*(alpha/m)
        parameters = parameters - gradient
        w1=parameters.item(0)
        w2=parameters.item(1)
        b=parameters.item(2)
        x1 = np.array([points[:,1].min() , points[:,0].max()])
        x2 = -b / w2 + x1 * (- w1 / w2)
        draw_line(x1,x2)
        print('Error rate'+ str(calculate_error(parameters,points,y)))
    #draw_line(x1,x2)

n_points = 100
np.random.seed(0)
bias = np.ones(n_points)



top_region = np.array([np.random.normal(10, 2, n_points),np.random.normal(12, 2, n_points), bias]).T
bottom_region = np.array([np.random.normal(5,2,n_points),np.random.normal(6,2,n_points), bias]).T

parameters = np.matrix([np.zeros(3)]).T
y = np.array([np.zeros(n_points),np.ones(n_points)]).reshape(n_points*2,1)
all_points = np.vstack([top_region,bottom_region])

_ , ax = plt.subplots(figsize=(4,4))

ax.scatter(top_region[:,0],top_region[:,1],color='r')
ax.scatter(bottom_region[:,0],bottom_region[:,1],color='b')
gradient_descent(parameters,all_points,y,0.06)
plt.show()