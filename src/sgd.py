import numpy as np
from tqdm import tqdm
from copy import copy
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class SGD(object):
    def __init__(self, rho, cost, dcost):
        # Given parameters
        self.rho = rho
        self.cost_func = cost
        self.dcost = dcost
        
        # Internal parameters
        self.w = None
        self.b = None
        self.loss = []

    def __shuffle(self, x, y):
        n = x.shape[0]
        
        # Create a permutation of the n points
        permutation = np.random.permutation(n)
        
        return x[permutation], y[permutation]
    
    def __init_params(self, x):
        n = x.shape[1]
        w, b = np.random.rand(n, 1), np.random.rand()
        return w, b
    
    def __update_weigths(self, x, y, lr):
        # Create a copy of the weights
        w, b = copy(self.w), copy(self.b)
        
        # Derivative of w and b
        linear = np.sum(self.w.T*x, axis = 1) - self.b
        dw, db = self.dcost(x, y, w, b, self.rho)
        
        # Update the weigths
        w -= lr*dw
        b -= lr*db
        
        return w, b
    
    def fit(self, x, y, epochs = 100, batch_size = 32, lr = 0.005):
        n = x.shape[0]
        
        # Check dimension
        if n != y.shape[0]:
            raise ValueError("X and Y have a different number of points")
        
        self.w, self.b = self.__init_params(x)
        
        # Iterate over all epochs
        print("-- running {} epochs\n".format(epochs))
        for epoch in tqdm(range(epochs)):
            x, y = self.__shuffle(x, y)
    
            # Mini-batch
            for i in range(0, n, batch_size):
                x_batch = x[i:(i+batch_size)]
                y_batch = y[i:(i+batch_size)]
                
                # update the model
                lr = 2./(2.*self.rho + (np.sum(x_batch**2)/batch_size))/100
                self.w, self.b = self.__update_weigths(x_batch, y_batch, lr)
   
                # Compute the loss
                self.loss.append(self.cost_func(x_batch, y_batch, self.w, self.b, self.rho))

        print("-- Training accuracy = {0:.2f}%".format(self.__train_acc(x, y)*100))
    
    def __train_acc(self, x, y):
        y_pred = np.sum(self.w.T*x, axis = 1) - self.b
        y_pred[y_pred<0] = -1 
        y_pred[y_pred>=0] = 1
        
        return accuracy_score(y_pred,y)
    
    def plot_loss(self):
        plt.plot(self.loss)
        plt.show()