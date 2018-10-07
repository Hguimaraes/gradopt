import numpy as np

def hinge_loss(x, y, w, b, rho):
    # Terms
    reg = rho*(np.linalg.norm(w)**2)
    cost = 1 - np.multiply(y, np.sum(w.T*x, axis = 1) - b)
    
    return np.mean(np.maximum(0, cost)**2) + reg

def dhinge_loss(x, y, w, b, rho):
    # Terms
    n = x.shape[0]
    cost = 1 - np.multiply(y, np.sum(w.T*x, axis = 1) - b)
    dreg = 2.*rho*w.T
    
    # Calculation
    dcost_w = np.sum(-2.*np.multiply(np.multiply(y[:, np.newaxis], x), np.maximum(0, cost)[:, np.newaxis]), axis = 0) + dreg
    dcost_b = np.sum(2.*np.multiply(y, np.maximum(0, cost)))
    
    return (1./n)*dcost_w.T, (1./n)*dcost_b