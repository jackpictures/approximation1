import numpy as np
import random
from sklearn.linear_model import SGDRegressor

def gradient_descent(alpha, x, y,thetas, ep=0.0001, max_iter=10000):
    converged = False
    iter = 0
    loss=[]
    m = x.shape[0]  # number of samples
    # initial theta
    t = np.ndarray(shape = (len(thetas),))
    for i in range(len(thetas)):
        t[i] = np.random.random(x.shape[1])
    l = len(thetas)
    # total error, J(theta)
    k = np.ndarray(shape=y.shape)
    eps = np.ndarray(shape=y.shape)
    for p in range(len(k)):
        k[p] = 0
    for p in range(len(k)):
        for h in range(len(thetas)):
            k[p] += float(t[h]) * (float(x[p][0]) ** h)
    J = sum([(k[i] - y[i]) ** 2 for i in range(m)])
    grad = np.ndarray(shape = (l,))
    # Iterate Loop
    while not converged:
        # for each training sample, compute the gradient (d/d_theta j(theta))
        for j in range(l):
            for p in range(len(k)):
                k[p] = 0
            for p in range(len(k)):
                for h in range(len(t)):
                    k[p] += float(t[h]) * (float(x[p][0]) ** h)
            grad[j] = 1.0 / m * sum([(k[i]-y[i])*(x[i]**j) for i in range(m)])
        # update theta
        for i in range(len(t)):
            t[i]=t[i]-alpha * grad[i]

        # mean squared error
        for p in range(len(k)):
            eps[p] = 0
        for p in range(len(k)):
            for h in range(len(thetas)):
                eps[p] += float(t[h]) * (float(x[p][0]) ** h)
        np.seterr(all = 'raise')
        try:
            e = sum([(eps[i] - y[i]) ** 2 for i in range(m)])
        except:
            t = []
            return t, loss, iter

        if abs(J - e) <= ep:
            print('Converged, iterations: ', iter, '!!!')
            converged = True
        loss.append(abs(float(J-e)))
        J = e  # update error
        iter += 1  # update iter
        if iter == max_iter:
            print('Max interactions exceeded!')
            converged = True

    return t, loss,iter

def st_gradient_descent(alpha, x, y,thetas, ep=0.0001, max_iter=1000):
    converged = False
    iter = 0
    loss=[]
    m = x.shape[0]  # number of samples
    # initial theta
    t = np.ndarray(shape = (len(thetas),))
    for i in range(len(thetas)):
        t[i] = np.random.random(x.shape[1])
    l = len(thetas)
    # total error, J(theta)
    k = np.ndarray(shape=y.shape)
    eps = np.ndarray(shape=y.shape)
    for p in range(len(k)):
        k[p] = 0
    for p in range(len(k)):
        for h in range(len(thetas)):
            k[p] += float(t[h]) * (float(x[p][0]) ** h)
    J = sum([(k[i] - y[i]) ** 2 for i in range(m)])
    grad = np.ndarray(shape = (l,))
    # Iterate Loop
    while not converged:
        # for each training sample, compute the gradient (d/d_theta j(theta))
        for i in range(m):
            o = np.random.randint(0, m)
            for j in range(l):
                for p in range(len(k)):
                    k[p] = 0
                for p in range(len(k)):
                    for h in range(len(t)):
                        k[p] += float(t[h]) * (float(x[p][0]) ** h)
                grad[j] = 1.0 / m * ((k[o] - y[o]) * (x[o] ** j))

            for i in range(len(t)):
                t[i] = t[i] - alpha * grad[i]

        # mean squared error
        for p in range(len(k)):
            eps[p] = 0
        for p in range(len(k)):
            for h in range(len(thetas)):
                eps[p] += float(t[h]) * (float(x[p][0]) ** h)
        np.seterr(all='raise')
        try:
            e = sum([(eps[i] - y[i]) ** 2 for i in range(m)])
        except:
            t = []
            return t, loss, iter

        if abs(J - e) <= ep:
            print('Converged, iterations: ', iter, '!!!')
            converged = True
        loss.append(abs(float(J-e)))
        J = e  # update error
        iter += 1  # update iter
        if iter == max_iter:
            print('Max interactions exceeded!')
            converged = True

    return t, loss,iter

def mb_gradient_descent(alpha, x1, y1,thetas,batch_size, ep, max_iter):
    converged = False
    iter = 0
    loss=[]
    m = x1.shape[0]  # number of samples
    # initial theta
    x=np.ndarray(shape = (m,1),dtype='double')
    for i in range(m):
        x[i][0]=x1[i]
    y=np.ndarray(shape = (m,),dtype='double')
    for i in range(m):
        y[i]=y1[i]
    t = np.ndarray(shape = (len(thetas),),dtype='double')
    for i in range(len(thetas)):
        t[i] = np.random.random(x.shape[1])
    l = len(thetas)
    # total error, J(theta)
    k = np.ndarray(shape=y.shape,dtype='double')
    eps = np.ndarray(shape=y.shape,dtype='double')
    for p in range(len(k)):
        k[p] = 0
    for p in range(len(k)):
        for h in range(len(thetas)):
            k[p] += float(t[h]) * (float(x[p][0]) ** h)
    J = sum([(k[i] - y[i]) ** 2 for i in range(m)])
    grad = np.ndarray(shape = (l,),dtype='double')
    # Iterate Loop
    F = False
    while not converged:
        # for each training sample, compute the gradient (d/d_theta j(theta))
        for j in range(l):
            for p in range(len(k)):
                k[p] = 0
            for p in range(len(k)):
                for h in range(len(t)):
                    k[p] += float(t[h]) * (float(x[p][0]) ** h)
            o = int(random.uniform(1,m-batch_size))
            grad[j] = 1.0 / m * sum([(k[i]-y[i])*(x[i]**j) for i in range(o,o+batch_size)])
        # update theta
        for i in range(len(t)):
            t[i]=t[i]-alpha * grad[i]

        # mean squared error
        for p in range(len(k)):
            eps[p] = 0
        for p in range(len(k)):
            for h in range(len(thetas)):
                eps[p] += float(t[h]) * (float(x[p][0]) ** h)
        np.seterr(all='raise')
        try:
            e = sum([(eps[i] - y[i]) ** 2 for i in range(m)])
        except:
            t = []
            return t,loss,iter
        loss.append(abs(float(J - e)))
        if abs(J - e) <= ep and np.mean(loss[-20:])<=ep:
            print('Converged, iterations: ', iter)
            converged = True

        J = e  # update error
        iter += 1  # update iter
        if iter == max_iter:
            print('Max interactions exceeded!')
            converged = True

    return t, loss,iter

def score(y_true, y_pred, sample_weight=None):

    weight = 1
    numerator = (weight * (y_true - y_pred) ** 2).sum(axis=0,
                                                      dtype=np.float64)
    denominator = (weight * (y_true - np.average(
        y_true, axis=0, weights=sample_weight)) ** 2).sum(axis=0,
                                                          dtype=np.float64)
    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = np.ones([y_true.shape[0]])
    output_scores[valid_score] = 1 - (numerator[valid_score] /
                                      denominator[valid_score])
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.

    return np.average(output_scores)



def ls_sklearn_sgd(x, y,max_iter,alpha,ep):
    # Parameter estimation by sklearn SGD
    sgd = SGDRegressor(max_iter=1000000, alpha=0.0001, tol=0.0001)
    sgd.fit(x,y)
    t=[]
    t.append(sgd.intercept_)
    for i in range(len(sgd.coef_)):
        t.append(sgd.coef_[i])
    return t, sgd, sgd.n_iter_
