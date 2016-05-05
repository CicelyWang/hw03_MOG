# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 09:04:43 2016

@author: Cicely
"""

import numpy as np
from scipy.stats import multivariate_normal as mvn


def getThetaOfMOG(x,n,k,d):
    
    #initial guesses for parameters
    fi = np.random.random(k)
    fi /= fi.sum() #高斯分量所符合的多项式分布的参数
    mu = np.random.random((k,d)) #高斯分量均值
    sigma = np.array([np.eye(d)]*k)#高斯分量方差
    
    likelihood,fi,mu,sigma = EM_MOG(x,fi,mu,sigma)
    
    return likelihood,fi,mu,sigma
    
    
    
#用em算法求解GMM
def EM_MOG(x,fi,mu,sigma,tol=0.01,max_iter = 100):
    #n:样本个数    d:维度   k: 高斯分布个数    
    n,d = x.shape
    k = len(fi)
    
    likelihood_old = 0
    iter = 0
    for i in range(max_iter):
        iter += 1
        likelihood_new = 0
        
        #E step:
        #k个高斯分布，n个样本，（j,i）第i个样本由第j个高斯分布生成的概率
        #w[j,i] = p(z(i) = j|x(i))  = p(x(i)|z(i)) * p(z(i))  / p(x(i))
        w = np.zeros((k,n))
        for j in range(k):
            for i in range(n):
                w[j,i] = fi[j] * mvn(mu[j],sigma[j]).pdf(x[i])
        w /= w.sum(0)
        
        
        #M step:
        fi = np.zeros(k)
        mu = np.zeros((k,d))
        sigma = np.zeros((k,d,d))
        for j in range(k):
            for i in range(n):
                fi[j] += w[j,i]
                mu[j] += w[j,i]*x[i]
            mu[j] /= w[j,:].sum()
        fi /= n
        
        for j in range(k):
            for i in range(n):
                ys = np.reshape(x[i] - mu[j],(2,1))
                sigma[j] += w[j,i] * np.dot(ys,ys.T)
            sigma[j] /= w[j,:].sum()
        
        #update log likelihood
        for i in range(n):
            s = 0
            for j in range(k):
                s += fi[j] * mvn(mu[j],sigma[j]).pdf(x[i])
            likelihood_new += np.log(s)
        
        if np.abs(likelihood_new - likelihood_old) < tol:
            break
        likelihood_old = likelihood_new
    
    print "iteration count:", iter    
    
    return likelihood_new,fi,mu,sigma      
        
