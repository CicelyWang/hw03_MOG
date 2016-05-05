# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 10:42:58 2016

@author: Cicely
"""

import gmm
import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
def createData(n,k,d):
    _fi=np.array([0.6,0.4])
    _mu = np.array([[1,5],[-3,4]])
    _sigma = np.array([[[3,0],[0,0.5]],[[1,1],[1,2]]])
    x = np.concatenate([np.random.multivariate_normal(m,s,int(f*n)) for f,m,s in zip(_fi,_mu,_sigma)])
    
    return x,_fi,_mu,_sigma
    
    
n = 1000
d = 2
k = 2

x,_fi,_mu,_sigma = createData(n,k,d)
likelihood,fi,mu,sigma = gmm.getThetaOfMOG(x,n,k,d)

print 'Real fi:',_fi
print 'Real mean:',_mu
print 'Real covarance:',_sigma
print 'likelihood:',likelihood
print 'fi by em:',fi
print 'mean by em:',mu
print 'covarance by em:',sigma    
                         
