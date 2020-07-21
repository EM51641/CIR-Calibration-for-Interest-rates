import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as wb
from arch import arch_model
from scipy.optimize import minimize 
from numba import jit
import statsmodels.api as sm

T     = 4
dt    = 1/252
n=int(T/dt)
rho=-0.90
ST,Output,Cond_Var=Stock_Selector('LMT').selector #select your stock
v0=Cond_Var[-1]
MU  = np.array([0, 0])
COV = np.matrix([[1, rho], [rho, 1]])
W   = np.random.multivariate_normal(MU, COV, n)
W_S = W[:,0]
W_v = W[:,1]

kappa,theta,xi=Initialize_parameters().kappa_sigma_theta_initial_estimators(dt,Cond_Var)
kappa_list=[]
theta_list=[]
xi_list=[]
p_list=[]

for i in range(1000000):#generate 1 Million Monte carlos simulations
    H=Initialize_parameters().HeMC (v0, kappa, theta, xi, n, dt,W_v,W_S)
    kappa_list.append(H[0])
    theta_list.append(H[1])
    xi_list.append(H[2])
    p_list.append(H[3])

rho   = pd.DataFrame(p_list).dropna().mean()[0] # Correlation
kappa = pd.DataFrame(kappa_list).dropna().mean()[0]# Revert rate
theta = pd.DataFrame(theta_list).dropna().mean()[0] # Long-term Variance
xi    = pd.DataFrame(xi_list).dropna().mean()[0] # Volatility of instantaneous volatility
print(kappa,theta,xi,rho)#print the parameters