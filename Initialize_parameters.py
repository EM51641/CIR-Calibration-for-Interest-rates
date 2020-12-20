import numpy as np
import pandas as pd
from pandas_datareader import data as wb
from scipy.optimize import minimize 
from numba import jit
import statsmodels.api as sm

class Initialize_parameters:
    
    def __init__(self):
        pass
    
    def kappa_sigma_theta_initial_estimators(self,dt,cond_v):
        DF=pd.DataFrame(cond_v).dropna()
        dif=np.array(DF.iloc[1:].values-DF.iloc[:-1].values)
        rs=np.array(DF.iloc[:-1].values)
        Y=(dif/np.sqrt(rs))
        Y=pd.DataFrame(Y)
        Y.columns=['Y']
        B1=dt/np.sqrt(rs)
        B1=pd.DataFrame(B1)
        B1.columns=['Beta1']
        B2=dt*np.sqrt(rs)
        B2=pd.DataFrame(B2)
        B2.columns=['Beta2']
        X=(B1.join(B2))
        modl=sm.OLS(Y,X)
        resl=modl.fit()
        kappa=-resl.params[-1]
        theta=resl.params[0]/kappa
        xi=np.std(resl.resid)/np.sqrt(dt)
        return  kappa,theta,xi
    
    @staticmethod
    @jit(nopython=True)
    def Monte_Carlo(cond_v, kappa, theta, xi,dt,n):
        r  = np.zeros(n)
        r[0] = cond_v
        for t in range(1,n):
            r[t] = np.maximum(r[t-1]+kappa*(theta - r[t-1])*dt + xi * np.sqrt(r[t-1])*np.sqrt(dt)*np.random.normal(0, 1),0)
        return r
    
    def LogL(self,params,args):
        kappa,theta,xi = params
        dt ,n,rfree = args
        c = 2*kappa/((xi**2)*(1-np.exp(-kappa*dt)))
        q = 2*kappa*(theta/xi**2)-1
        u = c*np.exp(-kappa*dt)*rfree[:-1].values
        v = c*rfree[1:].values
        z = 2*np.sqrt(u*v)
        bf = scipy.special.ive(q,z)
        lnL= -(n-1)* np.log(c) + np.sum(u + v - 0.5*q*np.log(v/u) - np.log(bf) - z)
        return lnL
    
    
    def MCR(self,cond_v, kappa, theta, xi,dt,n,J):
        rm = pd.DataFrame()
        for t in range(0,J):
            rm[t] = self.Monte_Carlo(cond_v, kappa, theta, xi,dt,n)
        return rm
