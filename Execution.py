import numpy as np
import pandas as pd
import quandl

dt    = 1/52
rfree = quandl.get("USTREASURY/YIELD")['1 YR']
rfree = rfree.loc['1995-01-01':].resample('1W').last()[1:]/100
rfree = pd.DataFrame(rfree)
Kappa,theta,xi = Initialize_parameters().kappa_sigma_theta_initial_estimators(dt,rfree)
args = [dt,len(Yield),Yield]
res = minimize(LogL,[Kappa,theta,xi],args,method='SLSQP')
Kappa,theta,xi=res.x
week_forecast = 52
J = 1000
H = Initialize_parameters().MCR(Yield.iloc[-1,0], Kappa, theta, xi,dt,week_forecast,J)
