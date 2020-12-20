import numpy as np
import pandas as pd
import quandl
import Initialize_parameters

dt    = 1/252
rfree = quandl.get("USTREASURY/YIELD")['1 YR']
rfree = pd.DataFrame(rfree)
Kappa,theta,xi = Initialize_parameters().kappa_sigma_theta_initial_estimators(dt,rfree)
args = [dt,len(rfree),rfree]
res = minimize(LogL,[Kappa,theta,xi],args,method='SLSQP')
Kappa,theta,xi=res.x
day_forecasts = 252
number_of_try = 1000
H = Initialize_parameters().MCR(rfree.iloc[-1,0], Kappa, theta, xi,dt,day_forecasts,number_of_try)
