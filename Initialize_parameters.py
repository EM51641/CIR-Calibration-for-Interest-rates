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
    print(resl.summary())
    kappa=-resl.params[-1]
    theta=resl.params[0]/kappa
    xi=np.std(resl.resid)/np.sqrt(dt)
    return  kappa,theta,xi

  @staticmethod
  @jit(nopython=True)
  def MCS(r,v0, kappa, theta, xi, n, dt, W_v, W_S):
    vs    = np.zeros(n)
    vs[0] = v0
    Sl    = np.zeros(n)
    Sl[0] = 0
    for t in range(1,n):
        #stochastic_term=(((1/2)*(xi**2)*(kappa**-1)*(1-np.exp(-2*kappa*dt))*vt[t-1])**(1/2))*np.random.normal(0,1) #np.random.normal(0,(vt[t-1]*(1-np.exp(-kappa*dt))*(1/2)*(xi**2)/kappa)**(1/2)) *vt[t-1]
        vs[t] = np.maximum(vs[t-1]+kappa*(theta-vs[t-1])*dt+xi*np.sqrt(vs[t-1])*W_v[t]*np.sqrt(dt),0.001) #(np.exp(-kappa*dt)*vt[t-1] + theta*(1-np.exp(-kappa*dt)) + stochastic_term)
        Sl[t] = (r - 0.5*vs[t])*dt + np.sqrt((1-(rho)**2)*vs[t]*dt)*W_S[t]+rho*np.sqrt(dt*vs[t])*W_v[t]
    return vs,Sl

  @staticmethod
  @jit(nopython=True)
  def Summation(n,dt,vt,St):
    B1=((np.sum(vt[1:])*np.sum(vt[:-1]**-1))-n*np.sum(vt[1:]*(vt[:-1]**-1)))/((np.sum(vt[:-1])*np.sum(vt[:-1]**-1))-n**2)
    kappa=-(1/dt)*np.log(B1)
    B2=((np.sum(vt[1:]*vt[:-1]**-1))-n*B1)/((1-B1)*np.sum(vt[:-1]**-1))
    theta=B2
    B3=np.sum(((vt[1:]-(vt[:-1])*B1-B2*(1-B1))**2)*(n*vt[:-1])**-1)
    xi=(2*kappa*B3)/(1-B1**2)
    deltaWS=(St[1:]-(r-(1/2)*vt[:-1])*dt)/(np.sqrt(vt[:-1]))
    deltaWV=(vt[1:]-vt[:-1]-kappa*(theta-vt[:-1])*dt)/(xi*np.sqrt(vt[:-1]))
    rho=(1/(n*dt))*np.sum(deltaWS*deltaWV)
    return kappa,theta,xi,rho

  def HeMC (self,r,v0, kappa, theta, xi, n, dt,W_v,W_S):

    # Generate paths
    MC=self.MCS(r,v0, kappa, theta, xi, n, dt,W_v,W_S)
    vt=MC[0]
    St=MC[1]



    kappa,theta,xi,p=self.Summation(n,dt,vt,St)



    return kappa,theta,xi,p
