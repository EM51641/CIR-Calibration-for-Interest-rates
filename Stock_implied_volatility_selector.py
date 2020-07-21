class Stock_Selector:

  def SSLECTOR(self,Ticker):

    ST=(wb.DataReader(Ticker,data_source='yahoo',start='1980-01-01')['Adj Close'])/100
    Y=ST.pct_change().dropna()*100
    am  = arch_model(Y,vol='Garch',p=1, o=1, q=1,rescale=False,mean='AR',dist='t', lags=1)
    res = am.fit(update_freq=5,disp='off')
    return ST,Y,((((res.conditional_volatility)/100)**2)*252)

  def __init__(self,Ticker):
    self.Ticker = Ticker
    self.selector=self.SSLECTOR(Ticker)
