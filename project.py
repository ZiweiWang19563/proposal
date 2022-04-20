f = open("data_rp.csv","r")
f = f.readlines()
import statistics
import numpy as np
import scipy.stats
stock_price = []
bonds_price = []
gold_price = []

for i in f[1:]:
    i = i.split(",")
    if i[1] != '' and i[2] != '' and i[3] != '\n':
        #Sift the data with all three assets' prices
        stock_price.append(i[1]) 
        bonds_price.append(i[2])
        gold_price.append(i[3])
        #Get the sifted prices for each asset

stock_return = []
bonds_return = []
gold_return = []
for a in range(len(stock_price)-1):
    rs = float(stock_price[a+1]) / float(stock_price[a]) - 1
    stock_return.append(rs)
    # Get the daily returns of stock
    rb = float(bonds_price[a+1]) / float(bonds_price[a]) - 1
    bonds_return.append(rb)
    # Get the daily returns of bonds
    rg = float(gold_price[a+1]) / float(gold_price[a]) - 1
    gold_return.append(rg)
    # Get the daily returns of gold


X = np.vstack([stock_return,bonds_return,gold_return])
V = np.cov(X) #Get the covariance matrix of the portfolio

def calculate_portfolio_var(w,V):
    # function that calculates portfolio risk
    w = np.matrix(w)
    return (w*V*w.T)[0,0]

def calculate_risk_contribution(w,V):
    # function that calculates asset contribution to total risk
    w = np.matrix(w)
    sigma = np.sqrt(calculate_portfolio_var(w,V))
    # Marginal Risk Contribution
    MRC = V*w.T
    # Risk Contribution
    RC = np.multiply(MRC,w.T)/sigma
    return RC

def risk_budget_objective(x,pars):
    # calculate portfolio risk
    V = pars[0]# covariance table 
    x_t = pars[1] # risk target in percent of portfolio risk 
    sig_p =  np.sqrt(calculate_portfolio_var(x,V)) # portfolio sigma
    risk_target = np.asmatrix(np.multiply(sig_p,x_t))
    asset_RC = calculate_risk_contribution(x,V)
    J = sum(np.square(asset_RC-risk_target.T))[0,0]*1000000 # sum of squared error
    return J

def total_weight_constraint(x):
    return np.sum(x)-1.0

def long_only_constraint(x):
    return x

w0 = [1/3]*3
x_t = [1/3]*3 # your risk budget percent of total portfolio risk (equal risk)
cons = ({'type': 'eq', 'fun': total_weight_constraint},
{'type': 'ineq', 'fun': long_only_constraint})
res= scipy.optimize.minimize(risk_budget_objective, w0, args=[V,x_t], method='SLSQP',constraints=cons, options={'disp': True})
w_rb = np.asmatrix(res.x)
print("Each asset's weight", w_rb)


