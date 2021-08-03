# coding=utf-8 
from __future__ import division
from scipy import integrate,stats
from xlrd import open_workbook
import numpy as np
import scipy as sp
import pylab as pl
import pytwalk
import scipy.stats as ss
import matplotlib.pyplot as plt
import pandas as pd
#import emcee
import time
import corner
import seaborn as sns
import arviz as az




TotalNumIter = 600000
burnin       = 300000
LastNumIter  = 3999
NumParams    = 6
country = 7

#xi  = 5e-5
#xi2 = 5e-4


if country == 1: #CZE
    
    data=pd.read_csv('CZE-HIV-AIDS.csv',header = None)
    mu    = 0.0093
    nu    = 0.0105
    kappa = 0.00001
    N     = 10649800
    S0 = 0.999999718
    I0 = 0.000000282
    A0 = 0.
    windowYlim_D_vs_S =300
    save_results_to = 'CZE/'
    
elif  country == 2: # JPN
    
    data=pd.read_csv('JPN-HIV-AIDS.csv',header = None)
    mu    = 0.0077
    nu    = 0.0098
    kappa = 0.00003
    N     =126317000
    S0 = 0.999999952
    I0 = 8e-9
    A0 = 4e-8
    windowYlim_D_vs_S =1200
    save_results_to = 'JPN/'
    
    
elif  country == 3: # CRO
    
    data=pd.read_csv('CRO-HIV-AIDS.csv',header = None)
    mu    = 0.0089
    nu    = 0.0129
    kappa = 0.00003
    N     = 4076246
    S0 = 0.999997301
    I0 = 0.000002699
    A0 = 0.
    windowYlim_D_vs_S =120
    save_results_to = 'CRO/'    
       
elif  country == 4: # TRI-TO
    
    data=pd.read_csv('TRI-TO-HIV-AIDS.csv',header = None)
    mu    = 0.0127
    nu    = 0.0088
    kappa = 0.000086816
    N     = 1359193
    S0 = 0.999988228
    I0 = 0.000005886
    A0 = 0.000005886
    windowXlim_D_vs_S,windowYlim_D_vs_S =33,1800
    save_results_to = 'TRI-TO/'
    
elif  country == 5: # MEX
    
    data=pd.read_csv('MEX-HIV-AIDS.csv',header = None)
    mu    = 0.0171361
    nu    = 0.0058
    kappa = 0.00003
    N     = 126577691
    S0 = 0.999998459
    I0 = 8e-9
    A0 = 0.000001533
    windowXlim_D_vs_S,windowYlim_D_vs_S =35, 12000       
    save_results_to = 'MEX/'
    
    
elif  country == 6: # LUX
    
    data=pd.read_csv('LUX-HIV-AIDS.csv',header = None)
    mu    = 0.0115
    nu    = 0.0073
    kappa = 0.00003
    N     = 613012
    S0 = 0.999995106
    I0 = 0.000003263
    A0 = 0.000001631
    windowYlim_D_vs_S = 100       
    save_results_to = 'LUX/'   


elif  country == 7: # UK
    
    data=pd.read_csv('UK-HIV-AIDS.csv',header = None)
    mu    = 0.0111
    nu    = 0.0073
    kappa = 0.00003
    N     = 66440000
    S0 = 0.999948465
    I0 = 0.000043934
    A0 = 0.000007601
    windowYlim_D_vs_S = 8000       
    save_results_to = 'UK1/'     
    
else :
    print('Invalid country number')


hiv_data = data[1]
aids_data = data[2]




ttime = np.linspace(0.0,float(len(hiv_data)),len(hiv_data))


NumDaysPred = len(ttime) + 2
times_pred = np.linspace(0.0,NumDaysPred-1,NumDaysPred) 
n_pred=len(times_pred)



# pesos para la cuadratura trapezoidal
weigths = np.ones(11)
weigths[0] = 0.5
weigths[-1] = 0.5


def modelo(x,t,p):
  """
  Entrada
  x - estado
  t -tiempo
  p - parámetros
  Salida
  fx - lado derecho de la ecuación diferencial parcial que define el modelo
  """
  
  fx = np.zeros(3)
  
  fx[0] = mu - (p[0] * x[1] + nu + p[4]   )*x[0]
  fx[1] = (p[0]*x[1] + p[4] )*x[0] - (p[1] + nu - p[5])*x[1]
  fx[2] = (p[1] - p[5])*x[1] - (kappa + nu)*x[2]
  return fx




def solve(p):
    x0 = np.array([S0,I0,A0])
    nn = len(ttime)
    dt = 1.0/(10.0*nn)
    n_quad = 10*nn+1
    t_quad = np.linspace(ttime[0],ttime[-1],n_quad)
    soln = integrate.odeint(modelo,x0,t_quad,args=(p,))
    result_hiv = np.zeros(nn)
    result_aids = np.zeros(nn)
    
    for k in np.arange(len(hiv_data)):       
        x_s = soln[10*k:10*(k+1)+1,0]
        x_i = soln[10*k:10*(k+1)+1,1]
        incidence_hiv = (p[0]*x_i+ p[4])*x_s
        incidence_aids = (p[1] - p[5])*x_i
        result_hiv[k] = dt*np.dot(weigths,incidence_hiv)
        result_aids[k] = dt*np.dot(weigths,incidence_aids)      
    return p[2]*result_hiv,p[3]*result_aids



def solve_pred(p):
    x0 = np.array([S0,I0,A0])
    nn = len(times_pred)
    dt = 1.0/(10.0*nn)
    n_quad = 10*nn+1
    t_quad = np.linspace(times_pred[0],times_pred[-1],n_quad)
    soln = integrate.odeint(modelo,x0,t_quad,args=(p,))
    result_hiv = np.zeros(nn)
    result_aids = np.zeros(nn)
    
    for k in np.arange(len(times_pred)):       
        x_s = soln[10*k:10*(k+1)+1,0]
        x_i = soln[10*k:10*(k+1)+1,1]
        incidence_hiv = (p[0]*x_i+ p[4])*x_s
        incidence_aids = (p[1] - p[5])*x_i
        result_hiv[k] = dt*np.dot(weigths,incidence_hiv)
        result_aids[k] = dt*np.dot(weigths,incidence_aids)      
    return p[2]*result_hiv,p[3]*result_aids
    
    
def energy(p):
    if support(p):
        my_soln_hiv, my_soln_aids = solve(p)
#        ind1 = np.where(my_soln_hiv < 10**-8)
#        ind2 = np.where(my_soln_aids < 10**-8)
#        my_soln_hiv[ind1] = np.ones(len(ind1))*10**-8
#        my_soln_aids[ind2] = np.ones(len(ind2))*10**-8
        log_likelihood1 = -np.sum(my_soln_hiv*N-hiv_data*np.log(my_soln_hiv*N)) 
        log_likelihood2 = -np.sum(my_soln_aids*N-aids_data*np.log(my_soln_aids*N))
        #log_likelihood = -np.sum(np.linalg.norm(my_soln*N-flu_data))**2/10.0**2
#        print(log_likelihood)
       # gamma distribution parameters for p[0] = beta
        k1 = 1.0
        theta1 = 1.0
        # gamma distribution parameters for p[1] = sigma
        k2 = 1.0
        theta2 = 1.0
        # gamma distribution parameters for p[3] = K1
        k3 = 1.0
        theta3 = 10.0
        # gamma distribution parameters for p[4] = K2
        k4 = 1.0
        theta4 = 10.0
        # gamma distribution parameters for p[5] = xi1  
        k5 = 1.0
        theta5 = 10.0
        # gamma distribution parameters for p[6] = xi2
        k6 = 1.0
        theta6 = 10.0
        a1 = (k1-1)*np.log(p[0])- (p[0]/theta1)
        a2 = (k2-1)*np.log(p[1])- (p[1]/theta2) 
        a3 = (k3-1)*np.log(p[2])- (p[2]/theta3)
        a4 = (k4-1)*np.log(p[3])- (p[3]/theta4)
        a5 = (k5-1)*np.log(p[4])- (p[4]/theta5)
        a6 = (k6-1)*np.log(p[5])- (p[5]/theta6)
        log_prior = a1 + a2 + a3 + a4 + a5 + a6
        return -log_likelihood1 -log_likelihood2  - log_prior
    return -np.infty







def support(p):
    rt = True
    rt &= (0.0 < p[0] < 2.5)
    rt &= (0.0 < p[1] < 2.5)
    rt &= (0.0 < p[2] < 1.0)
    rt &= (0.0 < p[3] < 1.0)
    rt &= (0.0 < p[4] < 1e-2)
    rt &= (0.0 < p[5] < 1e-4)

    return rt

# p - parámetros p = (beta, K)
def init():
    p = np.zeros(NumParams)
    p[0] = np.random.uniform(low=0.0,high=2.5)
    p[1] = np.random.uniform(low=0.0,high=2.5)
    p[2] = np.random.uniform(low=0.0,high=1.0)
    p[3] = np.random.uniform(low=0.0,high=1.0)
    p[4] = np.random.uniform(low=0.0,high=1e-2)
    p[5] = np.random.uniform(low=0.0,high=1e-4)
    return p


def euclidean(v1, v2):
    return sum((q1-q2)**2 for q1, q2 in zip(v1, v2))**.5

if __name__=="__main__": 
#    nn = len(flu_ttime)
#    print(nn)
#    input("Press Enter to continue...") 

    sir = pytwalk.pytwalk(n=NumParams,U=energy,Supp=support)
    sir.Run(T=TotalNumIter,x0=init(),xp0=init())
    
    fig0= plt.figure()
    ax0 = plt.subplot(111)
    sir.Ana(start=burnin)
    plt.savefig(save_results_to + 'trace_plot.eps')
    
    ppc_samples_I = np.zeros((LastNumIter,len(times_pred)))
    ppc_samples_A = np.zeros((LastNumIter,len(times_pred)))

    fig2 = plt.figure()
    ax2 = plt.subplot(111)
    qq = sir.Output[sir.Output[:,-1].argsort()] # MAP
    my_soln_hiv,my_soln_aids = solve(qq[0,:]) # solve for MAP
    ax2.plot(ttime,my_soln_hiv*N,'b')
    ax2.plot(ttime,my_soln_aids*N,'g')
    
    for k in np.arange(LastNumIter): # last 1000 samples
        ppc_samples_I[k],ppc_samples_A[k]=solve_pred(sir.Output[-k,:])
        # my_soln_hiv, my_soln_aids = solve(sir.Output[-k,:]) 
        # ax2.plot(ttime,my_soln_hiv*N,"#888888", alpha=.25)
        # ax2.plot(ttime,my_soln_aids*N,"#888888", alpha=.25) 
        
    ax2.plot(ttime,hiv_data,'r.-')
    ax2.plot(ttime,aids_data,'r.-')
    if country == 4   or country == 5: # only for country =TRI-TO(4) and MEX(5)
       plt.xlim(0.0,windowXlim_D_vs_S)  
    plt.ylim(0.0,windowYlim_D_vs_S)
    plt.savefig(save_results_to + 'data_vs_samples.eps')
        
    samples = sir.Output[burnin:,:-1]
    #samples[:,1] *= N
    #samples[:,2] *= N
    map = qq[0,:-1]
    #map[1] *= N
    #map[2] *= N    
    range = np.array([(0.95*x,1.05*x) for x in map])
    corner.corner(samples,labels=[r"$\beta$", r"$\sigma$" , r"$K1$", r"$K2$", r"$\xi_1$", r"$\xi_2$"],truths=map,range=range)
    plt.savefig(save_results_to + 'corner.eps')

    plt.figure()
    plt.hist(samples[:,0]/(samples[:,1]+nu),density=True)
    plt.savefig(save_results_to + 'R_0.eps')
    
   
    plt.figure()
    alpha, beta= 1.0, 1.0
    data=ss.gamma.rvs(alpha,loc=0.0,scale=beta,size=5000)
    myHist = plt.hist(data, 100, density=True)
    plt.hist(samples[:,0],density=True)
    plt.xlim(0.0,1.0)
    plt.ylim(0.0,10.0)
    plt.savefig(save_results_to + 'beta_prior_vs_posterior.eps')
    
    def normalise(x):
#        return x / x.max(axis=0)
        return (x - x.min(0)) / x.ptp(0)
    #normalize points
#    betaSamples = normalise(samples[:,0])
#    betaSample = normalize(samples[:,0].reshape(-1,1), axis=0, norm='max')
    
    plt.figure()
    alpha, beta= 1.0, 1.0
    data=ss.gamma.rvs(alpha,loc=0.0,scale=beta,size=5000)
#    az.plot_posterior(data, var_names=["beta"], rope=(-1, 1))
    ax = sns.kdeplot(data, shade=True, color="r")
    ax = sns.kdeplot(samples[:,0], shade=True, color="b")
#    ax = sns.kdeplot(betaSample, shade=True, color="c")
    plt.savefig(save_results_to + 'beta_prior_vs_posteriorSNS.png')

    
    plt.figure()
    alpha, beta= 1.0, 1.0
    data=ss.gamma.rvs(alpha,loc=0.0,scale=beta,size=5000)
    myHist = plt.hist(data, 100, density=True)
    plt.hist(samples[:,1],density=True)
    plt.xlim(0.0,1.0)
    plt.ylim(0.0,10.0)
    plt.savefig(save_results_to + 'sigma_prior_vs_posterior.eps')
    
    plt.figure()
    alpha, beta= 1.0, 1.0
    data=ss.gamma.rvs(alpha,loc=0.0,scale=beta,size=5000)
    myHist = plt.hist(data, 100, density=True)
    plt.hist(samples[:,2],density=True)
    plt.savefig(save_results_to + 'K1_prior_vs_posterior.eps')
    
    plt.figure()
    alpha, beta= 1.0, 1.0
    data=ss.gamma.rvs(alpha,loc=0.0,scale=beta,size=5000)
    myHist = plt.hist(data, 100, density=True)
    plt.hist(samples[:,3],density=True)
    plt.savefig(save_results_to + 'K2_prior_vs_posterior.eps')
    
    
    mean_ppc_I = ppc_samples_I.mean(axis=0)
    mean_ppc_A = ppc_samples_A.mean(axis=0)
    
    CriL_ppc_I = np.percentile(ppc_samples_I,q=2.5,axis=0)
    CriU_ppc_I = np.percentile(ppc_samples_I,q=97.5,axis=0)
    
    CriL_ppc_A = np.percentile(ppc_samples_A,q=2.5,axis=0)
    CriU_ppc_A = np.percentile(ppc_samples_A,q=97.5,axis=0)
    
    print(np.shape(CriL_ppc_I))
    print(np.shape(CriU_ppc_I))
    
    print(np.shape(CriL_ppc_A))
    print(np.shape(CriU_ppc_A))
    
    
    median_ppc_beta    = np.percentile(sir.Output[-LastNumIter:,0],q=50.,axis=0)
    median_ppc_sigma   = np.percentile(sir.Output[-LastNumIter:,1],q=50.,axis=0)
    median_ppc_K1      = np.percentile(sir.Output[-LastNumIter:,2],q=50.,axis=0)
    median_ppc_K2      = np.percentile(sir.Output[-LastNumIter:,3],q=50.,axis=0)
    median_ppc_xi1     = np.percentile(sir.Output[-LastNumIter:,4],q=50.,axis=0)
    median_ppc_xi2     = np.percentile(sir.Output[-LastNumIter:,5],q=50.,axis=0)
    
    
    CriL_ppc_beta = np.percentile(samples[:,0],q=2.5,axis=0)
    CriU_ppc_beta = np.percentile(samples[:,0],q=97.5,axis=0)
    
    CriL_ppc_sigma = np.percentile(samples[:,1],q=2.5,axis=0)
    CriU_ppc_sigma = np.percentile(samples[:,1],q=97.5,axis=0)
    
    CriL_ppc_K1 = np.percentile(samples[:,2],q=2.5,axis=0)
    CriU_ppc_K1 = np.percentile(samples[:,2],q=97.5,axis=0)
    
    CriL_ppc_K2 = np.percentile(samples[:,3],q=2.5,axis=0)
    CriU_ppc_K2 = np.percentile(samples[:,3],q=97.5,axis=0)
    
    
     
    print(median_ppc_beta)
    print(median_ppc_sigma)
    print(median_ppc_K1)
    print(median_ppc_K2)
     
    print(CriL_ppc_beta)
    print(CriU_ppc_beta)
    
    print(CriL_ppc_sigma)
    print(CriU_ppc_sigma)
    
    print(CriL_ppc_K1)
    print(CriU_ppc_K1)
     
    print(CriL_ppc_K2)
    print(CriU_ppc_K2)


    fig, ax= plt.subplots(dpi=120)
    ax.plot(ttime,hiv_data, linestyle='dashed', marker='o', 
        color='mediumblue',label="HIV cases")
    ax.plot(ttime,aids_data, linestyle='dashed', marker='o', 
        color='red',label="AIDS cases")
    ax.plot(times_pred,mean_ppc_I*N, color='orangered', lw=2)
    ax.plot(times_pred,mean_ppc_A*N, color='mediumvioletred', lw=2)
    ax.fill_between(times_pred,CriL_ppc_I*N,CriU_ppc_I*N, color='orange', alpha=0.3)
    ax.fill_between(times_pred,CriL_ppc_A*N,CriU_ppc_A*N, color='magenta', alpha=0.3)
    ax.set_xlabel('Time (years)')  # Add an x-label to the axes.
    ax.legend(loc="upper right")  # Add a legend.
    ax.figure.savefig(save_results_to +'BandsPredictions.pdf')
    
        
            
    fig, ax= plt.subplots(dpi=120)
    ax.plot(ttime,hiv_data, linestyle='dashed', marker='o', 
        color='mediumblue',label="HIV cases")
    ax.plot(ttime,aids_data, linestyle='dashed', marker='o', 
        color='red',label="AIDS cases")
    # q = np.array([sir.Output[burnin:,0].mean(),
    #               sir.Output[burnin:,1].mean(), 
    #               sir.Output[burnin:,2].mean(), 
    #               sir.Output[burnin:,3].mean()])
    q_m = np.array([np.percentile(sir.Output[-LastNumIter:,0],q=50.,axis=0),
              np.percentile(sir.Output[-LastNumIter:,1],q=50.,axis=0), 
              np.percentile(sir.Output[-LastNumIter:,2],q=50.,axis=0), 
              np.percentile(sir.Output[-LastNumIter:,3],q=50.,axis=0),              
              np.percentile(sir.Output[-LastNumIter:,4],q=50.,axis=0), 
              np.percentile(sir.Output[-LastNumIter:,5],q=50.,axis=0)]) 
    
    q_CriL = np.array([np.percentile(sir.Output[-LastNumIter:,0],q=2.5,axis=0),
              np.percentile(sir.Output[-LastNumIter:,1],q=2.5,axis=0), 
              np.percentile(sir.Output[-LastNumIter:,2],q=2.5,axis=0), 
              np.percentile(sir.Output[-LastNumIter:,3],q=2.5,axis=0),
              np.percentile(sir.Output[-LastNumIter:,4],q=2.5,axis=0), 
              np.percentile(sir.Output[-LastNumIter:,5],q=2.5,axis=0)]) 
    
    q_CriU = np.array([np.percentile(sir.Output[-LastNumIter:,0],q=97.5,axis=0),
              np.percentile(sir.Output[-LastNumIter:,1],q=97.5,axis=0), 
              np.percentile(sir.Output[-LastNumIter:,2],q=97.5,axis=0), 
              np.percentile(sir.Output[-LastNumIter:,3],q=97.5,axis=0),
              np.percentile(sir.Output[-LastNumIter:,4],q=97.5,axis=0), 
              np.percentile(sir.Output[-LastNumIter:,5],q=97.5,axis=0)])
    
    my_soln_I,my_soln_A = solve_pred(q_m)    
#    my_soln_S,my_soln_I = solve(q)
    CriL_ppc_I,CriL_ppc_A = solve_pred(q_CriL)
    CriU_ppc_I,CriU_ppc_A = solve_pred(q_CriU)
    ax.plot(times_pred,my_soln_I*N, color='orangered', lw=2)
    ax.plot(times_pred,my_soln_A*N, color='mediumvioletred', lw=2)
    ax.fill_between(times_pred,CriL_ppc_I*N,CriU_ppc_I*N, color='orange', alpha=0.3)
    ax.fill_between(times_pred,CriL_ppc_A*N,CriU_ppc_A*N, color='magenta', alpha=0.3)
    ax.set_xlabel('Time (years)')  # Add an x-label to the axes.
    ax.legend(loc="upper right")  # Add a legend.
    ax.figure.savefig(save_results_to +'BayesBandsPredictions.pdf')
#    ax2.plot(ttime,my_soln_I*N,'b',lw=2)
#    ax2.fill_between(ttime, CriL_ppc_I*N, CriU_ppc_I*N, color='b',alpha=0.2)
#    ax2.plot(ttime,my_soln_A*N,'c',lw=2)
#    ax2.fill_between(ttime, CriL_ppc_A*N, CriU_ppc_A*N, color='c',alpha=0.2)
#    plt.savefig(save_results_to + 'BayesBandsPredictions.pdf')
    
    
 
       
     
    res_mean, res_var, res_std = stats.bayes_mvs(samples[:,0], alpha=0.95)
    print(res_mean)
    print(res_var)
    print(res_std)
    res_mean, res_var, res_std = stats.bayes_mvs(samples[:,1], alpha=0.95)
    print(res_mean)
    print(res_var)
    print(res_std)
    res_mean, res_var, res_std = stats.bayes_mvs(samples[:,2], alpha=0.95)
    print(res_mean)
    print(res_var)
    print(res_std)
    
    res_mean, res_var, res_std = stats.bayes_mvs(samples[:,3], alpha=0.95)
    print(res_mean)
    print(res_var)
    print(res_std)
    
    
       
    print('Norm Square of True Suceptible - simulated Suceptible =')
    print(euclidean(my_soln_hiv, hiv_data))
    print('Norm Square of True Infected - simulated Infected =')
    print(euclidean(my_soln_aids, aids_data))
    
    

plt.figure()
alpha, beta= 1.0, 1.0
data=ss.gamma.rvs(alpha,loc=0.0,scale=beta,size=5000)
myHist = plt.hist(data, 100, density=True)
plt.hist(samples[:,0],density=True)
plt.savefig(save_results_to + 'beta_prior_vs_posterior.eps')

plt.figure()
alpha, beta= 1.0, 1.0
data=ss.gamma.rvs(alpha,loc=0.0,scale=beta,size=5000)
myHist = plt.hist(data, 100, density=True)
plt.hist(samples[:,1],density=True)
plt.savefig(save_results_to + 'sigma_prior_prior_vs_posterior.eps')

plt.figure()
alpha, beta= 1.0, 1.0
data=ss.gamma.rvs(alpha,loc=0.0,scale=beta,size=5000)
myHist = plt.hist(data, 100, density=True)
plt.hist(samples[:,2],density=True)
plt.savefig(save_results_to + 'K1_vs_posterior.eps')


plt.figure()
alpha, beta= 1.0, 1.0
data=ss.gamma.rvs(alpha,loc=0.0,scale=beta,size=5000)
myHist = plt.hist(data, 100, density=True)
plt.hist(samples[:,3],density=True)
plt.savefig(save_results_to + 'K2_prior_vs_posterior.eps')




R_0 = median_ppc_beta/(median_ppc_sigma + nu)
print(R_0)














































