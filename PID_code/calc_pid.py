import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, binom
import time
from BROJA_2PID import *
import ray
import os
import logging
from ray.exceptions import RayActorError 
import warnings
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600
warnings.filterwarnings('ignore')
os.environ['RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE']='1'
#ray.init(log_to_driver=False, logging_level=logging.FATAL)
ray.init()
RAY_DEDUP_LOGS=0
@ray.remote
class calc_Poisson_PID:

    def __init__(self, pMXY, f1, f2, M, X, Y):
        self.pMXY = pMXY
        self.f1, self.f2 = f1, f2
        self.M, self.X, self.Y = M, X, Y
   
    def _convert_pmf(self, pmf):
        pmf_convert = dict()
        for i in range(pmf.shape[0]):
            for ii in range(pmf.shape[1]):
                for iii in range(pmf.shape[2]):
                    pmf_convert[(i, ii, iii)] = float(pmf[i, ii, iii])
        return pmf_convert 
   
    def _calc_analytical_q(self):

        pMX = np.sum(self.pMXY, axis=2)
        pM = np.sum(pMX, axis=1)
        qMXY = np.zeros(self.pMXY.shape)
        ##### New proposal q
        f1M, f2M = self.f1(self.M), self.f2(self.M)
        fdiff = f1M-f2M
        minf2_lst = np.linspace(0,np.min(f2M),10)
        f1M_min = np.min(f1M)
        qMXY_best = None
        IqMXY_best = None
        for minf2 in list(minf2_lst):
            upper_bias_lim = np.min([f1M_min,np.min(fdiff)+minf2])
            minf1_lst = np.linspace(0,upper_bias_lim*0.99,10)
            for minf1 in list(minf1_lst):
                qMABCD = np.ones([self.pMXY.shape[0], self.pMXY.shape[1], self.pMXY.shape[1], self.pMXY.shape[2], self.pMXY.shape[2]])
                for i in range(qMABCD.shape[0]):
                    for ii in range(qMABCD.shape[1]):
                        for iii in range(qMABCD.shape[3]):
                            if self.f1(self.M[i])-minf1>=self.f2(self.M[i])-minf2-1e-02:
                                p = (self.f2(self.M[i])-minf2+1e-03)/(self.f1(self.M[i])-minf1+1e-03)
                                qMABCD[i,ii,:,iii,:] = qMABCD[i,ii,:,iii,:]*pM[i]*poisson.pmf(self.X[ii], mu=self.f1(self.M[i])-minf1+1e-03)*binom.pmf(k=self.Y[iii],n=self.X[ii], p=p)
                            else:
                                print('fuck')
                                p = (self.f1(self.M[i])-minf1+1e-03)/(self.f2(self.M[i])-minf2+1e-03)
                                qMABCD[i,ii,:,iii,:] =  qMABCD[i,ii,:,iii,:]*pM[i]*poisson.pmf(self.Y[iii], mu=self.f2(self.M[i])-minf2+1e-03)*binom.pmf(k=self.X[ii],n=self.Y[iii], p=p)
                
                indep_noise1 =poisson.pmf(self.Y, mu=minf2).reshape([1,1,1,1,qMABCD.shape[4]])
                indep_noise2 =poisson.pmf(self.X, mu=minf1).reshape([1,1,qMABCD.shape[2],1,1])

                qMABCD = np.multiply(qMABCD, indep_noise1)        
                qMABCD =np.multiply(qMABCD,indep_noise2)


                qMXCD = np.zeros([self.pMXY.shape[0], self.pMXY.shape[1], self.pMXY.shape[2], self.pMXY.shape[2]])
                for i in range(self.pMXY.shape[1]):
                    for k in range(i+1):
                        qMXCD[:,i,:,:] = qMXCD[:,i,:,:]+qMABCD[:,k,i-k,:,:]
                qMXY = np.zeros([self.pMXY.shape[0], self.pMXY.shape[1], self.pMXY.shape[2]])
                for i in range(self.pMXY.shape[2]):
                    for k in range(i+1):
                        qMXY[:,:,i] = qMXY[:,:,i]+qMXCD[:,:,k,i-k]
                pdf_q = self._convert_pmf(qMXY)
                IqMXY = I_YZ(pdf_q)
                if IqMXY_best is not None:
                    if IqMXY<IqMXY_best:
                        IqMXY_best = IqMXY
                        qMXY_best = qMXY
                else:
                    IqMXY_best = IqMXY
                    qMXY_best = qMXY
        return qMXY_best

    def compare_analytical_and_numerical_q(self,max_iters=500, verbose=True):
        
        ### Analytical PID
        pdf_p = self._convert_pmf(self.pMXY)
        qMXY = self._calc_analytical_q()
        pdf_q = self._convert_pmf(qMXY)
        pM = np.sum(np.sum(self.pMXY, axis=2), axis=1)
        
        IpMXY = I_YZ(pdf_p)
        IqMXY = I_YZ(pdf_q)
        IqMX = I_Y(pdf_q)
        IqMY = I_Z(pdf_q)
            
        SIq = IpMXY-IqMXY
        UIxq = IqMXY-IqMY
        UIyq = IqMXY-IqMX
        RIq = IqMXY-UIxq-UIyq
        
        params = dict()
        params['max_iters'] = max_iters
        if verbose:
            output = 2
        else:
            output = 0

        try:
            returndata = pid(pdf_p, cone_solver="ECOS", output=output, **params)
            RI, UIx, UIy, SI = returndata['SI'], returndata['UIY'], returndata['UIZ'], returndata['CI']
            h_M = -1*np.sum(np.multiply(pM, np.log2(pM)))
            if verbose: 
                print("RI-Analytical:", RIq, "RI-Numerical:", RI)
                print("IqMXY_Numerical:",RI+UIx+UIy,"IqMXY-Analytical:",IqMXY, "IpMXY:", IpMXY) 
                print("SI-Analytical:", SIq, "SI-Numerical:", SI)
                print("UIx-Analytical:", UIxq, "UIx-Numerical:", UIx)
                print("UIy-Analytical:", UIyq, "UIy-Numerical:", UIy)
                print("H(M):", h_M)

            if RI>=0 and RI<= h_M:
                Idiff = IqMXY-RI-UIx-UIy
                return [Idiff, RIq, UIxq, UIyq, SIq, RI, UIx, UIy, SI, h_M]

            else:
                return [np.NaN]*10

        except BROJA_2PID_Exception:
            print("Cone Programming solver failed to find (near) optimal solution. Please report the input probability density function to abdullah.makkeh@gmail.com")
            return [np.NaN]*14 

np.random.seed(1234)
func1_lst, func2_lst = [lambda X:3*X], [lambda X:3*np.log(1+X)] ## 1 
func1_lst, func2_lst = func1_lst+[lambda X: 1.5*X**3], func2_lst+[lambda X:0.5*np.exp(X)] ## 19
func1_lst, func2_lst = func1_lst+[lambda X:0.5*X**4], func2_lst+[lambda X:0.25*np.exp(X)*np.sin(X/8*np.pi)**2] ## 20
func1_lst, func2_lst = func1_lst+[lambda X:7*X], func2_lst+[lambda X:4*X*np.sin(X/8*np.pi)**2] ## 2
func1_lst, func2_lst = func1_lst+[lambda X:X**2], func2_lst+[lambda X:X] ## 3
func1_lst, func2_lst = func1_lst+[lambda X:X**3], func2_lst+[lambda X:X**2] ## 4
func1_lst, func2_lst = func1_lst+[lambda X: 5*X], func2_lst+[lambda X:5*np.sin(X*np.pi/8)] ## 5
func1_lst, func2_lst = func1_lst+[lambda X: X**2+X**3], func2_lst+[lambda X:2*X] ## 6
func1_lst, func2_lst = func1_lst+[lambda X: 7*X], func2_lst+[lambda X:X**2*np.cos(X/16*np.pi)**2] ## 7
func1_lst, func2_lst = func1_lst+[lambda X: 8*(X-np.cos(X*np.pi/8))], func2_lst+[lambda X:2*np.sin(X*np.pi/8)] ## 8 
func1_lst, func2_lst = func1_lst+[lambda X: X*np.exp(X)], func2_lst+[lambda X:X] ## 9
func1_lst, func2_lst = func1_lst+[lambda X: 3*X**2-2*X], func2_lst+[lambda X:X+np.log(X)] ## 10
func1_lst, func2_lst = func1_lst+[lambda X: 8*np.sin(np.pi*X/10)], func2_lst+[lambda X:2*np.sqrt(X)] ## 11
func1_lst, func2_lst = func1_lst+[lambda X: 2*np.cosh(X)], func2_lst+[lambda X:6*np.sin(X/8*np.pi)**2+1e-03] ## 12
func1_lst, func2_lst = func1_lst+[lambda X: 5*X], func2_lst+[lambda X:5*np.sqrt(X)] ## 13
func1_lst, func2_lst = func1_lst+[lambda X: np.exp(X)], func2_lst+[lambda X:np.sinh(X)] ## 14
func1_lst, func2_lst = func1_lst+[lambda X: 2*np.cosh(X)], func2_lst+[lambda X:np.sinh(X)] ## 15
func1_lst, func2_lst = func1_lst+[lambda X: 2.5*X**2+5*X], func2_lst+[lambda X:0.5*X**3+X] ## 16
func1_lst, func2_lst = func1_lst+[lambda X:1.5*np.cosh(X)], func2_lst+[lambda X:0.5*X**3] ## 17
func1_lst, func2_lst = func1_lst+[lambda X: 8*X+3], func2_lst+[lambda X:7*X] ## 18

n_trials = 25
max_iters = 300
verbose = False
savecwd = os.path.join(os.getcwd(),'Poisson-PID-Results_ver3')
Idiff_func_lst, Idiff_norm_func_lst, Idiff_norm2_func_lst= [], [], []
Iqmxy_numerical_func_lst, hm_func_lst = [], []
LOAD_DATA_FLAG = True
relative_median_lst = []
idx_prev = 0
for f in range(idx_prev,len(func1_lst)):
    start_time = time.time()
    
    Idiff_lst, Idiff_norm_lst, Idiff_norm2_lst = [], [], [] 
    Iqmxy_numerical_lst, hm_lst = [], []
    for dim_M in [2,4,8]:
        if not LOAD_DATA_FLAG:
            #### Specify PMXY
            M = np.random.uniform(low=1, high=4, size=(n_trials,dim_M,))
            p_m = np.random.exponential(scale=1, size=(n_trials,dim_M))
            p_m = p_m
            p_m = p_m/np.sum(p_m, axis=1).reshape(-1,1)

            pid_calc = []
            num_cores = 25
            summ_stats = []
            true_flag = 0
            for j in range(n_trials//num_cores):
                pid_calc = []
                for n in range(num_cores):
                    
                    #### Quantize X
                    X_quant = np.array([[poisson.ppf(0.01, mu=func1_lst[f](M[n+j*num_cores,i])), poisson.ppf(0.99,mu=func1_lst[f](M[n+j*num_cores,i]))] for i in range(dim_M)])
                    min_X, max_X = np.min(X_quant[:,0]), np.max(X_quant[:,1])+10
                    min_X = 0
                    X_quant = np.arange(int(min_X), int(max_X)+1)

                    #### Quantize Y
                    Y_quant = np.array([[poisson.ppf(0.01,mu=func2_lst[f](M[n+j*num_cores,i])), poisson.ppf(0.99,mu=func2_lst[f](M[n+j*num_cores,i]))] for i in range(dim_M)])
                    min_Y, max_Y = np.min(Y_quant[:,0]), np.max(Y_quant[:,1])+10
                    min_Y = 0
                    Y_quant = np.arange(int(min_Y), int(max_Y)+1) 
                    
                    #### Calc pX|M
                    pXgM = np.array([poisson.pmf(X_quant, mu=func1_lst[f](M[n+j*num_cores,i])) for i in range(dim_M)])
                    pXgM = pXgM/np.sum(pXgM,axis=1).reshape(-1,1)
                    
                    #### Calc pY|M
                    pYgM = np.array([poisson.pmf(Y_quant, mu=func2_lst[f](M[n+j*num_cores,i])) for i in range(dim_M)])
                    pYgM = pYgM/np.sum(pYgM, axis=1).reshape(-1,1)

                    #### Choosing a cannonical pMYZ = pmpx|mpy|m, note that for any qmxyz in Deltap, RI, UIx, and UIy would remain the same. So 
                    pMXY = np.empty([dim_M, pXgM.shape[1], pYgM.shape[1]])
                    for i in range(dim_M):
                        pMXY[i] = p_m[n+j*num_cores,i]*np.matmul(pXgM[i].reshape(-1,1), pYgM[i].reshape(1,-1))
                    pid_calc.append(calc_Poisson_PID.remote(pMXY=pMXY, f1=func1_lst[f], f2=func2_lst[f], M=M[n+j*num_cores], X=X_quant, Y=Y_quant))
                summ_stats_temp = ray.get([pid_calc[i].compare_analytical_and_numerical_q.remote(max_iters=max_iters, verbose=verbose) for i in range(num_cores)])
                summ_stats = summ_stats+summ_stats_temp
                del pid_calc
            
            Idiff = np.array([summ_stats[i][0] for i in range(n_trials)]).flatten()
            RIq = np.array([summ_stats[i][1] for i in range(n_trials)]).flatten()
            UIxq = np.array([summ_stats[i][2] for i in range(n_trials)]).flatten()
            UIyq = np.array([summ_stats[i][3] for i in range(n_trials)]).flatten()
            SIq = np.array([summ_stats[i][4] for i in range(n_trials)]).flatten()
            RI = np.array([summ_stats[i][5] for i in range(n_trials)]).flatten()
            UIx = np.array([summ_stats[i][6] for i in range(n_trials)]).flatten()
            UIy = np.array([summ_stats[i][7] for i in range(n_trials)]).flatten()
            SI = np.array([summ_stats[i][8] for i in range(n_trials)]).flatten()
            hm = np.array([summ_stats[i][9] for i in range(n_trials)]).flatten()
            savecwd_rawdata = os.path.join(savecwd, 'RawData')
            if not os.path.exists(savecwd_rawdata):
                os.makedirs(savecwd_rawdata)
            np.save(os.path.join(savecwd_rawdata, 'RIq_func%d_dim%d.npy'%(f, dim_M)),RIq)
            np.save(os.path.join(savecwd_rawdata, 'UIxq_func%d_dim%d.npy'%(f, dim_M)),UIxq)
            np.save(os.path.join(savecwd_rawdata, 'UIyq_func%d_dim%d.npy'%(f, dim_M)),UIyq)
            np.save(os.path.join(savecwd_rawdata, 'SIq_func%d_dim%d.npy'%(f, dim_M)),SIq)
            np.save(os.path.join(savecwd_rawdata, 'RI_func%d_dim%d.npy'%(f, dim_M)),RI)
            np.save(os.path.join(savecwd_rawdata, 'UIx_func%d_dim%d.npy'%(f, dim_M)),UIx)
            np.save(os.path.join(savecwd_rawdata, 'UIy_func%d_dim%d.npy'%(f,dim_M)),UIy)
            np.save(os.path.join(savecwd_rawdata, 'SI_func%d_dim%d.npy'%(f, dim_M)),SI)
            np.save(os.path.join(savecwd_rawdata, 'hm_func%d_dim%d.npy'%(f, dim_M)),hm)
        else:
            savecwd_rawdata = os.path.join(savecwd, 'RawData')
            RIq = np.load(os.path.join(savecwd_rawdata, 'RIq_func%d_dim%d.npy'%(f, dim_M)))
            UIxq = np.load(os.path.join(savecwd_rawdata, 'UIxq_func%d_dim%d.npy'%(f, dim_M)))
            UIyq = np.load(os.path.join(savecwd_rawdata, 'UIyq_func%d_dim%d.npy'%(f, dim_M)))
            SIq = np.load(os.path.join(savecwd_rawdata, 'SIq_func%d_dim%d.npy'%(f, dim_M)))
            RI = np.load(os.path.join(savecwd_rawdata, 'RI_func%d_dim%d.npy'%(f, dim_M)))
            UIx = np.load(os.path.join(savecwd_rawdata, 'UIx_func%d_dim%d.npy'%(f, dim_M)))
            UIy = np.load(os.path.join(savecwd_rawdata, 'UIy_func%d_dim%d.npy'%(f,dim_M)))
            SI = np.load(os.path.join(savecwd_rawdata, 'SI_func%d_dim%d.npy'%(f, dim_M)))
            hm = np.load(os.path.join(savecwd_rawdata, 'hm_func%d_dim%d.npy'%(f, dim_M)))
            IqMXY = RIq+UIxq+UIyq 
            Idiff = IqMXY-RI-UIx-UIy
        
        Iqmxy_numerical = RI+UIx+UIy
        print("Median Difference: %.2f  +- %.2f "%(np.nanmedian(Idiff), np.sqrt(np.nanvar(Idiff))))
        print("Time Taken: %.2f min"%((time.time()-start_time)/60))
        Idiff_lst.append(Idiff[np.logical_not(np.isnan(Idiff))])
        Iqmxy_numerical_lst.append(Iqmxy_numerical[np.logical_not(np.isnan(Iqmxy_numerical))])
        hm_lst.append(hm[np.logical_not(np.isnan(hm))])
    print("###################### Function Pair %d Done !!!! #############################"%(f+1)) 
    Idiff_func_lst.append(np.concatenate(Idiff_lst, axis=0))
    Iqmxy_numerical_func_lst.append(np.concatenate(Iqmxy_numerical_lst, axis=0))
    hm_func_lst.append(np.concatenate(hm_lst, axis=0))
    relative_median_lst.append(np.median(Idiff_func_lst[f])/np.median(Iqmxy_numerical_func_lst[f])*100)

savecwd_plots = os.path.join(savecwd, 'Plots')
if not os.path.exists(savecwd_plots):
    os.makedirs(savecwd_plots)

####### Plotting Fig 1c ###########
plt.bar(np.arange(len(func1_lst)).astype(np.int32)+1,relative_median_lst, color='C2')
plt.hlines(y=4, xmin=0, xmax=21, linestyle='--', color='black', linewidth=2)
plt.hlines(y=1, xmin=0, xmax=21, linestyle='--', color='black', linewidth=2)
plt.text(x=10,y=4.1, s="4%", fontsize=20, color='black')
plt.text(x=10,y=1.1, s="1%", fontsize=20, color='black')
plt.xlabel('Functions Pair Index', fontsize=21)
plt.ylabel('% Difference in Analytical MI\n  w.r.t. Numerical MI', fontsize=21)
plt.xticks([1,5,10,15,20], ['1','5','10','15','20'],fontsize=23)
plt.yticks(fontsize=23)
plt.ylim(ymax=5)
plt.tight_layout()
plt.savefig(os.path.join(savecwd_plots,"Idiff_bar_rel.png"))
plt.show()

boxprops = dict(facecolor=(0,0,0,0),color='black',linewidth=2)
medianprops = dict(color="black",linewidth=4)
alpha = 0.2
showfliers = False

####### Plotting Fig 1b ###########
Iqmxy_numerical_func_lst = [Iqmxy_numerical_func_lst[i][Idiff_func_lst[i]>=0] for i in range(20)] ### Removing outliers where the numerical solution is worse than the analytical solution due to numerical inaccuracies of ECOS
plt.boxplot(Iqmxy_numerical_func_lst, showfliers=showfliers,patch_artist=True, boxprops=boxprops, medianprops=medianprops)
x = []
for i in range(20):
    x.append(np.random.normal(i+1, 0.01, Iqmxy_numerical_func_lst[i].shape[0]))
for i in range(20):
    plt.scatter(x[i], Iqmxy_numerical_func_lst[i], c='C0', alpha=alpha)
plt.xlabel('Function Pair Index', fontsize=21)
plt.ylabel('Numerical MI (in bits)', fontsize=21)
plt.xticks([1,5,10,15,20], ['1','5','10','15','20'],fontsize=23)
plt.yticks(fontsize=23)
plt.tight_layout()
plt.savefig(os.path.join(savecwd_plots,"Iqmxy_numerical.png"))
plt.show()

####### Plotting Fig 1a ###########
Idiff_func_lst = [Idiff_func_lst[i][Idiff_func_lst[i]>=0] for i in range(20)] ### Removing outliers where the numerical solution is worse than the analytical solution due to numerical inaccuracies of ECOS
plt.boxplot(Idiff_func_lst,patch_artist=True, showfliers=showfliers, boxprops=boxprops, medianprops=medianprops)
x = []
for i in range(20):
    x.append(np.random.normal(i+1, 0.01, Idiff_func_lst[i].shape[0]))
for i in range(20):
    plt.scatter(x[i], Idiff_func_lst[i], c='C0', alpha=alpha)
plt.xlabel('Function Pair Index', fontsize=21)
plt.ylabel('Analytical MI$-$Numerical MI\n (in bits)', fontsize=20)
plt.xticks([1,5,10,15,20], ['1','5','10','15','20'],fontsize=23)
plt.yticks(fontsize=23)
plt.tight_layout()
plt.savefig(os.path.join(savecwd_plots,"Idiff.png"))
plt.show()

####### Plotting H(M) ###########
plt.boxplot(hm_func_lst, showfliers=showfliers,patch_artist=True, boxprops=boxprops, medianprops=medianprops)
x = []
for i in range(20):
    x.append(np.random.normal(i+1, 0.01, hm_func_lst[i].shape[0]))
for i in range(20):
    plt.scatter(x[i], hm_func_lst[i], c='C0', alpha=alpha)
plt.xlabel('Function Pair Index', fontsize=22)
plt.ylabel('$H(M)$ (in bits)', fontsize=22)
plt.xticks([1,5,10,15,20], ['1','5','10','15','20'],fontsize=21)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(savecwd_plots,"hm.png"))
plt.close()
