import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, hypergeom
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
class calc_Binom_PID:

    def __init__(self, pMXY, f1, f2, M, X, Y, p, bias_grid=False):
        self.pMXY = pMXY
        self.f1, self.f2 = f1, f2
        self.M, self.X, self.Y = M, X, Y
        self.p = p
        self.bias_grid = bias_grid
   
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
        if np.min(f2M)<=10 or self.bias_grid:
            minf2_lst = np.arange(0,np.min(f2M))
        else:
            minf2_lst = np.random.permutation(np.arange(0, np.min(f2M)))[:10]
        f1M_min = np.min(f1M)
        qMXY_best = None
        IqMXY_best = None
        for minf2 in list(minf2_lst):
            upper_bias_lim = np.min([f1M_min,np.min(fdiff)+minf2])
            if upper_bias_lim<=10 or len(minf2_lst)<=3 or self.bias_grid:
                minf1_lst = np.arange(0,upper_bias_lim)
            else:
                minf1_lst = np.random.permutation(np.arange(0, upper_bias_lim))[:10]
            for minf1 in list(minf1_lst):
                if np.abs(minf1) <= 1e-03 and np.abs(minf2)<=1e-03:
                    qMXY = np.ones([self.pMXY.shape[0], self.pMXY.shape[1], self.pMXY.shape[2]])
                    for i in range(qMXY.shape[0]):
                        for ii in range(qMXY.shape[1]):
                            for iii in range(qMXY.shape[2]):
                                if self.f1(self.M[i])-minf1>=self.f2(self.M[i])-minf2:
                                    M, n, N = self.f1(self.M[i])-minf1, self.f2(self.M[i])-minf2, self.X[ii]                                    
                                    if np.isnan(hypergeom.pmf(k=self.Y[iii],M=M, n=n, N=N)):
                                        qMXY[i,ii,iii] = 0
                                    else:
                                        qMXY[i,ii,iii] = qMXY[i,ii,iii]*pM[i]*binom.pmf(k=self.X[ii], n=int(self.f1(self.M[i])), p=self.p)*hypergeom.pmf(k=self.Y[iii],M=M, n=n, N=N)
                                else:
                                    print('fuck')
                                    a, b = int(self.f1(self.M[i])), int(self.f2(self.M[i])-self.f1(self.M[i]))
                                    qMXY[i,ii,iii] =  qMXY[i,ii,iii]*pM[i]*binom.pmf(self.Y[iii], n=int(self.f2(self.M[i])), p=self.p)*hypergeom.pmf(k=self.X[ii],M=self.Y[iii], n=n, N=N) 
                    pdf_q = self._convert_pmf(qMXY/np.sum(qMXY))
                    IqMXY = I_YZ(pdf_q)
                elif np.abs(minf1)>=1e-03 and np.abs(minf2)<=1e-03:
                    qMABCD = np.ones([self.pMXY.shape[0], self.pMXY.shape[1], self.pMXY.shape[1], self.pMXY.shape[2]])
                    for i in range(qMABCD.shape[0]):
                        for ii in range(qMABCD.shape[1]):
                            for iii in range(qMABCD.shape[3]):
                                if self.f1(self.M[i])-minf1>=self.f2(self.M[i])-minf2:
                                    M, n, N = self.f1(self.M[i])-minf1, self.f2(self.M[i])-minf2, self.X[ii]                                    
                                    if np.isnan(hypergeom.pmf(k=self.Y[iii],M=M, n=n, N=N)):
                                        qMABCD[i,ii,:,iii] = 0
                                    else:
                                        qMABCD[i,ii,:,iii] = qMABCD[i,ii,:,iii]*pM[i]*binom.pmf(self.X[ii], n=int(self.f1(self.M[i])-minf1), p=self.p)*hypergeom.pmf(k=self.Y[iii],M=M, n=n, N=N)
                                else:
                                    print('fuck')
                                    a, b = (self.f1(self.M[i])-minf1), (self.f2(self.M[i])-minf2)-(self.f1(self.M[i])-minf1)
                                    qMABCD[i,ii,:,iii] =  qMABCD[i,ii,:,iii]*pM[i]*binom.pmf(self.Y[iii], n=self.f2(self.M[i])-minf2, p=self.p)*hypergeom.pmf(k=self.X[ii],n=self.Y[iii], a=a, b=b)
                
                    indep_noise2 = binom.pmf(self.X, n=minf1, p=self.p).reshape([1,1,qMABCD.shape[2],1])
                    qMABCD = np.multiply(qMABCD,indep_noise2)

                    qMXY = np.zeros([self.pMXY.shape[0], self.pMXY.shape[1], self.pMXY.shape[2]])
                    for i in range(self.pMXY.shape[1]):
                        for k in range(i+1):
                            qMXY[:,i,:] = qMXY[:,i,:]+qMABCD[:,k,i-k,:]
                    qMXY[np.isnan(qMXY)]=0
                    pdf_q = self._convert_pmf(qMXY/np.sum(qMXY))
                    IqMXY = I_YZ(pdf_q)

                elif np.abs(minf1)<=1e-03 and np.abs(minf2)>=1e-03:
                    qMABCD = np.ones([self.pMXY.shape[0], self.pMXY.shape[1], self.pMXY.shape[2], self.pMXY.shape[2]])
                    for i in range(qMABCD.shape[0]):
                        for ii in range(qMABCD.shape[1]):
                            for iii in range(qMABCD.shape[3]):
                                if self.f1(self.M[i])-minf1>=self.f2(self.M[i])-minf2:
                                    M, n, N = self.f1(self.M[i])-minf1, self.f2(self.M[i])-minf2, self.X[ii]
                                    if np.isnan(hypergeom.pmf(k=self.Y[iii],M=M, n=n, N=N)):
                                        qMABCD[i,ii,iii,:] = 0
                                    else:
                                        qMABCD[i,ii,iii,:] = qMABCD[i,ii,iii,:]*pM[i]*binom.pmf(self.X[ii], n=self.f1(self.M[i])-minf1, p=self.p)*hypergeom.pmf(k=self.Y[iii],M=M, n=n, N=N)
                                else:
                                    print('fuck')
                                    a, b = (self.f1(self.M[i])-minf1), (self.f2(self.M[i])-minf2)-(self.f1(self.M[i])-minf1)
                                    qMABCD[i,ii,iii,:] =  qMABCD[i,ii,iii,:]*pM[i]*binom.pmf(self.Y[iii], n=self.f2(self.M[i])-minf2, p=self.p)*hypergeom.pmf(k=self.X[ii],n=self.Y[iii], a=a, b=b)
                
                    indep_noise1 = binom.pmf(self.Y, n=minf2, p=self.p).reshape([1,1,1,qMABCD.shape[3]])
                    qMABCD = np.multiply(qMABCD, indep_noise1)        
                    

                    qMXY = np.zeros([self.pMXY.shape[0], self.pMXY.shape[1], self.pMXY.shape[2]])
                    for i in range(self.pMXY.shape[2]):
                        for k in range(i+1):
                            qMXY[:,:,i] = qMXY[:,:,i]+qMABCD[:,:,k,i-k]
                    qMXY[np.isnan(qMXY)]=0
                    pdf_q = self._convert_pmf(qMXY/np.sum(qMXY))
                    IqMXY = I_YZ(pdf_q)
                else:
                    qMABCD = np.ones([self.pMXY.shape[0], self.pMXY.shape[1], self.pMXY.shape[1], self.pMXY.shape[2], self.pMXY.shape[2]])
                    for i in range(qMABCD.shape[0]):
                        for ii in range(qMABCD.shape[1]):
                            for iii in range(qMABCD.shape[3]):
                                if self.f1(self.M[i])-minf1>=self.f2(self.M[i])-minf2:
                                    M, n, N = self.f1(self.M[i])-minf1, self.f2(self.M[i])-minf2, self.X[ii]
                                    if np.isnan(hypergeom.pmf(k=self.Y[iii],M=M, n=n, N=N)):
                                        qMABCD[i,ii,:,iii,:] = 0
                                    else:
                                        qMABCD[i,ii,:,iii,:] = qMABCD[i,ii,:,iii,:]*pM[i]*binom.pmf(self.X[ii], n=self.f1(self.M[i])-minf1, p=self.p)*hypergeom.pmf(k=self.Y[iii],M=M, n=n, N=N)
                                else:
                                    print('fuck')
                                    a, b = (self.f1(self.M[i])-minf1), (self.f2(self.M[i])-minf2)-(self.f1(self.M[i])-minf1)
                                    qMABCD[i,ii,:,iii,:] =  qMABCD[i,ii,:,iii,:]*pM[i]*binom.pmf(self.Y[iii], n=self.f2(self.M[i])-minf2, p=self.p)*hypergeom.pmf(k=self.X[ii],n=self.Y[iii], a=a, b=b)
                
                    indep_noise1 = binom.pmf(self.Y, n=minf2, p=self.p).reshape([1,1,1,1,qMABCD.shape[4]])
                    indep_noise2 = binom.pmf(self.X, n=minf1, p=self.p).reshape([1,1,qMABCD.shape[2],1,1])
                    qMABCD = np.multiply(qMABCD, indep_noise1)        
                    qMABCD = np.multiply(qMABCD,indep_noise2)

                    qMXCD = np.zeros([self.pMXY.shape[0], self.pMXY.shape[1], self.pMXY.shape[2], self.pMXY.shape[2]])
                    for i in range(self.pMXY.shape[1]):
                        for k in range(i+1):
                            qMXCD[:,i,:,:] = qMXCD[:,i,:,:]+qMABCD[:,k,i-k,:,:]
                    qMXY = np.zeros([self.pMXY.shape[0], self.pMXY.shape[1], self.pMXY.shape[2]])
                    for i in range(self.pMXY.shape[2]):
                        for k in range(i+1):
                            qMXY[:,:,i] = qMXY[:,:,i]+qMXCD[:,:,k,i-k]
                    qMXY[np.isnan(qMXY)]=0
                    pdf_q = self._convert_pmf(qMXY/np.sum(qMXY))
                    IqMXY = I_YZ(pdf_q)
                #print("MI:", IqMXY, minf1, minf2)
                if IqMXY == 0:
                    IqMXY = np.inf
                if IqMXY_best is not None and IqMXY !=0:
                    if IqMXY<IqMXY_best:
                        IqMXY_best = IqMXY
                        qMXY_best = qMXY
                else:
                    IqMXY_best = IqMXY
                    qMXY_best = qMXY

        
        print("Best MI:",IqMXY_best, np.sum(qMXY))
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
            
            print("IqMXY:", IqMXY, "IpMXY:", IpMXY, "IqMXY-num", RI+UIx+UIy)
            if verbose: 
                print("RI-Analytical:", RIq, "RI-Numerical:", RI)
                print("IqMXY_Numerical:",RI+UIx+UIy,"IqMXY-Analytical:",IqMXY, "IpMXY:", IpMXY) 
                print("SI-Analytical:", SIq, "SI-Numerical:", SI)
                print("UIx-Analytical:", UIxq, "UIx-Numerical:", UIx)
                print("UIy-Analytical:", UIyq, "UIy-Numerical:", UIy)
                print("H(M):", h_M)

            if RI>=0 and RI<= h_M:
                Idiff = IqMXY-RI-UIx-UIy
                if RI+UIx+UIy<=1e-03 and IqMXY<=1e-03:
                    Idiff = 0
                return [Idiff, RIq, UIxq, UIyq, SIq, RI, UIx, UIy, SI, h_M]

            else:
                return [np.NaN]*10

        except BROJA_2PID_Exception:
            print("Cone Programming solver failed to find (near) optimal solution. Please report the input probability density function to abdullah.makkeh@gmail.com")
            return [np.NaN]*14 

np.random.seed(1234)
func1_lst, func2_lst = [],[]
func1_lst, func2_lst = func1_lst+[lambda X:np.array(0.5*X**2, dtype=np.int32)], func2_lst+[lambda X:X] ## 1
func1_lst, func2_lst = func1_lst+[lambda X: np.array(0.4*0.4*0.25*X**3, dtype=np.int32)], func2_lst+[lambda X: np.array(0.1*X**2, dtype=np.int32)] ## 2
func1_lst, func2_lst = func1_lst+[lambda X:4*X], func2_lst+[lambda X:np.array(X*np.sin(X/16*np.pi)**2+2, dtype=np.int32)] ## 3
func1_lst, func2_lst = func1_lst+[lambda X: 5*X], func2_lst+[lambda X:np.array(10*np.abs(np.sin(X*np.pi/16)), dtype=np.int32)+2] ## 4
func1_lst, func2_lst = func1_lst+[lambda X: np.array(0.05*(X**2+X**3), dtype=np.int32)], func2_lst+[lambda X:X] ## 5
func1_lst, func2_lst = func1_lst+[lambda X: 7*X], func2_lst+[lambda X:np.array(0.5*X**2*np.cos(X/36*np.pi)**2+2, dtype=np.int32)] ## 6
func1_lst, func2_lst = func1_lst+[lambda X: np.array(2*(X-np.cos(X*np.pi/8)), dtype=np.int32)], func2_lst+[lambda X:np.array(12*np.sin(X*np.pi/40), dtype=np.int32)+2] ## 7 
func1_lst, func2_lst = func1_lst+[lambda X: 2*X], func2_lst+[lambda X:np.array(0.004*X*np.exp(X/3), dtype=np.int32)+5] ## 8
func1_lst, func2_lst = func1_lst+[lambda X: np.array(0.1*(3*X**2-2*X), dtype=np.int32)], func2_lst+[lambda X:np.array(X+np.log(X), dtype=np.int32)] ## 9
func1_lst, func2_lst = func1_lst+[lambda X: np.array(19*np.abs(np.sin(np.pi*X/40)), dtype=np.int32)], func2_lst+[lambda X:np.array(3*np.sqrt(X), dtype=np.int32)] ## 10

bias_grid =[False]*10 #[False,False,True,True,False,False,False,False, False, True] #[False,False,False,False,False,True]
n_trials = 25
max_iters = 500
verbose = False
savecwd = os.path.join(os.getcwd(),'Binom-PID-Results')
if not os.path.exists(savecwd):
    os.makedirs(savecwd)

Idiff_func_lst, Idiff_norm_func_lst, Idiff_norm2_func_lst= [], [], []
Iqmxy_numerical_func_lst, hm_func_lst = [], []
LOAD_DATA_FLAG = True
relative_median_lst = []
idx_curr = 2
UIx_func_lst, UIy_func_lst = [], []
for f in range(len(func1_lst)):
    start_time = time.time()
    
    Idiff_lst, Idiff_norm_lst, Idiff_norm2_lst = [], [], [] 
    Iqmxy_numerical_lst, hm_lst, UIx_lst, UIy_lst = [], [], [], []
    for dim_M in [2,4]:
        if not LOAD_DATA_FLAG:
            #### Specify PMXY
            M = np.random.randint(low=5, high=15, size=(n_trials,dim_M,))
            p_m = np.random.exponential(scale=1, size=(n_trials,dim_M))
            p_m = p_m
            p_m = p_m/np.sum(p_m, axis=1).reshape(-1,1)
            
            b_p = np.random.uniform(low=0.01,high=0.99, size=(n_trials))
            pid_calc = []
            num_cores = np.min([n_trials,25])
            summ_stats = []
            true_flag = 0
            for j in range(n_trials//num_cores):
                pid_calc = []
                for n in range(num_cores):
                    
                    #### Quantize X
                    max_X = np.max(func1_lst[f](M[n+j*num_cores]))
                    X_quant = np.arange(0, max_X+1)

                    #### Quantize Y
                    max_Y = np.max(func2_lst[f](M[n+j*num_cores]))
                    Y_quant = np.arange(0, max_Y+1)
                    
                    #### Calc pX|M
                    pXgM = np.array([binom.pmf(X_quant, n=func1_lst[f](M[n+j*num_cores,i]), p=b_p[n+j*num_cores]) for i in range(dim_M)])
                    pXgM = pXgM/np.sum(pXgM,axis=1).reshape(-1,1)
                    
                    #### Calc pY|M
                    pYgM = np.array([binom.pmf(Y_quant, n=func2_lst[f](M[n+j*num_cores,i]), p=b_p[n+j*num_cores]) for i in range(dim_M)])
                    pYgM = pYgM/np.sum(pYgM, axis=1).reshape(-1,1)

                    #### Choosing a cannonical pMYZ = pmpx|mpy|m, note that for any qmxyz in Deltap, RI, UIx, and UIy would remain the same. So 
                    pMXY = np.empty([dim_M, pXgM.shape[1], pYgM.shape[1]])
                    for i in range(dim_M):
                        pMXY[i] = p_m[n+j*num_cores,i]*np.matmul(pXgM[i].reshape(-1,1), pYgM[i].reshape(1,-1))
                    pid_calc.append(calc_Binom_PID.remote(pMXY=pMXY, f1=func1_lst[f], f2=func2_lst[f], M=M[n+j*num_cores], X=X_quant, Y=Y_quant, p=b_p[n+j*num_cores], bias_grid=bias_grid[f]))
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
            print(RIq.shape)
        Iqmxy_numerical = RI+UIx+UIy
        print("Median Difference: %.2f  +- %.2f "%(np.nanmedian(Idiff), np.sqrt(np.nanvar(Idiff))))
        print("Time Taken: %.2f min"%((time.time()-start_time)/60))
        Idiff_lst.append(Idiff[np.logical_not(np.isnan(Idiff))])
        Iqmxy_numerical_lst.append(Iqmxy_numerical[np.logical_not(np.isnan(Iqmxy_numerical))])
        hm_lst.append(hm[np.logical_not(np.isnan(hm))])
        UIx_lst.append(UIx[np.logical_not(np.isnan(Iqmxy_numerical))])
        UIy_lst.append(UIy[np.logical_not(np.isnan(Iqmxy_numerical))])
    print("###################### Function Pair %d Done !!!! #############################"%(f+1)) 
    Idiff_func_lst.append(np.concatenate(Idiff_lst, axis=0))
    UIx_func_lst.append(np.concatenate(UIx_lst, axis=0))
    UIy_func_lst.append(np.concatenate(UIy_lst, axis=0))
    Iqmxy_numerical_func_lst.append(np.concatenate(Iqmxy_numerical_lst, axis=0))
    hm_func_lst.append(np.concatenate(hm_lst, axis=0))
    relative_median_lst.append(np.median(Idiff_func_lst[-1])/np.median(Iqmxy_numerical_func_lst[-1])*100)
    print(relative_median_lst)
savecwd_plots = os.path.join(savecwd, 'Plots')
if not os.path.exists(savecwd_plots):
    os.makedirs(savecwd_plots)

####### Plotting Fig 1c ###########
plt.bar(np.arange(len(func1_lst)).astype(np.int32)+1,relative_median_lst, color='C2')
plt.hlines(y=4, xmin=0, xmax=11, linestyle='--', color='black', linewidth=2)
plt.hlines(y=1, xmin=0, xmax=11, linestyle='--', color='black', linewidth=2)
plt.text(x=10,y=4.1, s="4%", fontsize=20, color='black')
plt.text(x=10,y=1.1, s="1%", fontsize=20, color='black')
plt.xlabel('Functions Pair Index', fontsize=21)
plt.ylabel('% Difference in Analytical MI\n  w.r.t. Numerical MI', fontsize=21)
plt.xticks([1,5,10], ['1','5','10'],fontsize=23)
plt.yticks(fontsize=23)
plt.ylim(ymax=5)
plt.tight_layout()
plt.savefig(os.path.join(savecwd_plots,"Idiff_bar_rel.png"))
plt.close()

boxprops = dict(facecolor=(0,0,0,0),color='black',linewidth=2)
medianprops = dict(color="black",linewidth=4)
alpha = 0.2
showfliers = False

####### Plotting Fig 1b ###########
#Iqmxy_numerical_func_lst = [Iqmxy_numerical_func_lst[i][Idiff_func_lst[i]>=0] for i in range(len(func1_lst))] ### Removing outliers where the numerical solution is worse than the analytical solution due to numerical inaccuracies of ECOS
plt.boxplot(Iqmxy_numerical_func_lst, showfliers=showfliers,patch_artist=True, boxprops=boxprops, medianprops=medianprops)
x = []
for i in range(len(func1_lst)):
    x.append(np.random.normal(i+1, 0.01, Iqmxy_numerical_func_lst[i].shape[0]))
for i in range(len(func1_lst)):
    plt.scatter(x[i], Iqmxy_numerical_func_lst[i], c='C0', alpha=alpha)
plt.xlabel('Function Pair Index', fontsize=21)
plt.ylabel('Numerical MI (in bits)', fontsize=21)
plt.xticks([1,5,10], ['1','5','10'],fontsize=23)
plt.yticks(fontsize=23)
plt.tight_layout()
plt.savefig(os.path.join(savecwd_plots,"Iqmxy_numerical.png"))
plt.close()

####### Plotting Fig 1a ###########
#Idiff_func_lst = [Idiff_func_lst[i][Idiff_func_lst[i]>=0] for i in range(len(func1_lst))] ### Removing outliers where the numerical solution is worse than the analytical solution due to numerical inaccuracies of ECOS
plt.boxplot(Idiff_func_lst,patch_artist=True, showfliers=showfliers, boxprops=boxprops, medianprops=medianprops)
x = []
for i in range(len(func1_lst)):
    x.append(np.random.normal(i+1, 0.01, Idiff_func_lst[i].shape[0]))
for i in range(len(func1_lst)):
    plt.scatter(x[i], Idiff_func_lst[i], c='C0', alpha=alpha)
plt.xlabel('Function Pair Index', fontsize=21)
plt.ylabel('Analytical MI$-$Numerical MI\n (in bits)', fontsize=20)
plt.xticks([1,5,10], ['1','5','10'],fontsize=23)
plt.yticks(fontsize=23)
plt.tight_layout()
plt.savefig(os.path.join(savecwd_plots,"Idiff.png"))
plt.close()

####### Plotting H(M) ###########
plt.boxplot(hm_func_lst, showfliers=showfliers,patch_artist=True, boxprops=boxprops, medianprops=medianprops)
x = []
for i in range(len(func1_lst)):
    x.append(np.random.normal(i+1, 0.01, hm_func_lst[i].shape[0]))
for i in range(len(func1_lst)):
    plt.scatter(x[i], hm_func_lst[i], c='C0', alpha=alpha)
plt.xlabel('Function Pair Index', fontsize=22)
plt.ylabel('$H(M)$ (in bits)', fontsize=22)
plt.xticks([1,5,10], ['1','5','10'],fontsize=21)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(savecwd_plots,"hm.png"))
plt.close()

Iqmxy_numerical_func_lst = [Iqmxy_numerical_func_lst[i][UIy_func_lst[i]>=0.01] for i in range(10)]
Idiff_func_lst = [Idiff_func_lst[i][UIy_func_lst[i]>=0.01] for i in range(10)] 
UIx_func_lst = [UIx_func_lst[i][UIy_func_lst[i]>0.01] for i in range(10)]
UIy_func_lst = [UIy_func_lst[i][UIy_func_lst[i]>0.01] for i in range(10)]

Iqmxy_numerical_func_lst = [Iqmxy_numerical_func_lst[i][UIx_func_lst[i]>=0.01] for i in range(10)]
Idiff_func_lst = [Idiff_func_lst[i][UIx_func_lst[i]>=0.01] for i in range(10)]
UIy_func_lst = [UIy_func_lst[i][UIx_func_lst[i]>0.01] for i in range(10)]
UIx_func_lst = [UIx_func_lst[i][UIx_func_lst[i]>0.01] for i in range(10)]
relative_median_nonzeroUI = []
for i in range(10):
    if len(Idiff_func_lst[i])!=0:
        relative_median_nonzeroUI.append(np.median(Idiff_func_lst[i])/np.median(Iqmxy_numerical_func_lst[i])*100)
    else:
        relative_median_nonzeroUI.append(0)

plt.boxplot(UIx_func_lst, showfliers=showfliers,patch_artist=True, boxprops=boxprops, medianprops=medianprops)
x = []
for i in range(10):
    x.append(np.random.normal(i+1, 0.01, UIx_func_lst[i].shape[0]))
for i in range(10):
    plt.scatter(x[i], UIx_func_lst[i], c='C0', alpha=alpha)
plt.xlabel('Function Pair Index', fontsize=22)
plt.ylabel('$UI_x$ (in bits)', fontsize=22)
plt.xticks([1,5,10], ['1','5','10'],fontsize=21)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(savecwd_plots,"UIx.png"))
plt.close()

plt.boxplot(UIy_func_lst, showfliers=showfliers,patch_artist=True, boxprops=boxprops, medianprops=medianprops)
x = []
for i in range(10):
    x.append(np.random.normal(i+1, 0.01, UIy_func_lst[i].shape[0]))
for i in range(10):
    plt.scatter(x[i], UIy_func_lst[i], c='C0', alpha=alpha)
plt.xlabel('Function Pair Index', fontsize=22)
plt.ylabel('$UI_y$ (in bits)', fontsize=22)
plt.xticks([1,5,10], ['1','5','10'],fontsize=21)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(savecwd_plots,"UIy.png"))
plt.close()

plt.bar(np.arange(len(func1_lst)).astype(np.int32)+1,relative_median_nonzeroUI, color='C2')
plt.hlines(y=4, xmin=0, xmax=11, linestyle='--', color='black', linewidth=2)
plt.hlines(y=1, xmin=0, xmax=11, linestyle='--', color='black', linewidth=2)
plt.text(x=10,y=4.1, s="4%", fontsize=20, color='black')
plt.text(x=10,y=1.1, s="1%", fontsize=20, color='black')
plt.xlabel('Functions Pair Index', fontsize=21)
plt.ylabel('% Difference in Analytical MI\n  w.r.t. Numerical MI', fontsize=21)
plt.xticks([1,5,10], ['1','5','10'],fontsize=23)
plt.yticks(fontsize=23)
#plt.ylim(ymax=5)
plt.tight_layout()
plt.savefig(os.path.join(savecwd_plots,"Idiff_bar_rel_nonzeroUI.png"))
plt.show()
