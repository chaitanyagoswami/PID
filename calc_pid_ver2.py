import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, binom
import time
from BROJA_2PID import *
import ray
import os
import logging
from ray.exceptions import RayActorError 
os.environ['RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE']='1'
#ray.init(log_to_driver=False, logging_level=logging.FATAL)
@ray.remote
class calc_Poisson_PID:

    def __init__(self, pMXY, qMXY, pM):
        self.pdf_p = pMXY 
        self.pdf_q = qMXY
        self.pM = pM

    def compare_analytical_and_numerical_q(self,max_iters=500, verbose=True):
        
        ### Analytical PID
        
        IpMXY = I_YZ(self.pdf_p)
        IqMXY = I_YZ(self.pdf_q)
        IqMX = I_Y(self.pdf_q)
        IqMY = I_Z(self.pdf_q)
            
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
            returndata = pid(self.pdf_p, cone_solver="ECOS", output=output, **params)
            
            RI, UIx, UIy, SI = returndata['SI'], returndata['UIY'], returndata['UIZ'], returndata['CI']
            h_M = -1*np.sum(np.multiply(self.pM, np.log2(self.pM)))
            if verbose: 
                print("RI-Analytical:", RIq, "RI-Numerical:", RI)
                print("IqMXY_Numerical:",RI+UIx+UIy,"IqMXY-Analytical:",IqMXY, "IpMXY:", IpMXY) 
                print("SI-Analytical:", SIq, "SI-Numerical:", SI)
                print("UIx-Analytical:", UIxq, "UIx-Numerical:", UIx)
                print("UIy-Analytical:", UIyq, "UIy-Numerical:", UIy)
                print("H(M):", h_M)

            if RI>=0 and RI<= h_M:
                max_term_diff = (np.max([RIq,UIxq,UIyq,SIq])-np.max([RI,UIx,UIy,SI]))
                max_term_normHm = (np.max([RIq,UIxq,UIyq,SIq])-np.max([RI,UIx,UIy,SI]))/h_M*100 #np.max([RI+UIx+UIy,0.1*h_M])*100
                Idiff = IqMXY-RI-UIx-UIy
                Idiff_normHm = (IqMXY - RI-UIx-UIy)/h_M*100#np.max([RI+UIx+UIy,0.1*h_M])*100
                Idiff_normImxy = (IqMXY - RI-UIx-UIy)/np.max([RI+UIx+UIy,0.1*h_M])*100
                return [max_term_diff, max_term_normHm, Idiff, Idiff_normHm, RIq, UIxq, UIyq, SIq, RI, UIx, UIy, SI, Idiff_normImxy, h_M]

            else:
                return [np.NaN]*14

        except BROJA_2PID_Exception:
            print("Cone Programming solver failed to find (near) optimal solution. Please report the input probability density function to abdullah.makkeh@gmail.com")
            return [np.NaN]*14 

np.random.seed(1234)
func1_lst =  [lambda X:3*X, lambda X:3*X, lambda X:X**2,lambda X:X**3,lambda X:X**4,lambda X:3*X,lambda X:X**2+X**3,lambda X: 3*X,lambda X: 3*np.cos((X-1)/3*np.pi/2),lambda X: X*np.exp(X)]
func2_lst = [lambda X:3*np.log(1+X), lambda X:3*X*np.sin(X)**2,lambda X: X,lambda X: X ** 2,lambda X:np.exp(X)*np.sin(X)**2,lambda X: 3*np.sin((X-1)/3*np.pi/2),lambda X:2*X,lambda X: 3*X*np.cos(X)**2,lambda X: 3*np.sin((X-1)/3*np.pi/2),lambda X: X]
func1_lst = func1_lst+[lambda X:X**2-X, lambda X:2*np.sin(2*X)**2, lambda X: 3*np.abs(np.cosh(X)), lambda X:np.sqrt(X), lambda X:np.exp(X), lambda X:np.exp(-1*X), lambda X:2*X**2+3*X, lambda X:np.cosh(X)*np.exp(-1*X), lambda X:X,lambda X:3*X**3]
func2_lst = func2_lst+[lambda X:X+np.log(X), lambda X:2*np.sin(X)**2, lambda X: np.abs(np.cos(X))**2, lambda X:X, lambda X:np.sinh(X), lambda X:np.log(X), lambda X:X**3+2*X, lambda X:X**3, lambda X:2*X+3,lambda X: np.exp(X)]
n_trials = 50
max_iters = 100
verbose = True
savecwd = os.path.join(os.getcwd(),'Poisson-PID-Results')
Idiff_func_lst, Idiff_norm_func_lst, Idiff_norm2_func_lst= [], [], []
Iqmxy_numerical_func_lst, hm_func_lst = [], []
LOAD_DATA_FLAG  = False

for f in range(len(func1_lst)):
    start_time = time.time()
    
    Idiff_lst, Idiff_norm_lst, Idiff_norm2_lst = [], [], [] 
    Iqmxy_numerical_lst, hm_lst = [], []
    for dim_M in [4,8,16]:
        if not LOAD_DATA_FLAG:
            #### Specify PMXY
            M = np.random.uniform(low=1, high=4, size=(n_trials,dim_M,))
            p_m = np.random.exponential(scale=1, size=(n_trials,dim_M))
            p_m = p_m/np.sum(p_m, axis=1).reshape(-1,1)
            pid_calc = []
            if f==19:
                num_cores = 2
            elif f==4:
                num_cores = 10
            else:
                num_cores = 1
            summ_stats = []
            for j in range(n_trials//num_cores):
                pid_calc = []
                for n in range(num_cores):
                    
                    minf1 = np.min(func1_lst[f](M[n+j*num_cores]))
                    minf2 = np.min(func2_lst[f](M[n+j*num_cores]))
                    pMXY, qMXY = dict(), dict()
                    for m in range(dim_M):        
                        ########################### Specify PMXY #################################
                        ##########################################################################
                        #### Quantize X
                        min_X, max_X = poisson.ppf(0.01, mu=func1_lst[f](M[n+j*num_cores,m])), poisson.ppf(0.99,mu=func1_lst[f](M[n+j*num_cores,m]))
                        min_X = 0
                        X_quant = np.arange(int(min_X), int(max_X)+1)

                        #### Quantize Y
                        min_Y, max_Y =poisson.ppf(0.01,mu=func2_lst[f](M[n+j*num_cores,m])), poisson.ppf(0.99,mu=func2_lst[f](M[n+j*num_cores,m]))
                        min_Y = 0
                        Y_quant = np.arange(int(min_Y), int(max_Y)+1) 
                    
                        #### Calc pX|M
                        pXgM = poisson.pmf(X_quant, mu=func1_lst[f](M[n+j*num_cores,m])) 
                        pXgM = pXgM/np.sum(pXgM)
                    
                        #### Calc pY|M
                        pYgM = poisson.pmf(Y_quant, mu=func2_lst[f](M[n+j*num_cores,m]))
                        pYgM = pYgM/np.sum(pYgM)

                        #### Choosing a cannonical pMYZ = pmpx|mpy|m, note that for any qmxyz in Deltap, RI, UIx, and UIy would remain the same. So 

                        pXYgM = np.matmul(pXgM.reshape(-1,1), pYgM.reshape(1,-1))
                        pXYgM = pXYgM/np.sum(pXYgM)
                        pMXY_m = pXYgM*p_m[n+j*num_cores,m]
                        for x in range(len(X_quant)):
                            for y in range(len(Y_quant)):
                                pMXY[(m,int(X_quant[x]),int(Y_quant[y]))] = float(pMXY_m[x,y])
                        ##########################################################################
                        ##########################################################################
                        
                        ########################### Specify QMXY #################################
                        ##########################################################################
                        #### New proposal q
                        f1M, f2M = func1_lst[f](M[n+j*num_cores,m])-minf1, func2_lst[f](M[n+j*num_cores,m])-minf2
                        if f1M>=f2M:
                            min_X1, max_X1 = poisson.ppf(0.01, mu=f1M), poisson.ppf(0.99,mu=f1M)
                            X1_quant = np.arange(int(min_X1), int(max_X1)+1)
                            
                            min_X2, max_X2 = poisson.ppf(0.01, mu=minf1), poisson.ppf(0.99,mu=minf1)
                            X2_quant = np.arange(int(min_X2), int(max_X2)+1)
                            
                            min_Y2, max_Y2 = poisson.ppf(0.01, mu=minf2), poisson.ppf(0.99,mu=minf2)
                            Y2_quant = np.arange(int(min_Y2), int(max_Y2)+1)
                        
                            qX2 =poisson.pmf(X2_quant, mu=minf1)
                            qX2 = qX2/np.sum(qX2)
                            qY2 =poisson.pmf(Y2_quant, mu=minf2)
                            qY2 = qY2/np.sum(qY2)
                            qX1 =poisson.pmf(X1_quant, mu=f1M)
                            qX1 = qX1/np.sum(qX1)
                            p = (f2M+1e-06)/(f1M+1e-06)                            
                            qX1Y_lst, Y_quant_lst = [], []
                            minY_quant, maxY_quant = np.inf, 0
                            for x1 in range(len(X1_quant)):
                                Y1_quant = np.arange(0,x1+1)
                                qX1Y1 = qX1[x1]*binom.pmf(k=Y1_quant,n=X1_quant[x1], p=p)
                                
                                Y_quant = np.arange(0+np.min(Y2_quant),x1+int(np.max(Y2_quant))+1) 
                                if np.min(Y_quant)<minY_quant:
                                    minY_quant = int(np.min(Y_quant))
                                if np.max(Y_quant)>maxY_quant:
                                    maxY_quant = int(np.max(Y_quant))
                                qX1Y = np.zeros(len(Y_quant))
                                for sum_val in range(int(np.min(Y_quant)),int(np.max(Y_quant))+1):
                                    for k in range(int(np.max([0,sum_val-np.max(Y2_quant)])), int(np.min([x1,sum_val-np.min(Y2_quant)]))+1):
                                        qX1Y[sum_val] = qX1Y[sum_val]+qX1Y1[k]*qY2[sum_val-k-np.min(Y2_quant)]
                                qX1Y_lst.append(qX1Y.copy()) 
                                Y_quant_lst.append(Y_quant.copy())

                            Y_quant = np.arange(minY_quant, maxY_quant+1)
                            for x1 in range(len(X1_quant)):
                                if Y_quant_lst[x1][0]-minY_quant>0:
                                    qX1Y_lst[x1] = np.concatenate([np.zeros(Y_quant_lst[x1][0]-minY_quant), qX1Y_lst[x1]], axis=0)
                                if maxY_quant-int(np.max(Y_quant_lst[x1]))>0:
                                    qX1Y_lst[x1] = np.concatenate([qX1Y_lst[x1],np.zeros(maxY_quant-int(np.max(Y_quant_lst[x1])))], axis=0)                           
                            X_quant = np.arange(int(np.min(X1_quant)+np.min(X2_quant)), int(np.max(X1_quant)+np.max(X2_quant)))
                            qXY_lst = []
                            for sum_val in range(int(np.min(X_quant)),int(np.max(X_quant))+1):
                                qXY = np.zeros([len(Y_quant)])
                                for k in range(int(np.max([np.min(X1_quant), sum_val-np.max(X2_quant)])),int(np.min([np.max(X1_quant), sum_val-np.min(X2_quant)]))+1):
                                    qXY = qXY+qX1Y_lst[int(k-np.min(X1_quant))]*qX2[sum_val-k-int(np.min(X2_quant))]
                                qXY_lst.append(qXY.copy())
                            for x in range(len(X_quant)):
                                for y in range(len(Y_quant)):
                                    qMXY[(m,int(X_quant[x]),int(Y_quant[y]))] = float(p_m[n+j*num_cores,m]*qXY_lst[x][y])
                        else:
                            min_Y1, max_Y1 = poisson.ppf(0.01, mu=f2M), poisson.ppf(0.99,mu=f2M)
                            Y1_quant = np.arange(int(min_Y1), int(max_Y1)+1)
                            
                            min_X2, max_X2 = poisson.ppf(0.01, mu=minf1), poisson.ppf(0.99,mu=minf1)
                            X2_quant = np.arange(int(min_X2), int(max_X2)+1)
                            
                            min_Y2, max_Y2 = poisson.ppf(0.01, mu=minf2), poisson.ppf(0.99,mu=minf2)
                            Y2_quant = np.arange(int(min_Y2), int(max_Y2)+1)
                        
                            qX2 =poisson.pmf(X2_quant, mu=minf1)
                            qX2 = qX2/np.sum(qX2)
                            qY2 =poisson.pmf(Y2_quant, mu=minf2)
                            qY2 = qY2/np.sum(qY2)
                            qY1 =poisson.pmf(Y1_quant, mu=f1M)
                            qY1 = qY1/np.sum(qY1)

                            p = (f1M+1e-06)/(f2M+1e-06)

                            qY1X_lst, X_quant_lst = [], [] 
                            minX_quant, maxX_quant = np.inf, 0

                            for y1 in range(len(Y1_quant)):
                                X1_quant = np.arange(0,y1+1)
                                qX1Y1 = qY1[y1]*binom.pmf(k=X1_quant,n=Y1_quant[y1], p=p)
                                
                                X_quant = np.arange(0+np.min(X2_quant),y1+int(np.max(X2_quant))+1) 
                                if np.min(X_quant)<minX_quant:
                                    minX_quant = int(np.min(X_quant))
                                if np.max(X_quant)>maxY_quant:
                                    maxX_quant = int(np.max(X_quant))

                                qY1X = np.zeros(len(Y_quant))
                                for sum_val in range(int(np.min(X_quant)),int(np.max(X_quant))+1):
                                    for k in range(int(np.max([0,sum_val-np.max(X2_quant)])), int(np.min([y1,sum_val-np.min(X2_quant)]))+1):
                                        qY1X[sum_val] = qY1X[sum_val]+qX1Y1[k]*qX2[sum_val-k-np.min(X2_quant)]
                                qY1X_lst.append(qY1X) 
                                X_quant_lst.append(X_quant)

                            X_quant = np.arange(minX_quant, maxX_quant+1)
                            for y1 in range(len(Y1_quant)):
                                if Y_quant_lst[y1][0]-minX_quant>0:
                                    qX1Y_lst[y1] = np.concatenate([np.zeros(X_quant_lst[y1][0]-minX_quant), qX1Y_lst[y1]], axis=0)
                                if maxX_quant-int(np.max(X_quant_lst[y1]))>0:
                                    qX1Y_lst[y1] = np.concatenate([qX1Y_lst[y1],np.zeros(maxX_quant-int(np.max(X_quant_lst[y1])))], axis=0)    
                            
                            Y_quant = np.arange(int(np.min(Y1_quant)+np.min(Y2_quant)), int(np.max(Y1_quant)+np.max(Y2_quant)))
                            qXY_lst = []
                            for sum_val in range(int(np.min(Y_quant)),int(np.max(Y_quant))+1):
                                qXY = np.zeros([len(Y_quant)])
                                for k in range(int(np.max([np.min(Y1_quant), sum_val-np.max(Y2_quant)])),int(np.min([np.max(Y1_quant), sum_val-np.min(Y2_quant)]))+1):
                                    qXY = qXY+qY1X_lst[int(k-np.min(Y1_quant))]*qY2[sum_val-k-int(np.min(Y2_quant))]
                                qXY_lst.append(qXY)
                            
                            for y in range(len(Y_quant)):
                                for x in range(len(X_quant)):
                                    qMXY[(m,int(X_quant_lst[x]),int(Y_quant[y]))] = float(p_m[n+j*num_cores,m]*qXY_lst[y][x])

                        ##########################################################################
                        ##########################################################################
                    #sum_p = 0
                    #for key, val in pMXY.items():
                    #    sum_p = sum_p+val
                    #print("Sum P:", sum_p)
                    #sum_q = 0
                    #for key, val in qMXY.items():
                    #    sum_q = sum_q+val
                    #print("Sum Q:", sum_p)

                    pid_calc.append(calc_Poisson_PID.remote(pMXY=pMXY, qMXY=qMXY, pM=p_m[n+j*num_cores]))     
                summ_stats_temp = ray.get([pid_calc[i].compare_analytical_and_numerical_q.remote(max_iters=max_iters, verbose=verbose) for i in range(num_cores)])
                exit()
                summ_stats = summ_stats+summ_stats_temp
                del pid_calc
            max_term_diff = np.array([summ_stats[i][0] for i in range(n_trials)]).flatten()
            max_term_normHm = np.array([summ_stats[i][1] for i in range(n_trials)]).flatten()
            Idiff = np.array([summ_stats[i][2] for i in range(n_trials)]).flatten()
            Idiff_normhm = np.array([summ_stats[i][3] for i in range(n_trials)]).flatten()
            RIq = np.array([summ_stats[i][4] for i in range(n_trials)]).flatten()
            UIxq = np.array([summ_stats[i][5] for i in range(n_trials)]).flatten()
            UIyq = np.array([summ_stats[i][6] for i in range(n_trials)]).flatten()
            SIq = np.array([summ_stats[i][7] for i in range(n_trials)]).flatten()
            RI = np.array([summ_stats[i][8] for i in range(n_trials)]).flatten()
            UIx = np.array([summ_stats[i][9] for i in range(n_trials)]).flatten()
            UIy = np.array([summ_stats[i][10] for i in range(n_trials)]).flatten()
            SI = np.array([summ_stats[i][11] for i in range(n_trials)]).flatten()
            Idiff_normImxy = np.array([summ_stats[i][12] for i in range(n_trials)]).flatten()
            hm = np.array([summ_stats[i][13] for i in range(n_trials)]).flatten()
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
            RIq = np.load(os.path.join(savecwd_rawdata, 'RIq_func%d_dim%d.npy'%(f, dim_M)))
            UIxq = np.load(os.path.join(savecwd_rawdata, 'UIxq_func%d_dim%d.npy'%(f, dim_M)))
            UIyq = np.load(os.path.join(savecwd_rawdata, 'UIyq_func%d_dim%d.npy'%(f, dim_M)))
            SIq = np.load(os.path.join(savecwd_rawdata, 'SIq_func%d_dim%d.npy'%(f, dim_M)))
            RI = np.load(os.path.join(savecwd_rawdata, 'RI_func%d_dim%d.npy'%(f, dim_M)))
            UIx = np.load(os.path.join(savecwd_rawdata, 'UIx_func%d_dim%d.npy'%(f, dim_M)))
            UIy = np.load(os.path.join(savecwd_rawdata, 'UIy_func%d_dim%d.npy'%(f,dim_M)))
            SI = np.load(os.path.join(savecwd_rawdata, 'SI_func%d_dim%d.npy'%(f, dim_M)))
            hm = np.load(os.path.join(savecwd_rawdata, 'hm_func%d_dim%d.npy'%(f, dim_M)))
        Iqmxy_numerical = RI+UIx+UIy
        print("Max Normalized-2 Difference: %.2f  +- %.2f "%(np.nanmedian(Idiff_normImxy), np.sqrt(np.nanvar(Idiff_normImxy))))
        print("Max Normalized-1 Difference: %.2f  +- %.2f "%(np.nanmedian(Idiff_normhm), np.sqrt(np.nanvar(Idiff_normhm))))
        print("Max Difference: %.2f  +- %.2f "%(np.nanmedian(Idiff), np.sqrt(np.nanvar(Idiff))))
        print("Time Taken: %.2f min"%((time.time()-start_time)/60))
        Idiff_lst.append(Idiff[np.logical_not(np.isnan(Idiff))])
        Idiff_norm_lst.append(Idiff_normhm[np.logical_not(np.isnan(Idiff_normhm))])
        Idiff_norm2_lst.append(Idiff_normImxy[np.logical_not(np.isnan(Idiff_normImxy))])
        Iqmxy_numerical_lst.append(Idiff[np.logical_not(np.isnan(Iqmxy_numerical))])
        hm_lst.append(Idiff[np.logical_not(np.isnan(hm))])


    print("###################### Function Pair %d Done !!!! #############################"%(f+1)) 
    Idiff_func_lst.append(np.concatenate(Idiff_lst, axis=0))
    Idiff_norm_func_lst.append(np.concatenate(Idiff_norm_lst, axis=0))
    Idiff_norm2_func_lst.append(np.concatenate(Idiff_norm2_lst, axis=0))
    Iqmxy_numerical_func_lst.append(np.concatenate(Iqmxy_numerical_lst, axis=0))
    hm_func_lst.append(np.concatenate(hm_lst, axis=0))

savecwd_plots = os.path.join(savecwd, 'Plots')
if not os.path.exists(savecwd_plots):
    os.makedirs(savecwd_plots)

plt.boxplot(Idiff_func_lst, showfliers=False)
plt.title('Analytical Vs Numerical Comparison', fontsize=19)
plt.xlabel('Different Pairs of Functions', fontsize=19)
plt.ylabel('Difference in estimates\n (in bits)', fontsize=19)
plt.xticks(fontsize=16)
plt.yticks(fontsize=19)
plt.tight_layout()
plt.savefig(os.path.join(savecwd_plots,"Idiff.png"))
plt.show()
plt.boxplot(Idiff_norm_func_lst, showfliers=False)
plt.title('Analytical Vs Numerical Comparison', fontsize=19)
plt.xlabel('Different Pairs of Functions', fontsize=19)
plt.ylabel('Difference in estimates\n (normalized in %)', fontsize=19)
plt.xticks(fontsize=16)
plt.yticks(fontsize=19)
plt.tight_layout()
plt.savefig(os.path.join(savecwd_plots,"Idiff_norm.png"))
plt.show()
plt.boxplot(Idiff_norm2_func_lst, showfliers=False)
plt.title('Analytical Vs Numerical Comparison', fontsize=19)
plt.xlabel('Different Pairs of Functions', fontsize=19)
plt.ylabel('Difference in estimates\n (normalized in %)', fontsize=19)
plt.xticks(fontsize=16)
plt.yticks(fontsize=19)
plt.tight_layout()
plt.savefig(os.path.join(savecwd_plots,"Idiff_norm2.png"))
plt.show()
plt.boxplot(Iqmxy_numerical_func_lst, showfliers=False)
plt.title('Iqmxy-numerical', fontsize=19)
plt.xlabel('Different Pairs of Functions', fontsize=19)
plt.ylabel('IqMXY-Numerical (in bits)', fontsize=19)
plt.xticks(fontsize=16)
plt.yticks(fontsize=19)
plt.tight_layout()
plt.savefig(os.path.join(savecwd_plots,"Iqmxy_numerical.png"))
plt.show()
plt.boxplot(hm_func_lst, showfliers=False)
plt.title('H(M)', fontsize=19)
plt.xlabel('Different Pairs of Functions', fontsize=19)
plt.ylabel('H(M) (in bits)', fontsize=19)
plt.xticks(fontsize=16)
plt.yticks(fontsize=19)
plt.tight_layout()
plt.savefig(os.path.join(savecwd_plots,"hm.png"))
plt.show()
