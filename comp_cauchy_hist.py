import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nbinom, betabinom
import time
from BROJA_2PID import *
import ray
import os
import logging
from ray.exceptions import RayActorError 
import warnings
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600
np.random.seed(1234)

func1_lst, func2_lst = [],[]
func1_lst, func2_lst = func1_lst+[lambda X:np.array(0.5*X**2)], func2_lst+[lambda X:X] ## 1
func1_lst, func2_lst = func1_lst+[lambda X: np.array(0.4*0.25*X**3)], func2_lst+[lambda X: np.array(0.25*X**2)] ## 2
func1_lst, func2_lst = func1_lst+[lambda X:4*X], func2_lst+[lambda X:X*np.sin(X/16*np.pi)**2+2] ## 3
func1_lst, func2_lst = func1_lst+[lambda X: 5*X], func2_lst+[lambda X:10*np.abs(np.sin(X*np.pi/16))+2] ## 4
func1_lst, func2_lst = func1_lst+[lambda X: np.array(0.05*(X**2+X**3))], func2_lst+[lambda X:X] ## 5
func1_lst, func2_lst = func1_lst+[lambda X: 7*X], func2_lst+[lambda X:np.array(0.5*X**2*np.cos(X/36*np.pi)**2+2)] ## 6
func1_lst, func2_lst = func1_lst+[lambda X: np.array((X-np.cos(X*np.pi/8)))], func2_lst+[lambda X:np.array(2*np.sin(X*np.pi/16))+2] ## 7 
func1_lst, func2_lst = func1_lst+[lambda X: 2*X], func2_lst+[lambda X:np.array(0.004*X*np.exp(X/2))+9] ## 8
func1_lst, func2_lst = func1_lst+[lambda X: np.array(0.5*(3*X**2-2*X))], func2_lst+[lambda X:np.array(X+np.log(X))] ## 9
func1_lst, func2_lst = func1_lst+[lambda X: np.array(18*np.abs(np.sin(np.pi*X/18)))], func2_lst+[lambda X:np.array(5*np.sqrt(X))] ## 10

savecwd_NB = os.path.join(os.getcwd(),'NB-PID-Results')
savecwd_binom = os.path.join(os.getcwd(),'Binom-PID-Results')

Idiff_func_lst, Idiff_norm_func_lst, Idiff_norm2_func_lst= [], [], []
Iqmxy_numerical_func_lst, hm_func_lst = [], []
LOAD_DATA_FLAG = True
relative_median_lst_NB, relative_median_lst_binom = [],[]
idx_curr = 2
UIx_func_lst, UIy_func_lst = [], []
for f in range(len(func1_lst)):
    start_time = time.time()
    
    Idiff_lst = []
    Iqmxy_numerical_lst, hm_lst = [], []

    for dim_M in [2,4]:    
        savecwd_rawdata = os.path.join(savecwd_NB, 'RawData')
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
    relative_median_lst_NB.append(np.median(Idiff_func_lst[-1])/np.median(Iqmxy_numerical_func_lst[-1])*100)

    Idiff_lst = []
    Iqmxy_numerical_lst, hm_lst = [], []
    for dim_M in [2,4]:    
        savecwd_rawdata = os.path.join(savecwd_binom, 'RawData')
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
    relative_median_lst_binom.append(np.median(Idiff_func_lst[-1])/np.median(Iqmxy_numerical_func_lst[-1])*100)


####### Plotting Fig 1c ###########
plt.bar(np.arange(len(func1_lst)).astype(np.int32)+1,relative_median_lst_binom, width=0.4,color='C0', label='Binomial')
plt.bar(np.arange(len(func1_lst)).astype(np.int32)+1+0.4,relative_median_lst_NB, color='C1', width=0.4, label='Neg. Binomial')
plt.legend(fontsize=20)
plt.hlines(y=4, xmin=0.5, xmax=10.5, linestyle='--', color='black', linewidth=2)
plt.hlines(y=1, xmin=0.5, xmax=10.5, linestyle='--', color='black', linewidth=2)
plt.text(x=1,y=4.1, s="4%", fontsize=20, color='black')
plt.text(x=1,y=1.1, s="1%", fontsize=20, color='black')
plt.xlabel('Functions Pair Index', fontsize=21)
plt.ylabel('% Difference in Analytical MI\n  w.r.t. Numerical MI', fontsize=21)
plt.xticks([1,5,10], ['1','5','10'],fontsize=23)
plt.yticks(fontsize=23)
plt.ylim(ymax=5)
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(),"Idiff_bar_binom-neg.png"))
plt.close()


