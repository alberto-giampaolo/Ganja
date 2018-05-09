import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ganja.plotting as plotting
from scipy import stats
import matplotlib

def ks_test(data1,data2,bins):
    data1, data2 = map(np.asarray, (data1, data2))
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    n1 = len(data1)
    n2 = len(data2)
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    
    cdf1 = np.searchsorted(data1,bins,side='right')/(1.0*n1)
    cdf2 = (np.searchsorted(data2,bins,side='right'))/(1.0*n2)
    d = np.max(np.absolute(cdf1-cdf2))
    # Note: d absolute not signed distance
    en = np.sqrt(n1*n2/float(n1+n2))
    try:
        prob = distributions.kstwobign.sf((en + 0.12 + 0.11 / en) * d)
    except:
        prob = 1.0
    return d, prob

def ks_percentile_sum(var,ep_list,epoch,bins): # For given var and epoch, return sum of ks values at different percentiles
    x = ep_list[epoch]
    ks_values = []
    pt = x['pt']
    dx5 = np.percentile(pt,np.linspace(0,100,5))   # pt percentile values
    for i in np.arange(1,len(dx5)):                # pt percentile loop
        upper, lower = dx5[i], dx5[i-1]
        var_pred = x['pred_'+var][(pt>=lower)&(pt<upper)]
        var_reco = x['reco_'+var][(pt>=lower)&(pt<upper)]
    #    ks_values += [stats.ks_2samp(var_pred,var_reco)[0]]
    ks_values += [ks_test(var_pred,var_reco,bins)[0]]
    return sum(ks_values)

def minimizing_epoch(var,epoch_list,bins):    # Epoch that minimizes the ks score for given variable
    min_ep, min_sum = None, None
    for ep in range(len(epoch_list)):
        ks_psum = ks_percentile_sum(var,epoch_list, ep,bins)
        if (min_sum==None or ks_psum < min_sum): min_ep, min_sum = ep, ks_psum
    return min_ep

def minimizing_epoch_sum(var_list,epoch_list,bins):   # Epoch that minimizes the sum of ks scores for given variables
    min_ep, min_sum = None, None
    for ep in range(len(epoch_list)):
        ks_var_sum = sum([ks_percentile_sum(v,epoch_list,ep,bins) for v in var_list])
        if (min_sum==None or ks_var_sum < min_sum): min_ep, min_sum = ep, ks_var_sum
    return min_ep

def plot_ks_sum(var_list,ep_list,bins):
    ks_sum = []
    ks_sum_all = []
    for e in range(len(ep_list)): 
        vsum = [ks_percentile_sum(v,ep_list,e,bins) for v in var_list]
        ks_sum += [sum(vsum)]
        ks_sum_all.append(vsum)
    ks_sum_all = np.array(ks_sum_all )
    plt.plot(range(len(ep_list)),ks_sum)
    plt.plot(range(len(ep_list)),ks_sum_all)
    plt.legend(["all"]+list(range(len(ep_list))))
    plt.xlabel("Training Epochs")
    plt.title("Ks value during training, summed over percentiles, and variables")
    plt.show()

def plot_ks(var,ep_list,pct=5):                          #plot ks value of a variable, for different epochs and pt bins
    for i in np.arange(1,pct+1):         #decile loop
        ksx=[]
        for x in ep_list:
            pt= x['pt']
            dx0 = np.percentile(pt,np.linspace(0,100,pct+1))   #decile pt values
            upper, lower = dx0[i], dx0[i-1]
            ptD_r_ring = x['reco_'+var][(pt<upper)&(pt>=lower)]
            ksx+=[stats.ks_2samp(ptD_r_ring,x['pred_'+var][(pt>=lower)&(pt<upper)])[0]]
        
        plt.plot(range(len(ep_list)),ksx,label=str(i*2-2)+'0 - '+str(i*2)+'0th percentile')
    

    plt.xlabel("Training Epochs")
    plt.title("Ks value of "+var+" distribution during training, by pt decile bins")
    plt.legend()
    plt.show()