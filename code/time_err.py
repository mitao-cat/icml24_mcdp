import numpy as np
import pickle
import os
import argparse
import time
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from statsmodels.distributions.empirical_distribution import ECDF


def MCDP_exact(y_pred, y_gt, s, epsilon=0.01):
    start_time = time.time()
    y_pred, y_gt, s = y_pred.ravel(), y_gt.ravel(), s.ravel()
    y_pre_1, y_pre_0 = y_pred[s==1], y_pred[s==0]
    ecdf0, ecdf1 = ECDF(y_pre_0), ECDF(y_pre_1)
    if epsilon == 0:
        mcdp = np.max(np.abs(ecdf0(y_pred) - ecdf1(y_pred)))*100
    else:
        y_pred = np.r_[0,np.sort(y_pred),1]
        mcdp = np.min(y_pred[y_pred <= epsilon])
        y_arr = y_pred[y_pred <= 1-epsilon].reshape((-1,1)) # yi,i\in\mathcal{I}
        delta_ecdf = np.r_[1.1, np.abs(ecdf0(y_pred) - ecdf1(y_pred))]

        selected_indices = np.where(y_arr <= y_pred,1,0) * np.where(y_arr+2*epsilon >= y_pred, 1, 0)
        selected_indices = selected_indices.reshape(-1) * np.tile(np.arange(len(y_pred))+1,len(y_arr))
        selected_indices = selected_indices.reshape((len(y_arr),len(y_pred)))
        delta_ecdf = delta_ecdf[selected_indices]
        mcdp = max(mcdp, np.max(np.min(delta_ecdf,axis=1)))*100
    end_time = time.time()
    return end_time-start_time, mcdp


def MCDP_approximate(y_pred, y_gt, s, epsilon, K=1):
    assert epsilon>0
    start_time = time.time()

    y_pred, y_gt, s = y_pred.ravel(), y_gt.ravel(), s.ravel()
    y_pre_1, y_pre_0 = y_pred[s==1], y_pred[s==0]
    ecdf0, ecdf1 = ECDF(y_pre_0), ECDF(y_pre_1)
    delta = epsilon / K
    mcdp = np.min(np.abs(ecdf0(np.arange(K+1)*delta) - ecdf1(np.arange(K+1)*delta)))
    probing_points = np.arange(0,1,delta)
    delta_ecdf = np.abs(ecdf0(probing_points) - ecdf1(probing_points))
    indices = np.arange(2*K).reshape((-1,1)) + np.arange(1,np.ceil(1/delta)-2*K+1).astype(int)
    mcdp = max(mcdp, np.max(np.min(delta_ecdf[indices],axis=0)))*100

    end_time = time.time()
    return end_time-start_time, mcdp


def update_eps(epsilon_df, selected, method, lam):
    selected = selected[selected['lam']==lam]
    for eps in epsilon_list:
        ve_arr = selected[selected['eps']==eps]['ve'].values
        epsilon_df.append([method,lam,eps,np.mean(ve_arr),np.std(ve_arr)])


epsilon_list, K_list = [0.1,0.05,0.02,0.01], [1,32,128,512]
for dataset in ['adult','celeba-a']:
    root_dir, results_df = f'../results/{dataset}', []     # filename,eps,M,K,ratio,err
    for filename in os.listdir(root_dir):
        filepath = os.path.join(root_dir, filename)
        if not os.path.isfile(filepath) or not filename.endswith('.pkl'):
            continue
        setting_dict, _, _, y_gt, y_pred,s = pickle.load(open(filepath,'rb'))
        for epsilon in epsilon_list:
            te, ve = MCDP_exact(y_pred, y_gt, s, epsilon)
            for K in K_list:
                ta, va = MCDPT_approximate(y_pred, y_gt, s, epsilon, K=K)
                results_df.append([filename[:-4], epsilon, K, te/ta, 100*(va-ve)/ve, ta, te, va, ve])
                
    results_df = pd.DataFrame(results_df, columns=['log','eps','K','ratio','err','ta','te','va','ve'])
    for epsilon in epsilon_list:
        for K in K_list:
            selected = results_df[(results_df['eps']==epsilon) & (results_df['K']==K)][['ratio','err','ta']]
            ratios, errs, tas = selected['ratio'].values, selected['err'].values, selected['ta'].values
            results_df = results_df.append({'log':'mean', 'eps':epsilon, 'K':K, 'ratio':np.mean(ratios), 'err':np.mean(errs), 'ta':np.mean(tas), 'te':-1, 'va':-1, 've':-1}, ignore_index=True)
            results_df = results_df.append({'log':'std', 'eps':epsilon, 'K':K, 'ratio':np.std(ratios), 'err':np.std(errs), 'ta':np.std(tas), 'te':-1, 'va':-1, 've':-1}, ignore_index=True)
    results_df.to_csv(f'{root_dir}/results.csv',index=False)
