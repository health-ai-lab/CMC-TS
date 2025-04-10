import numpy as np
import pandas as pd
from scipy.stats import zscore, norm
import sys
#import progressbar as pb
from joblib import Parallel, delayed
from numba import jit
import glob as glob
import os
import pickle
import multiprocessing

from functools import partial

'''Code to run the base causal inference methods'''
'''This code can be swapped with an alterantive method as long as the returned results is the format CMC expects'''


def generate_hypothesis_w(causes,effects,w):
    Hypo_list = []
    for effect in effects:
        for cause in causes:
            if effect in causes: #added for NICU data
                Hypo_list.append((cause,effect,w[0],w[1]))

    return Hypo_list

def generate_hypothesis(causes,effects,W):
    Hypo_list = []
    for effect in effects:
        for cause in causes:
            for w in W:
                #if cause != effect:
                Hypo_list.append((cause,effect,w[0],w[1]))

    return Hypo_list

def setup_prima_old(hypothesis,df):
    c,e,ws,we = hypothesis
    dataset_ids = np.asarray(df["dID"].unique())

    ind = []
    for id in dataset_ids:
        c_val = np.asarray(df[["timestamp",c,"dID","pID"]].copy())
        c_val = c_val[c_val[:,2] == id] #get only info associated with id
        e_val = np.asarray(df[["timestamp",e,"dID","pID"]].copy())
        e_val = e_val[e_val[:,2] == id]

        timestamps = np.asarray(df[["timestamp","dID"]].copy())
        timestamps = timestamps[timestamps[:,1] == id]
        #within a dataset/population, we have different patients with seperate pIDs
        sub_dataset_df = df[df["dID"] == id].copy(deep = True)
        sub_dataset_ids = np.asarray(sub_dataset_df["pID"].unique())

        P_e = np.nansum(e_val[:,1])/len(timestamps) #equivalnent to E[e]
        num,denum,_ = get_c_and_e(c_val,e_val,ws,we,sub_dataset_ids)
        P_e_c = 0 if denum == 0 else num/denum

        if P_e_c > P_e:
            ind.append(1)
            #print(f"P[e]: {P_e}, P[e|c]: {P_e_c}")
        else:
            ind.append(0)
    
    #print(f"rela: {hypothesis}, res: {ind}")
    if np.sum(ind) == 0: #if all datasets produce 0 then not prima facie for any dataset
        return ("no", (c,e,ws,we))
    else:
        #if prima facie for at least one dataset, then we do not discard it
        return ("yes",(c,e,ws,we))
    
def setup_prima(hypothesis,df):
    c,e,ws,we = hypothesis
    temp_ids = np.asarray(df["dID"].unique())
    P_e_c, P_e, w_e = {},{},{}
    
    dataset_ids = []
    for id in temp_ids:
        sub_df = df[df["dID"] == id].copy(deep=True)
        if sub_df[c].isna().sum() != len(sub_df):
            dataset_ids.append(id)

    ind = []
    for id in dataset_ids:
        c_val = np.asarray(df[["timestamp",c,"dID","pID"]].copy())
        c_val = c_val[c_val[:,2] == id] #get only info associated with id
        e_val = np.asarray(df[["timestamp",e,"dID","pID"]].copy())
        e_val = e_val[e_val[:,2] == id]

        timestamps = np.asarray(df[["timestamp","dID"]].copy())
        timestamps = timestamps[timestamps[:,1] == id]
        #within a dataset/population, we have different patients with seperate pIDs
        sub_dataset_df = df[df["dID"] == id].copy(deep = True)
        sub_dataset_ids = np.asarray(sub_dataset_df["pID"].unique())

        P_e[id] = np.nansum(e_val[:,1])/np.sum(~np.isnan(e_val[:,1])) #equivalnent to E[e]
        num,denum,_ = get_c_and_e(c_val,e_val,ws,we,sub_dataset_ids)
        P_e_c[id] = 0 if denum == 0 else num/denum
        w_e[id] = 0.001 if np.sum(~np.isnan(e_val[:,1])) == 0 else np.sum(~np.isnan(e_val[:,1]))/len(timestamps)
        
    weights = get_weights(hypothesis,df,dataset_ids)
    P_e_c = sum(P_e_c[dataset_id] * weights[dataset_id] for dataset_id in dataset_ids)/sum(weights[dataset_id] for dataset_id in dataset_ids)
    #weights = get_weights(hypothesis,df,dataset_ids,e_only = True)
    P_e = sum(P_e[dataset_id] * w_e[dataset_id] for dataset_id in dataset_ids)/sum(w_e[dataset_id] for dataset_id in dataset_ids)
    
    if np.round(P_e_c,1) > np.round(P_e,1):
        return ("yes",(c,e,ws,we))
    else:
        return ("no", (c,e,ws,we))
        
    
def test_for_prima_facie(hypotheses,df,prima_facie_history = []):
    prima_facie_for_effects,to_run,result_prev = {},[],[]

    for hypothesis in hypotheses:
        c,e,ws,we = hypothesis
        if (c,e,ws,we) in prima_facie_history:
            result_prev.append(("yes", (c,e,ws,we)))
        else:
            to_run.append(hypothesis)

    print(f"Original number is {len(hypotheses)}, saved time and only running {len(to_run)}")
    results = Parallel(n_jobs=150)(delayed(setup_prima)(hypothesis,df) for hypothesis in to_run)
    results = results + result_prev

    for result in results:
        ind, hypothesis = result
        if ind == "yes":
            if hypothesis[1] in prima_facie_for_effects:
                prima_facie_for_effects[hypothesis[1]].append((hypothesis[0],hypothesis[2],hypothesis[3]))
            else:
                prima_facie_for_effects[hypothesis[1]] = [(hypothesis[0],hypothesis[2],hypothesis[3])]

            prima_facie_history.append(hypothesis)
        else:
            pass
            #print(f"Not prima facie: {hypothesis}")
    
    prima_facie_history = list(set(prima_facie_history))
    return prima_facie_for_effects,prima_facie_history

def test_for_prima_facie_w(hypotheses_W,df,prima_facie_history):
    prima_facie_for_effects, result_prev = {},[]

    to_run = []
    for w,hypotheses in hypotheses_W.items():
        for hypothesis in hypotheses:
            c,e,ws,we = hypothesis
            if (c,e,ws,we) in prima_facie_history:
                result_prev.append(("yes", (c,e,ws,we))) #since one dataset was previously prima facie, we dont need to recompute it. Helps save a lof of compute time
            else:
                to_run.append(hypothesis)

    print(f"Num of hypothesis: {len(to_run)}")
    results = Parallel(n_jobs=150)(delayed(setup_prima)(hypothesis,df) for hypothesis in to_run)
    results = results + result_prev

    for result in results:
        ind, hypothesis = result
        if ind == "yes":
            c,e,ws,we = hypothesis
            if (ws,we) in prima_facie_for_effects:
                if e in prima_facie_for_effects[(ws,we)]:
                    prima_facie_for_effects[(ws,we)][e].append((c,ws,we))
                else:
                    prima_facie_for_effects[(ws,we)][e] = [(c,ws,we)]
            else:
                #if not window, create it and add information for effect since its the first of this window
                prima_facie_for_effects[(ws,we)] = {}
                prima_facie_for_effects[(ws,we)][e] = [(c,ws,we)]

            prima_facie_history.append((c,e,ws,we))

        else:
            pass
            #print(f"Not prima facie: {hypothesis}")

    prima_facie_history = list(set(prima_facie_history))
    return prima_facie_for_effects,prima_facie_history

def compute_causal_significance(prima_facie_for_effects,df,epx):
    effects = list(prima_facie_for_effects.keys())
    '''
    e_avg = {}
    to_test,results = [],[]

    for effect in effects:
        prima_facie_causes = prima_facie_for_effects[effect]
        for hypothesis in prima_facie_causes:
            eval = (hypothesis,prima_facie_causes.copy(),effect)
            to_test.append(eval)
        
    results = Parallel(n_jobs=200)(delayed(setup)(test,df,epx) for test in to_test)
    for group in results:
        cause,effect,r,s,hypothesis_eavg,epx_run = group
        e_avg[(cause,effect,r,s)] = hypothesis_eavg
        epx = {**epx,**epx_run}
    '''

    all_hypothesis_tested, to_test, sole_causes, already_tested = [],[],[],[]
    e_avg,X_len = {},{}
    dataset_ids = list(df["dID"].unique())
    X_len = {id:{} for id in dataset_ids}
    
    for effect in effects:
        prima_facie_causes = prima_facie_for_effects[effect]
        for hypothesis in prima_facie_causes:
            cause,r,s = hypothesis
            
            X = prima_facie_causes.copy()
            #other_causes = [group for group in X if group[0] != cause]
            other_causes = [group for group in X if group != hypothesis]

            if len(other_causes) != 0:
                e_avg[(cause,effect,r,s)] = {}
                #X_len[(cause,effect,r,s)] = len(other_causes)
                all_hypothesis_tested.append((cause,effect,r,s))
                

                for id in dataset_ids:
                    X_len[id][(cause,effect,r,s)] = 0
                    e_avg[(cause,effect,r,s)][id] = 0
                    sub_df = df[df["dID"] == id].copy(deep=True)
                    
                    if sub_df[cause].isna().sum() != len(sub_df): #continue if cause is measured
                        for x in other_causes:
                            if sub_df[x[0]].isna().sum() != len(sub_df):
                                X_len[id][(cause,effect,r,s)] += 1 #doing the count of X for each dataset_id

                                if id in epx:
                                    if (effect,(cause,r,s),x) in epx[id]:
                                        #if it has been pre-computed in a previous round, we dont bother re-running it again
                                        v = epx[id][(effect,(cause,r,s),x)]
                                        already_tested.append((cause,effect,r,s,x,v,id))

                                    else:
                                        to_test.append((cause,effect,r,s,x,id))
                                        #if (effect,(cause,r,s),x) not in epx[id]:
                                        #    epx[id][(effect,(cause,r,s),x)] = 0
                                else:
                                    to_test.append((cause,effect,r,s,x,id))

            else:
                #e_avg[(cause,effect,r,s)] = "sole-cause"
                sole_causes.append((cause,effect,r,s))

    print(f"Compute. Numbers of combination to compute: {len(to_test)}, excluding: {len(already_tested)}")
    #results = Parallel(n_jobs=200)(delayed(setup_compute_epsilon_x_probability_difference)(test,df,epx) for test in to_test)

    pool = multiprocessing.Pool(processes=220)
    partial_work = partial(setup_compute_epsilon_x_probability_difference, df)
    results = pool.map(partial_work, to_test)
    pool.close()
    pool.join()

    results = results + already_tested

    
    for group in results:
        cause,effect,r,s,x,val,dataset_id = group
        e_avg[(cause,effect,r,s)][dataset_id] += val
        try:
            epx[dataset_id][(effect,(cause,r,s),x)] = val
        except:
            epx[dataset_id] = {}
            epx[dataset_id][(effect,(cause,r,s),x)] = val

    for key in all_hypothesis_tested:
        for dataset_id in dataset_ids:
            
            if X_len[dataset_id][key] == 0:
                e_avg[key][dataset_id] = 0
            else:
                e_avg[key][dataset_id] = e_avg[key][dataset_id] / X_len[dataset_id][key]

    final_e_avg = {} #aggregate eavgs acorss all dataset_ids
    weights = {}

    for key in all_hypothesis_tested:
        c,e,r,s = key
        weights = get_weights(key,df,dataset_ids)
        final_e_avg[key] = 0
        temp_res = sum(e_avg[key][dataset_id] * weights[dataset_id] for dataset_id in dataset_ids)/sum(weights[dataset_id] for dataset_id in dataset_ids)#len(dataset_ids)
        final_e_avg[key] = temp_res
        print("Eavgs:", end=' ')
        for dataset_id in dataset_ids:
            print(f"{dataset_id}: {e_avg[key][dataset_id]};", end=' ')
        print(f"all: {temp_res}")
        print("\n")

    return final_e_avg,epx

def compute_causal_significance_w(prima_facie_for_effects,df,epx):
    all_hypothesis_tested, to_test, sole_causes, already_tested = [],[],[],[]
    e_avg,X_len = {},{}
    dataset_ids = list(df["dID"].unique())
    
    

    for w,effects_info in prima_facie_for_effects.items():
        e_avg[w],X_len[w] = {},{}
        effects = list(effects_info.keys())

        for effect in effects:
            prima_facie_causes = effects_info[effect]
            for hypothesis in prima_facie_causes:
                cause,r,s = hypothesis

                X = prima_facie_causes.copy()
                other_causes = [group for group in X if group[0] != cause]

                if len(other_causes) != 0:
                    e_avg[w][(cause,effect,r,s)] = {}
                    X_len[w][(cause,effect,r,s)] = len(other_causes)
                    all_hypothesis_tested.append((cause,effect,r,s))

                    for id in dataset_ids:
                        e_avg[w][(cause,effect,r,s)][id] = 0
                        for x in other_causes:
                            if id in epx:
                                if (effect,(cause,r,s),x) in epx[id]:
                                    #if it has been pre-computed in a previous round, we dont bother re-running it again
                                    v = epx[id][(effect,(cause,r,s),x)]
                                    already_tested.append((cause,effect,r,s,x,v,id))
                                else:
                                    to_test.append((cause,effect,r,s,x,id))
                            else:
                                to_test.append((cause,effect,r,s,x,id))

                        '''
                        epx[id] = {} if id not in epx else epx[id] #add a new dict if id has not been seen before, else return the stored dict
                        for x in other_causes:
                            to_test.append((cause,effect,r,s,x,id))
                            #epx[id][(effect,(cause,r,s),x)] = 0
                        '''
                else:
                    sole_causes.append((cause,effect,r,s))

    print(f"Compute_w. Numbers of combination to compute: {len(to_test)}, excluding: {len(already_tested)}")
    print("\n")
    #results = Parallel(n_jobs=200)(delayed(setup_compute_epsilon_x_probability_difference)(test,df,epx) for test in to_test)
    pool = multiprocessing.Pool(processes=150)
    partial_work = partial(setup_compute_epsilon_x_probability_difference, df)
    results = pool.map(partial_work, to_test)
    pool.close()
    pool.join()

    results = results + already_tested

    for group in results:
        cause,effect,r,s,x,val,dataset_id = group
        e_avg[(r,s)][(cause,effect,r,s)][dataset_id] += val

        try:
            epx[dataset_id][(effect,(cause,r,s),x)] = val
        except:
            epx[dataset_id] = {}
            epx[dataset_id][(effect,(cause,r,s),x)] = val

        #epx[dataset_id][(effect,(cause,r,s),x)] = val

    for key in all_hypothesis_tested:
        for dataset_id in dataset_ids:
            e_avg[(key[2],key[3])][key][dataset_id] /= X_len[(key[2],key[3])][key]

    final_e_avg = {} #aggregate eavgs acorss all dataset_ids
    for key in all_hypothesis_tested:
        c,e,r,s = key
        weights = get_weights(key,df,dataset_ids)

        if (r,s) not in final_e_avg:
            final_e_avg[(r,s)] = {}

        final_e_avg[(r,s)][key] = 0
        temp_res = sum(e_avg[(r,s)][key][dataset_id] * weights[dataset_id] for dataset_id in dataset_ids)/sum(weights[dataset_id] for dataset_id in dataset_ids)
        final_e_avg[(r,s)][key] = temp_res
        print("Eavgs:", end=' ')
        for dataset_id in dataset_ids:
            print(f"{dataset_id}: {e_avg[(r,s)][key][dataset_id]};", end=' ')
        print(f"all: {temp_res}")
        print("\n")


    return final_e_avg,epx

def get_weights(info,df,dataset_ids,e_only = False):
    #Currently, i consider the weight (or confidence proxy) of each relationship to be the combination
    #of its sample size and edge probability

    #sample size - tells us how often we actually have the data to evaulate the causal relationship
    #edge probabaility - tells us how often the causal relatioship occurs in the data

    #currently, i take the average of both these quantitites to get the weight of a causal relationship for each id
    c,e,r,s = info
    measure_of_missingness = {}
    per_dataset_id_weight = {}
    print(f"Relation to re-weight: {info}")

    for dataset_id in dataset_ids:
        sub_dataset_df = df[df["dID"] == dataset_id].copy(deep = True)
        sub_dataset_ids = np.asarray(sub_dataset_df["pID"].unique())

        cause = np.asarray(df[["timestamp",c,"dID","pID"]].copy())
        c_vals = cause[cause[:,2] == dataset_id]
        effect = np.asarray(df[["timestamp",e,"dID","pID"]].copy())
        e_vals = effect[effect[:,2] == dataset_id]

        '''Step 1: get sample size'''
        e_count = 0
        len_c_id = 0

        for id in sub_dataset_ids:
            a,b = setup_weights_compute(id,c_vals,e_vals,r,s)
            len_c_id += a
            e_count += b
        #use when we have a lot of sub_dataset_ids
        '''
        results = Parallel(n_jobs=200)(delayed(setup_weights_compute)(id,c_vals,e_vals,r,s) for id in sub_dataset_ids)
        for a,b in results:
            len_c_id += a
            e_count += b
        '''

        dataset_id_sample_size = 0 if len_c_id == 0 else float(e_count/len_c_id)

        '''Step 2: get edge probability'''
        #num,denum = get_c_and_e(c_vals,e_vals,r,s,sub_dataset_ids)
        #dataset_id_edge_prob = float(num/denum)

        '''Testing with prevelance of cause'''
        #prevalence_of_cause = 0 if len(c_vals) == 0 else np.sum(~np.isnan(c_vals[:,1]))/len(c_vals)
        prevalence_of_cause = 0 if dataset_id_sample_size == 0 else np.sum(~np.isnan(c_vals[:,1]))/len(c_vals)


        '''Step 3: average them'''
        if e_only == True:
            temp_res = 0.001 if prevalence_of_cause == 0 else prevalence_of_cause
        else:
            #temp_res = np.average([dataset_id_edge_prob,dataset_id_sample_size], weights = [0.70,0.30])
            if (prevalence_of_cause == 0) and (dataset_id_sample_size == 0):
                temp_res = 0.001
            else:
                temp_res = np.average([prevalence_of_cause,dataset_id_sample_size], weights = [1,1])
             
        measure_of_missingness[dataset_id] = temp_res
        #print(f"weights for dataset {dataset_id} are s-size: {dataset_id_sample_size}, edge-prob: {dataset_id_edge_prob}. Average is {temp_res}")
        print(f"weights for dataset {dataset_id} are s-size: {dataset_id_sample_size}, caue-prevelance: {prevalence_of_cause}. Average is {temp_res}")


    #'''Step 4: scale weights?'''
    min_measure_of_missingness = min(measure_of_missingness.values())
    max_measure_of_missingness = max(measure_of_missingness.values())

    for key,val in measure_of_missingness.items():
        if max_measure_of_missingness != min_measure_of_missingness:
            new_weight = (val - min_measure_of_missingness)/(max_measure_of_missingness - min_measure_of_missingness)
        else:
            new_weight = 1

        per_dataset_id_weight[key] = new_weight
    #weight_total = sum(per_dataset_id_weight[dataset_id] for dataset_id in dataset_ids)
    #n_weights = len(per_dataset_id_weight)
    #for dataset_id in dataset_ids:
    #    per_dataset_id_weight[dataset_id] = per_dataset_id_weight[dataset_id]/weight_total * n_weights
    #print(f"hypo: {info}, s-size: {dataset_id_sample_size}, edge-prob: {dataset_id_edge_prob}")
    return per_dataset_id_weight

@jit(nopython=True)
def setup_weights_compute(id,c_vals,e_vals,r,s):
    e_count = 0
    c_ts = c_vals[c_vals[:, 3] == id] #pID column is index 3
    e_ts = e_vals[e_vals[:, 3] == id]
    c_id = c_ts[c_ts[:, 1] > 0, 0] #get timestamps where c is greater than 0 (cause could be probabilsitic)
    
    for t in c_id:
        filtered_ek_el = e_ts[(e_ts[:, 0] >= r+t) & (e_ts[:, 0] <= s+t)]
        filtered_ek_el = filtered_ek_el[:,1]

        if np.all(np.isnan(filtered_ek_el)):
            pass
        else:
            e_count += 1

    return (len(c_id),e_count)

def setup(info,df,epx):
    hypothesis,prima_facie_causes,effect = info
    cause,r,s = hypothesis
    X = prima_facie_causes.copy()
    other_causes = [group for group in X if group[0] != cause]

    sum_epx = 0
    epx_new = {}

    for x in other_causes:
        if (effect,hypothesis,x) in epx:
            #if it has been pre-computed in a previous round, re-use value
            result = epx[(effect,hypothesis,x)]
        else:
            result = compute_epsilon_x_probability_difference(cause,effect,x,df,r,s)
        sum_epx = sum_epx + result
        epx_new[(effect,hypothesis,x)] = result
    return (cause,effect,r,s,sum_epx/len(X),epx_new)

def setup_compute_epsilon_x_probability_difference(df,info):#,df,epx):
    cause,effect,r,s,x,dataset_id = info
    #print(f"{dataset_id},{cause}, {effect}, {r},{s}, {x}")
    val = compute_epsilon_x_probability_difference(cause,effect,x,df,r,s,dataset_id)
    
    '''
    try:
        if (effect,(cause,r,s),x) in epx[dataset_id]:
            #if it has been pre-computed in a previous round, re-use value
            val = epx[dataset_id][(effect,(cause,r,s),x)]
        else:
            val = compute_epsilon_x_probability_difference(cause,effect,x,df,r,s,dataset_id)
    except:
        val = compute_epsilon_x_probability_difference(cause,effect,x,df,r,s,dataset_id)
    '''
    return (cause,effect,r,s,x,val,dataset_id)

@jit(nopython=True)
def get_overlap(r,s,r1,s1):
    if s < r1 or s1 < r:
        return (0,0)
    
    #print(r)
    #print(r1)
    #print("overlap")
    r = np.asarray([r,r1])
    s = np.asarray([s,s1])
    overlap_start = np.max(r)
    overlap_end = np.min(s)

    return (overlap_start,overlap_end)

def compute_epsilon_x_probability_difference(c,e,x,df,r,s,dataset_id):
    x,r1,s1 = x
    cause = np.asarray(df[["timestamp",c,"dID","pID"]].copy())
    c_vals = cause[cause[:,2] == dataset_id] #get only info associated with dataset_id
    effect = np.asarray(df[["timestamp",e,"dID","pID"]].copy())
    e_vals = effect[effect[:,2] == dataset_id]
    other_X = np.asarray(df[["timestamp",x,'dID',"pID"]].copy())
    x_vals = other_X[other_X[:,2] == dataset_id]

    #get the pid associated with dataset_id
    sub_dataset_df = df[df["dID"] == dataset_id].copy(deep = True)
    sub_dataset_ids = np.asarray(sub_dataset_df["pID"].unique())

    num,denum = get_c_and_x(c_vals,x_vals,r,s,r1,s1,e_vals,sub_dataset_ids)
    P_e_given_c_and_x = 0 if denum == 0 else num/denum

    num,denum = get_notc_and_x(c_vals,x_vals,r,s,r1,s1,e_vals,sub_dataset_ids)
    P_e_given_notc_and_x = 0 if denum == 0 else num/denum

    epx_c_e = P_e_given_c_and_x - P_e_given_notc_and_x
    return epx_c_e

@jit(nopython=True)
def get_c_and_e(C_ts,E_ts,ws,we,sub_dataset_ids):
    '''Function to count the number of time an event happens in a cause time windows'''
    '''Inputs:
            c_idxs: times when cause happens
            e_ts: effect time series
            ws,we are the time window
        Outputs:
            count_c_and_e : number of time e happens in c time window
    '''

    num,denum,sub_p_e = 0,0,0
    #since dataset could be combination of different patients or datasets, we use the sub_dataset_ids to segment computation
    #this is useful since different patients are combined for timestamp will not be consistent
    for id in sub_dataset_ids:
        c_ts = C_ts[C_ts[:, 3] == id] #pID column is index 3
        e_ts = E_ts[E_ts[:, 3] == id]

        timestamps = c_ts[c_ts[:, 1] > 0, 0] #valid timestamp where cause value is > 0

        for t in timestamps:
            p_ct = c_ts[np.where(c_ts[:,0] == t)][0][1]
            filtered_ek_el = e_ts[(e_ts[:, 0] >= ws+t) & (e_ts[:, 0] <= we+t)]
            filtered_ek_el = filtered_ek_el[:,1]
            filtered_ek_el = filtered_ek_el[~np.isnan(filtered_ek_el)]
            p_ek_el = 1 - np.prod(1-filtered_ek_el)

            r = np.asarray([p_ct,p_ek_el])
            top = np.prod(r)
            bot = p_ct

            r = np.asarray([num,top])
            num = np.nansum(r)
            r = np.asarray([denum,bot])
            denum = np.nansum(r)

        temp = np.nansum(e_ts[:,1])/np.sum(~np.isnan(e_ts[:,1]))
        temp = np.asarray([sub_p_e,temp])
        sub_p_e = np.nansum(temp)
    
    p_e = sub_p_e/len(sub_dataset_ids)
    return num,denum,p_e
        
@jit(nopython=True)
def get_c_and_x(C_ts,X_ts,r,s,r1,s1,E_ts,sub_dataset_ids):

    num,denum = 0,0
    for id in sub_dataset_ids:
        
        c_ts = C_ts[C_ts[:, 3] == id] #pID column is index 3
        e_ts = E_ts[E_ts[:, 3] == id]
        x_ts = X_ts[X_ts[:, 3] == id]
        
        timestamps = c_ts[c_ts[:, 1] > 0, 0]

        for t in timestamps:
            p_ct = c_ts[np.where(c_ts[:,0] == t)][0][1]

            #check if p_ct is nan
            if np.isnan(p_ct):
                continue 

            filtered_xi_xj = x_ts[(x_ts[:, 0] >= r+t-s1) & (x_ts[:, 0] <= s+t-r1)]
            filtered_xi_xj = filtered_xi_xj[:,1]

            if np.isnan(filtered_xi_xj).all():
                continue 

            #valid_values = ~np.isnan(filtered_xi_xj[:,1]) #get bool for non-nan values
            filtered_xi_xj = filtered_xi_xj[~np.isnan(filtered_xi_xj)]
            p_xi_xj = 1 - np.prod(1-filtered_xi_xj)

            filtered_xi_xj = x_ts[(x_ts[:, 0] >= r+t-s1) & (x_ts[:, 0] <= s+t-r1)]
            temp = np.where(filtered_xi_xj[:,1] > 0)[0] #where x in [i,j] is non-zero
            
            if len(temp) == 0:
                continue 
            
            x_s = filtered_xi_xj[temp[0],0] #first time point where x in [i,j] is non-zero
            x_e = filtered_xi_xj[temp[-1],0] #last
            k = np.asarray([x_s + r1, r + t])
            l = np.asarray([x_e + s1, s + t])
            k = np.amax(k)
            l = np.amin(l)

            filtered_ek_el = e_ts[(e_ts[:, 0] >= k) & (e_ts[:, 0] <= l)]
            filtered_ek_el = filtered_ek_el[:,1]
            #valid_values = ~np.isnan(filtered_ek_el[:,1]) #get bool for non-nan values
            filtered_ek_el = filtered_ek_el[~np.isnan(filtered_ek_el)]
            p_ek_el = 1 - np.prod(1-filtered_ek_el)

            rr = np.asarray([p_ct,p_ek_el,p_xi_xj])
            top = np.prod(rr)
            rr = np.asarray([p_ct,p_xi_xj])
            bot = np.prod(rr)

            rr = np.asarray([num,top])
            num = np.nansum(rr)
            rr = np.asarray([bot,denum])
            denum = np.nansum(rr)

    return num,denum

@jit(nopython=True)
def get_notc_and_x(C_ts,X_ts,r,s,r1,s1,E_ts,sub_dataset_ids):

    num,denum = 0,0
    for id in sub_dataset_ids:
        
        c_ts = C_ts[C_ts[:, 3] == id] #pID column is index 3
        e_ts = E_ts[E_ts[:, 3] == id]
        x_ts = X_ts[X_ts[:, 3] == id]

        timestamps = x_ts[x_ts[:, 1] > 0, 0]

        for t in timestamps:
            p_xt = x_ts[np.where(x_ts[:,0] == t)][0][1]
            
            if np.isnan(p_xt):
                continue

            filtered_cg_ch = c_ts[(c_ts[:, 0] >= r1+t-s) & (c_ts[:, 0] <= s1+t-r)]
            filtered_cg_ch = filtered_cg_ch[:,1]

            if np.isnan(filtered_cg_ch).all():
                continue

            #valid_values = ~np.isnan(filtered_cg_ch[:,1]) #get bool for non-nan values
            filtered_cg_ch = filtered_cg_ch[~np.isnan(filtered_cg_ch)]
            p_cg_ch = np.prod(1-filtered_cg_ch)

            filtered_ek_el = e_ts[(e_ts[:, 0] >= t+r1) & (e_ts[:, 0] <= t+s1)]
            filtered_ek_el = filtered_ek_el[:,1]
            #valid_values = ~np.isnan(filtered_ek_el[:,1]) #get bool for non-nan values
            filtered_ek_el = filtered_ek_el[~np.isnan(filtered_ek_el)]
            p_ek_el = 1 - np.prod(1-filtered_ek_el)

            rr = np.asarray([p_cg_ch,p_xt,p_ek_el])
            top = np.prod(rr)
            rr = np.asarray([p_cg_ch,p_xt])
            bot = np.prod(rr)

            rr = np.asarray([num,top])
            num = np.nansum(rr)
            rr = np.asarray([bot,denum])
            denum = np.nansum(rr)

    return num,denum

def compute_pvalues(sigs,interval,d_id = None, data = None):
    zscores = zscore(list(sigs.values()), nan_policy='raise')
    #https://www.statology.org/p-value-from-z-score-by-hand/
    abszscores = [abs(z) for z in zscores]
    pvalues = {}
    
    zscores = zscores[np.isfinite(zscores)]
    #loc,scale = norm.fit(zscores) #newly added, see if ti makes a difference

    if data is None:
        for i, (key,val) in enumerate(sigs.items()):
            pval = 1.0 - norm.cdf(abszscores[i])
            pvalues[key] = [pval,val]
        return pvalues
    
    else:
        df = pd.DataFrame(columns = ["d_id","c-e","ws-we","epsilon-sign","p-value"])
        for i, (k,eavg) in enumerate(sigs.items()):
            pval = 1.0 - norm.cdf(abszscores[i])
            sign = 1 if eavg > 0 else -1

            #compute sample size weight of relationship
            #sample_size = get_sample_sizes(data,k[0],k[1],k[2],k[3])
            #compute estimated probability of the relationship
            #edge_prob = get_edge_probabilities(k[0],k[1],k[2],k[3],data)
            df.loc[len(df)] = [d_id,k[0] +"-"+ k[1], str(int(k[2]/interval)) +"-"+ str(int(k[3]/interval)),sign,pval]

        df.sort_values(by = ["c-e","p-value"],ascending=[True,True],ignore_index= True,inplace = True)
        #for real data, remove duplicates
        #df.drop_duplicates(subset=["c-e"], keep = "first", ignore_index = True, inplace = True)
        return df

def get_sample_sizes(df,c,e,ws,we):
    '''Function to compute the sample size weight between cause and effect'''
    #e has data in window/ #c occurs
    c_ts = np.asarray(df[["timestamp",c]].copy())
    e_ts = np.asarray(df[["timestamp",e]].copy())
    c_id = c_ts[c_ts[:, 1] > 0, 0] #get timestamps where c is greater than 0 (cause could be probabilsitic)
    e_count = 0

    for t in c_id:
        filtered_ek_el = e_ts[(e_ts[:, 0] >= ws+t) & (e_ts[:, 0] <= we+t)]
        filtered_ek_el = filtered_ek_el[:,1]

        if np.all(np.isnan(filtered_ek_el)):
            pass
        else:
            e_count = e_count + 1

    return float(e_count/len(c_id))

def get_edge_probabilities(c,e,ws,we,df,sgn):

    dataset_ids = np.asarray(df["dID"].unique())
    weights = get_weights((c,e,ws,we),df,dataset_ids)
    P_e_c,P_e,w_e = {},{},{}

    for id in dataset_ids:
        c_val = np.asarray(df[["timestamp",c,"dID","pID"]].copy())

        #if sgn == -1:
        #    print("Found negative cause - exiting")
        #    exit()
            #this gets the complement of the cause since the relationship is negative
        #    mask = c_val[:,1] == 1
        #    c_val[mask,1] = 0
        #    c_val[~mask,1] = 1

        c_val = c_val[c_val[:,2] == id] #get only info associated with id
        e_val = np.asarray(df[["timestamp",e,"dID","pID"]].copy())
        e_val = e_val[e_val[:,2] == id]

        timestamps = np.asarray(df[["timestamp","dID"]].copy())
        timestamps = timestamps[timestamps[:,1] == id]
        #within a dataset/population, we have different patients with seperate pIDs
        sub_dataset_df = df[df["dID"] == id].copy(deep = True)
        sub_dataset_ids = np.asarray(sub_dataset_df["pID"].unique())

        num,denum,_ = get_c_and_e(c_val,e_val,ws,we,sub_dataset_ids)
        P_e_c[id] = 0 if denum == 0 else num/denum
        P_e[id] = np.nansum(e_val[:,1])/np.sum(~np.isnan(e_val[:,1]))
        w_e[id] = 0.001 if np.sum(~np.isnan(e_val[:,1])) == 0 else np.sum(~np.isnan(e_val[:,1]))/len(timestamps)

    p_e_c = sum(P_e_c[dataset_id] * weights[dataset_id] for dataset_id in dataset_ids)/sum(weights[dataset_id] for dataset_id in dataset_ids)
    weights = get_weights((c,e,ws,we),df,dataset_ids,e_only = True)
    p_e = sum(P_e[dataset_id] * w_e[dataset_id] for dataset_id in dataset_ids)/sum(w_e[dataset_id] for dataset_id in dataset_ids)
    return p_e_c,p_e

def refine_eavg(hypothesis,e_new,prima_facie_causes,df,min_lag,max_lag,interval,epx):
    c,e,ws,we = hypothesis
    #print(f"Initial hypothesis---{hypothesis}----{e_new}")
    e_max = e_new
    X = prima_facie_causes.copy()
    X = [group for group in X if group[0] not in [c]]
    ind = True
    ran_hypothesis = []

    count = 0

    while (ind == True) and (count < 5):
        
        t = we-ws
        e_max = e_new
        #operation 1 ws = ws-[t/4], we = we + [t/4]
        temp_hypo = (c,e,max(int(ws-t/4),min_lag*interval),min(int(we+t/4),max_lag*interval))
        ran_hypothesis.append(temp_hypo)
        e_temp,epx = compute_causal_significance_refine(temp_hypo,df,X,epx)
        if abs(e_temp) > abs(e_new):
            current_max = e_temp
            current_hypo = temp_hypo
        else:
            current_max = e_new

        #operation 2 ws = ws, we = ws + [t/2]
        temp_hypo = (c,e,ws,int(ws+t/2))
        if temp_hypo not in ran_hypothesis:
            ran_hypothesis.append(temp_hypo)
            e_temp,epx = compute_causal_significance_refine(temp_hypo,df,X,epx)
            if abs(e_temp) > abs(current_max):
                current_max = e_temp
                current_hypo = temp_hypo

        #operation 3 ws = ws+[t/2], we = we
        temp_hypo = (c,e,int(ws+t/2),we)
        if temp_hypo not in ran_hypothesis:
            ran_hypothesis.append(temp_hypo)
            e_temp,epx = compute_causal_significance_refine(temp_hypo,df,X,epx)
            if abs(e_temp) > abs(current_max):
                current_max = e_temp
                current_hypo = temp_hypo

        #operation 4 ws = ws-1, we = we-1
        temp_hypo = (c,e,max(ws-(interval*min_lag),min_lag*interval),max(we-(interval*min_lag),min_lag*interval))
        if temp_hypo not in ran_hypothesis:
            ran_hypothesis.append(temp_hypo)
            e_temp,epx = compute_causal_significance_refine(temp_hypo,df,X,epx)
            if abs(e_temp) > abs(current_max):
                current_max = e_temp
                current_hypo = temp_hypo

        #operation 5 ws = ws+1, we = we+1
        temp_hypo = (c,e,min(ws+(interval*min_lag),max_lag*interval),min(we+(interval*min_lag),max_lag*interval))
        if temp_hypo not in ran_hypothesis:
            ran_hypothesis.append(temp_hypo)
            e_temp,epx = compute_causal_significance_refine(temp_hypo,df,X,epx)
            if abs(e_temp) > abs(current_max):
                current_max = e_temp
                current_hypo = temp_hypo

        #operation 6 ws = ws, we = we+1
        temp_hypo = (c,e,ws,min(we+(min_lag*interval),max_lag*interval))
        if temp_hypo not in ran_hypothesis:
            ran_hypothesis.append(temp_hypo)
            e_temp,epx = compute_causal_significance_refine(temp_hypo,df,X,epx)
            #print(f"window is [{temp_hypo[2],temp_hypo[3]}] with value {e_temp}")
            if abs(e_temp) > abs(current_max):
                #print(f"{temp_hypo}, old: {current_max}, new:{e_temp}")
                current_max = e_temp
                current_hypo = temp_hypo

        #operation 7 ws = ws+1, we = we
        temp_hypo = (c,e,min(ws+(min_lag*interval), min_lag*interval),we)
        if temp_hypo not in ran_hypothesis:
            ran_hypothesis.append(temp_hypo)
            e_temp,epx = compute_causal_significance_refine(temp_hypo,df,X,epx)
            #print(f"window is [{temp_hypo[2],temp_hypo[3]}] with value {e_temp}")
            if abs(e_temp) > abs(current_max):
                #print(f"{temp_hypo}, old: {current_max}, new:{e_temp}")
                current_max = e_temp
                current_hypo = temp_hypo

        e_new = current_max

        if abs(e_new) > abs(e_max): #i.e the causal significane has increased
            ws = current_hypo[2]
            we = current_hypo[3]
        else:
            ind = False

        count = count + 1
        print(f"count: {count}")

    #print(f"Refined Hypothesis---{(c,e,ws,we)}----{e_max}")
    return (c,e,ws,we), epx

def compute_causal_significance_refine(hypo,df,X,epx):

    c,e,r,s = hypo
    to_test,already_tested = [],[]
    temp_ids = list(df["dID"].unique())
    
    dataset_ids = []
    for id in temp_ids:
        sub_df = df[df["dID"] == id].copy(deep=True)
        if sub_df[c].isna().sum() != len(sub_df):
            dataset_ids.append(id)
    
    X_len = {dataset_id:0 for dataset_id in dataset_ids}
    
    for x in X:
        for dataset_id in dataset_ids:
            sub_df = df[df["dID"] == dataset_id].copy(deep=True)
            if sub_df[x[0]].isna().sum() != len(sub_df): #x is not all nan values in sub_df
                X_len[dataset_id] += 1
            
                if dataset_id in epx:
                    if (e,(c,r,s),x) in epx[dataset_id]:
                        #if it has been pre-computed in a previous round, we dont bother re-running it again
                        v = epx[dataset_id][(e,(c,r,s),x)]
                        already_tested.append((c,e,r,s,x,v,dataset_id))
                    else:
                        to_test.append((c,e,x,r,s,dataset_id))
                else:
                    to_test.append((c,e,x,r,s,dataset_id))
                
    print(f"Refinement. Numbers of combination to compute: {len(to_test)}, excluding: {len(already_tested)}")
    
    #results = Parallel(n_jobs=150)(delayed(setup_refine)(info) for info in to_test)
    pool = multiprocessing.Pool(processes=150)
    partial_work = partial(setup_refine, df)
    results = pool.map(partial_work, to_test)
    pool.close()
    pool.join()
    
    results = results + already_tested

    per_dataset_eavg = {}

    for group in results:
        c,e,r,s,x,epx_t,dataset_id = group
        if dataset_id not in per_dataset_eavg:
            per_dataset_eavg[dataset_id] = 0
        per_dataset_eavg[dataset_id] += epx_t
        epx[dataset_id][(e,(c,r,s),x)] = epx_t

    for dataset_id in dataset_ids:
        per_dataset_eavg[dataset_id] = 0 if X_len[dataset_id] == 0 else per_dataset_eavg[dataset_id]/X_len[dataset_id]

    #weights = {key: 1 for key in dataset_ids}
    weights = get_weights(hypo,df,dataset_ids)
    weigted_avg_eavg = sum(per_dataset_eavg[dataset_id] * weights[dataset_id] for dataset_id in dataset_ids)/sum(weights[dataset_id] for dataset_id in dataset_ids)

    return weigted_avg_eavg,epx

def setup_refine(df,info):
    c,e,x,r,s,dataset_id = info
    temp = compute_epsilon_x_probability_difference(c,e,x,df,r,s,dataset_id)
    return (c,e,r,s,x,temp,dataset_id)
    

def refine(W,stats_sig_across_windows,e_avg_across_windows,prima_facie_across_windows,sig_th,df,min_lag,max_lag,interval,Hypo_across_windows,effects,epx):

    '''1. find rela to refine. Pick rela less than sig and only one occurence of it'''
    uniq_rela = {}
    for w in W:
        if w in stats_sig_across_windows:
            relas = stats_sig_across_windows[w]
            for key,val in relas.items():
                if val[0] <= sig_th:
                    if (key[0],key[1]) in uniq_rela:
                        if val[0] < uniq_rela[(key[0],key[1])][4]:
                            uniq_rela[(key[0],key[1])] = (key[0],key[1],key[2],key[3],val[0])   
                    else:
                        uniq_rela[(key[0],key[1])] = (key[0],key[1],key[2],key[3],val[0])
                    
    uniq_rela = [val for _,val in uniq_rela.items()]
    to_refine = {w:{} for w in W}

    for rela in uniq_rela:
        c,e,ws,we,p = rela
        to_refine[(ws,we)].update({(c,e,ws,we):p})

    '''2. Refine found relations'''
    refined_relationships = []
    for w in W:
        if w in e_avg_across_windows:
            p_values_w = to_refine[w]
            e_avgs_w = e_avg_across_windows[w]
            prima_causes_w = prima_facie_across_windows[w]
            
            for rela,v in p_values_w.items():
                rela_new, epx = refine_eavg(rela,e_avgs_w[rela],prima_causes_w[rela[1]],df,min_lag,max_lag,interval,epx)
                refined_relationships.append(rela_new)

    '''3. get previous hypotheses'''
    all_hypo = []       
    for w,hypotheses in Hypo_across_windows.items():
        for hypothesis in hypotheses:
            all_hypo.append(hypothesis)

    '''4. Remove hypothesis with overlapping windows with refined relationships'''
    for rela in refined_relationships:
        c,e,ws,we = rela[0], rela[1], rela[2],rela[3]
        
        #Look at hypothesis with matching cause and effect
        selected_hypothesis = [data for data in all_hypo if (data[0],data[1]) == (c,e)]

        #for each hypothesis, check if they overlap with time window of rela, if so remove them
        for hypothesis in selected_hypothesis:
            ws_h,we_h = hypothesis[2],hypothesis[3]
            overlap_window = get_overlap(ws,we,ws_h,we_h)

            if overlap_window[0] + overlap_window[1] != 0: #if no overlap, overla_window is (0,0)
                if overlap_window[1] - overlap_window[0] != 0: #checks if overlap is on one timepoint e.g 5-15, 15-20
                    #print(f"Removing {hypothesis} as it overlaps with refined hypothesis {(c,e,ws,we)}")
                    all_hypo.remove(hypothesis)

    all_hypo = all_hypo + refined_relationships
    all_hypo = list(set(all_hypo))

    return all_hypo,epx

#for real data experiments with time window refinements 
def run_causal_discovery_pop(causes,effects,min_lag,max_lag,interval,df,level,data_id,log_path,sig,new_ids = None):

    #W = [(5,15),(15,30),(30,45),(45,60)]
    #W  = [(15,30),(30,45),(45,60)]
    W = [(1,5),(5,10),(10,15),(15,20),(20,25),(25,30),(30,35),(35,40),(45,50),(50,55),(55,60)]
    #W = [(1,10),(10,20),(20,30),(30,40),(40,50),(50,60)]
    W = [(x*interval,y*interval) for (x,y) in W]
    
    Hypo_across_windows,prima_facie_across_windows,e_avg_across_windows,stats_sig_across_windows,cmc_sk_log = {},{},{},{},{}

    if level == 0:
        if data_id == None:
            #here new_ids contains a list of ids to be loaded
            epx,prima_facie_history = {},[]
            for o_id in new_ids:
                if os.path.exists(log_path + "sk_runlog_" + str(o_id) + ".pickle"):
                    cmc_sk_log = pd.read_pickle(log_path + "sk_runlog_" + str(o_id) + ".pickle")
                    t_epx = cmc_sk_log["epx"]
                    t_prima_facie_history = cmc_sk_log["prima"]
                    epx.update(t_epx)
                    prima_facie_history.extend(t_prima_facie_history)
                    
            prima_facie_history = list(set(prima_facie_history))
                  
        else:
            epx,prima_facie_history = {},[]
                                       
        for w in W:
            Hypo_across_windows[w] = generate_hypothesis_w(causes,effects,w)

        prima_facie_across_windows,prima_facie_history = test_for_prima_facie_w(Hypo_across_windows,df,prima_facie_history)
        e_avg_across_windows,epx = compute_causal_significance_w(prima_facie_across_windows,df,epx)

        for w in W:
            if w in e_avg_across_windows:
                p_values = compute_pvalues(e_avg_across_windows[w],interval)
                stats_sig_across_windows[w] = p_values

    else:
        #print(log_path)
        #print(data_id)
        if os.path.exists(log_path + "sk_runlog_" + str(data_id) + ".pickle"):
            cmc_sk_log = pd.read_pickle(log_path + "sk_runlog_" + str(data_id) + ".pickle")
            epx = cmc_sk_log["epx"]
            prima_facie_history = cmc_sk_log["prima"]
            
            for w in W:
                Hypo_across_windows[w] = generate_hypothesis_w(causes,effects,w)

            prima_facie_across_windows, prima_facie_history = test_for_prima_facie_w(Hypo_across_windows,df,prima_facie_history)
            e_avg_across_windows,epx = compute_causal_significance_w(prima_facie_across_windows,df,epx)

            for w in W:
                if w in e_avg_across_windows:
                    p_values = compute_pvalues(e_avg_across_windows[w],interval)
                    stats_sig_across_windows[w] = p_values

    all_hypotheses,epx = refine(W,stats_sig_across_windows,e_avg_across_windows,prima_facie_across_windows,
                                        sig,df,min_lag,max_lag,interval,Hypo_across_windows,effects,epx)
    
    print("Done Refining")
    prima_facie_for_effects,prima_facie_history = test_for_prima_facie(all_hypotheses,df,prima_facie_history)
    print("Computing Final eavgs")
    e_avgs,epx= compute_causal_significance(prima_facie_for_effects,df,epx)
    print("Computed Final eavgs")
    results = compute_pvalues(e_avgs,interval,data_id,df)

    #save computed information
    cmc_sk_log = {"epx":epx, "prima":prima_facie_history}
    if level == 0:
        if data_id == None:
            with open(log_path + "sk_runlog_final_.pickle",'wb') as f:
                pickle.dump(cmc_sk_log,f,pickle.HIGHEST_PROTOCOL)
        else:   
            with open(log_path + "sk_runlog_" + str(data_id) + ".pickle",'wb') as f:
                pickle.dump(cmc_sk_log,f,pickle.HIGHEST_PROTOCOL)
    else:
        with open(log_path + "sk_runlog_" + str(new_ids) + ".pickle",'wb') as f:
            pickle.dump(cmc_sk_log,f,pickle.HIGHEST_PROTOCOL)

    return results

#for CMC simulated experiments with no refinement process
def run_causal_discovery_pop_sim(causes,effects,min_lag,max_lag,interval,df,level,data_id,log_path,sig,new_ids = None):
    W  = [(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10)]
    W = [(x*interval,y*interval) for (x,y) in W]
    
    all_hypotheses = generate_hypothesis(causes,effects,W)
    if level == 0:
        
        if data_id == None:
            #here new_ids contains a list of ids to be loaded
            epx,prima_facie_history = {},[]
            for o_id in new_ids:
                if os.path.exists(log_path + "sk_runlog_" + str(o_id) + ".pickle"):
                    cmc_sk_log = pd.read_pickle(log_path + "sk_runlog_" + str(o_id) + ".pickle")
                    t_epx = cmc_sk_log["epx"]
                    t_prima_facie_history = cmc_sk_log["prima"]
                    epx.update(t_epx)
                    prima_facie_history.extend(t_prima_facie_history)
                    
            prima_facie_history = list(set(prima_facie_history))
                  
        else:
            epx,prima_facie_history = {},[]

        prima_facie_for_effects,prima_facie_history = test_for_prima_facie(all_hypotheses,df,prima_facie_history)
        e_avgs,epx = compute_causal_significance(prima_facie_for_effects,df,epx)

    else:
        print(log_path)
        print(data_id)
        if os.path.exists(log_path + "sk_runlog_" + str(data_id) + ".pickle"):
            cmc_sk_log = pd.read_pickle(log_path + "sk_runlog_" + str(data_id) + ".pickle")
            epx = cmc_sk_log["epx"]
            prima_facie_history = cmc_sk_log["prima"]
            prima_facie_for_effects,prima_facie_history = test_for_prima_facie(all_hypotheses,df,prima_facie_history)
            e_avgs,epx = compute_causal_significance(prima_facie_for_effects,df,epx)

        else:
            #likley no causal relations in previous run
            epx,prima_facie_history = {},[]
            prima_facie_for_effects,prima_facie_history = test_for_prima_facie(all_hypotheses,df,prima_facie_history)
            e_avgs,epx = compute_causal_significance(prima_facie_for_effects,df,epx)
        

    results = compute_pvalues(e_avgs,interval,data_id,df)

    #save computed information
    cmc_sk_log = {"epx":epx, "prima":prima_facie_history}
    if level == 0:
        if data_id == None:
            with open(log_path + "sk_runlog_final_.pickle",'wb') as f:
                pickle.dump(cmc_sk_log,f,pickle.HIGHEST_PROTOCOL)
        else:   
            with open(log_path + "sk_runlog_" + str(data_id) + ".pickle",'wb') as f:
                pickle.dump(cmc_sk_log,f,pickle.HIGHEST_PROTOCOL)
    else:
        with open(log_path + "sk_runlog_" + str(new_ids) + ".pickle",'wb') as f:
            pickle.dump(cmc_sk_log,f,pickle.HIGHEST_PROTOCOL)

    return results

#for sk-base baseline
def run_causal_discovery_sim(causes,effects,interval,df):
    W  = [(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10)]
    W = [(x*interval,y*interval) for (x,y) in W]
    all_hypotheses = generate_hypothesis(causes,effects,W)

    epx,prima_facie_history = {},[]
    prima_facie_for_effects,prima_facie_history = test_for_prima_facie(all_hypotheses,df,prima_facie_history)
    e_avgs,epx = compute_causal_significance(prima_facie_for_effects,df,epx)
    pvals = compute_pvalues(e_avgs,interval,df)
    return pvals






            
