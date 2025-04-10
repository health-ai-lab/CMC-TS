import numpy as np
from scipy.stats import gamma
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection
import pandas as pd
from itertools import product,chain,combinations
from collections import Counter
import operator

np.random.seed(0)
from CMC_SK_prob_pop import get_edge_probabilities


def relationship_status(df,pids,sig):
    '''Function to return status of for a specific causal relationships acorss all forms (dataset, time windows)'''
    
    '''Testing non-significance'''
    if (df["p-value"] >= sig).all():
        return "non-sig"
    
    '''Testing significance'''
    store, direc = [],[]
    
    for pid in pids:
        df_pid = df[df["d_id"] == pid].copy(deep = True)
        if (df_pid["p-value"] < sig).any():
            store.append(1)
            direc.append(df_pid[df_pid["p-value"] < sig]["epsilon-sign"].mode().tolist()[0])

    #all datasets have a p-value less than sig
    if len(store) == len(pids):
        #and they have the same effect direction
        if all(x == direc[0] for x in direc):
            return "all-sig"
        else:
            #all dataset have sig windows, but their effect direction is different hence we need to resolve it
            return "conflict"
    else:
        #Conflict in relations - some datasets have rela, some dont
        return "conflict"
    
def resolve_relations(df,pids,summary_df,sig_level,D,interval):
    
    '''Function to resolve conflicts in causal relationships'''

    df["c-e-ws-we"] = df[["c-e","ws-we"]].agg("-".join, axis = 1)
    ndf = df[["d_id", "c-e-ws-we"]].copy(deep = True)
    dfs = []
    for pid in pids:
        df_id = ndf[ndf["d_id"] == pid]
        dfs.append(pd.DataFrame({"rela" + pid : df_id["c-e-ws-we"]}))
        
    # Create all possible combinations of rows acorss all dataset_ids
    #Ex: a row could be bolus-hyper-15-30 | bolus-hyper-15-20
    combinations = list(product(*[list(chain.from_iterable(tdf.values.tolist())) for tdf in dfs]))
    rdf = pd.DataFrame(combinations, columns=pids)

    temp = []
    
    #for each row, we get the information associated with each column (i.e dataset id), use wfisher and resolve time window
    for _,row in rdf.iterrows():
        pvals,sgn_effect_sizes,weights,t,edges = [],[],[],[],[]
        
        for pid in pids:
            temp_summary_df = summary_df[summary_df["d_id"] == pid].copy(deep = True)
            temp_summary_df.set_index("c-e-ws-we", inplace = True)
            c,e,ws,we = row[pid].split("-")
            pvals.append(temp_summary_df.at[row[pid],"p-value"])
            sgn_effect_sizes.append(temp_summary_df.at[row[pid],"epsilon-sign"])
            weights.append(temp_summary_df.at[row[pid],"weight"])
            t.append(temp_summary_df.at[row[pid],"ws-we"])
            edges.append(get_edge_probabilities(c,e,float(ws)*interval,float(we)*interval,D[pid]))
            
        combined_pval,combined_sgn_effect_size = wFisher(np.array(pvals), np.array(sgn_effect_sizes), np.array(weights), is_onetail = False)
        ws_we = resolve_time_conflict(t)
        avg_edge = min(0.90,np.ma.average(edges,weights = weights))
        
        temp.append((combined_pval, combined_sgn_effect_size,ws_we,avg_edge))

    #Since we may have multiple forms in temp, we pick the one with the highest p-val and non mathching interval (i.e ws != we)
    temp.sort(key=lambda x: x[0]) #sort by p-value
    if temp[0][0] >= sig_level:
        return (temp[0])
    else:
        for p,ey,t,edge in temp:
            if (t[0] != t[1]) and (p < sig_level):
                return (p,ey,t,edge)
    return (temp[1])       
        
def resolve_time_conflict(windows):
    '''Function find the best overlap of time to use for conflicting time windows'''
    time_windows = []
    #1. convert windows to int
    windows = list(set(windows))
    for window in windows:
        window = window.split("-")
        ws = int(window[0])
        we = int(window[1])
        time_windows.append((ws,we))

    if len(time_windows) == 1:
        return (time_windows[0][0],time_windows[0][1])
    else:
        #crete all possible pairs
        z = list(combinations(time_windows,2))
        overlap_windows = []
        for i in z:
            x,y = i
            x = np.arange(x[0],x[1]+1)
            y = np.arange(y[0],y[1]+1)
            
            overlap_window = sorted(tuple(set(x).intersection(set(y))))
            overlap_window = tuple(overlap_window)
            overlap_windows.append(overlap_window)
            
        overlap_windows = Counter(overlap_windows)
        selected_window = max(overlap_windows.items(), key=operator.itemgetter(1))[0]

        if len(selected_window) == 0:
            #no mathing windows, relationships not likely to be true?
            ws,we = "ws","we"
        else:
            ws,we = selected_window[0],selected_window[-1]
        return (ws,we)
    
def fdr_bh(df,alpha,state=None):
    '''Compute the Benjamini-Hochberg correction for the list of pvals 
        and get the rejected hypotheses (i.e where we reject the null hypo
        that the relationship is not significant)'''

    sig_relationships = pd.DataFrame(columns= ["c-e-ws-we","combined-p-val","epsilon-sign"])

    p_values = df["combined-p-val"].to_numpy()
    #http://www.biostathandbook.com/multiplecomparisons.html
    rejected, corrected = fdrcorrection(p_values, alpha, method='indep')
    
    for i,row in df.iterrows():
        if rejected[i] == True:
            sig_relationships.loc[len(sig_relationships)] = [row["c-e-ws-we"],row["combined-p-val"],row["epsilon-sign"]]

    return sig_relationships

def wFisher(p, sgn_effect_sizes = None, weight = None, is_onetail = True):
    ''' Function used to compute the weighted fisher method'''
    '''Inputs:
            p : a numeric list or array of p-values for each experiment
            sgn_effect_sizes: a list of signs of effect sizes, either 1 or -1 (i assume the avg causal significance are effect sizes)
            weight: a list of weight or sample size for each experiment
            is_onetail : specify is p-valuesa re combined without conideration of effect 
        Outputs:
          p: the combined p-value
          overall_eff_direction: the direction of combined effects
    '''
    def get_result(p,Ns):
        G = []
        for i in range(0,len(p)):
            G.append(np.round(gamma.isf(q=p[i],a=Ns[i],scale=2),6))
        Gsum = np.sum(G)
        resultP = gamma.sf(x=Gsum,a=N,scale=2)
        return resultP

    try:
        if weight is None:
            weight = np.ones(len(p))
        
        #get index where p-value is NA
        idx_na = np.where(np.isnan(p))[0]

        #delete the values at the index where pvalue is NA
        if (len(idx_na) > 0):
            for i in idx_na:
                del p[i]
                del weight[i]

                if(is_onetail == False):
                    del sgn_effect_sizes[i]

        NP = len(p)             #number of experiments
        NS = len(weight)
        assert NP == NS
        N = NS
        Ntotal = np.sum(weight) #sum of total weight
        ratio = weight/Ntotal  #standardize by dividing each weight by the total weight
        Ns = N*ratio           #scale weights by the number of experiments

        if is_onetail == True:
            resultP = get_result(p,Ns)

        else:
            p1=p2=p
            idx_pos = np.where(sgn_effect_sizes > 0)[0]
            idx_neg = np.where(sgn_effect_sizes < 0)[0]

            #positive direction
            p1 = [p[i]/2 if i in idx_pos else p1[i] for i in range(0,len(p1))]
            p1 = [1-(p[i]/2) if i in idx_neg else p1[i] for i in range(0,len(p1))]
            resultP1 = get_result(p1,Ns)

            #negative direction
            p2 = [1-(p[i]/2) if i in idx_pos else p2[i] for i in range(0,len(p2))]
            p2 = [p[i]/2 if i in idx_neg else p2[i] for i in range(0,len(p2))]
            resultP2 = get_result(p2,Ns)

            resultP = 2 * min(resultP1,resultP2)
            if resultP > 1.0:
                resultP = 1.0
        
            if resultP1 <= resultP2:
                overall_eff_direction = 1
            else:
                overall_eff_direction = -1

        if is_onetail == True:
            RES = [min(1,resultP)]
        else:
            RES = [min(1,resultP),overall_eff_direction]
    
        return RES

    except:
        return [1.0,1]