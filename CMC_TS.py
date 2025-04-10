import sys
sys.path.append("../")
import os
import matplotlib.pyplot as plt

#basics
import numpy as np
import pandas as pd
from collections import Counter
#from CMC_utils import wFisher, reconstruct_latent_variables_CMC, is_proper_subset, mod_tricube, score_latent_variables,run_causal_discovery, fdr_bh,resolve_time_conflict
import glob as glob
from pathlib import Path
from joblib import Parallel, delayed

#for clustering
import networkx as nx
from scipy.stats import gamma
import pickle

from CMC_SK_prob_pop import run_causal_discovery_pop_sim,get_edge_probabilities
from CMC_combine import fdr_bh
from CMC_Score import is_proper_subset, mod_tricube, score_latent_variables
from CMC_Reconstruct import reconstruct_latent_variables_CMC,latent_var_post_processing

np.random.seed(10)

class CMC():
    def __init__(self,level,N,effects_to_test,causes_to_exclude,linkedvarsgroup,d = {},v={},recon = {}):
        self.level = level #the level in the clustering hierachy
        self.variable_count = N
        self.effects = effects_to_test
        self.excluded_causes = causes_to_exclude
        self.D = d         #dictionary of datasets with key tied to level and count, value are dataframes
        self.V = v         #variables
        self.CG = {}        #causal structures - using netwrokx to represent
        self.SK = {}       #results from running the base SK code
        self.variable_groups_and_id = {} #list of tuples | (variable group, datasets ids in the same variable group) 
        self.SK_sig = {}
        self.end = False
        self.reconstructed_info = recon
        self.LinkedVarsGroup = linkedvarsgroup
        
    def add_datasets(self,list_of_datasets):
        '''Function to add datasets in level 0 to model'''
        if self.level == 0:
            for i in range(0,len(list_of_datasets)):
                df = list_of_datasets[i]
                df["dID"] = i
                self.D[str(self.level) + "_" +str(i)] = df
                
        else:
            print(f"Only level 0 should have access to this operation, currently at level {self.level}")
            exit()
            
    def get_datasets(self):
        return self.D
            
    def add_variable_sets(self,variable_sets):
        '''Function to add variable sets to model'''
        if self.level == 0:
            for i in range(0,len(variable_sets)):
                new = sorted(variable_sets[i])
                #print(new)
                self.V[str(self.level) + "_" +str(i)] = new
            
            assert len(self.D) == len(self.V) #ensure the dictionaries are the same length
        else:
            print(f"Only level 0 should have access to this operation, currently at level {self.level}")
            exit()

        self.create_uniq_vargroups()

    def create_uniq_vargroups(self):

        #group variable into unique variable set
        all_variable_sets = list(self.V.values())
        unique_variable_groups = list(Counter(map(tuple,all_variable_sets))) #a list of list where each list is a unique variable set.
        print(f"uniq var groups are: {unique_variable_groups}")

        #Store each unique variable set and its dataset ids in dictionary where variable set is the key and dataset ids are the values
        for group in unique_variable_groups:
            ids_in_same_variable_group = [dataset_id for dataset_id in self.V if self.V[dataset_id] == list(group)]
            self.variable_groups_and_id[group] = ids_in_same_variable_group

    def get_variable_sets(self):
        return self.V

    def update_CMC(self,datasets,variables,recon):
        '''Function to update D and V after level = 0'''
        self.D = datasets
        variables = {key: sorted(value) for key, value in variables.items()}
        self.V = variables
        self.reconstructed_info = recon
        self.create_uniq_vargroups()
    
    #TO DO: Add a way to save var groups that were augmented s that we dont re-compute weight eery time
    '''Causal Relationship Related Functions'''  
    def find_causal_relationships(self,min_lag,max_lag,lag_interval,log_path,augmented_datasets = []):
        '''Function to find causal relationships in time series data'''
        
        print(f"augmented_datsets: {augmented_datasets}")

        clusters = self.get_variable_clusters()
        for var_group,dataset_ids in clusters.items():
            data = pd.DataFrame()
            causes = [cause for cause in var_group if cause not in self.excluded_causes]
            effects = causes.copy() #in sim effects are the same as causes
            id_store = []
            for id in dataset_ids:
                data = pd.concat([data,self.D[id]], ignore_index = True)
                id_store.append(int(id.split("_")[1])) #level_i | here we get the i

            id_store.sort()
            joint_ids = "-".join(map(str,id_store))

            if self.level == 0:
                if len(causes) < 2:
                    self.SK[var_group] = pd.DataFrame(columns = ["combined-p-val","epsilon-sign","edge-prob","c-e","ws-we"])
                    print(f"Skipping var group {var_group} with dataset ids {dataset_ids}, only {causes} measured")
                else:
                    print(f"Running Sk in level {self.level} for var group {var_group} with ids {dataset_ids}")
                    result_df = run_causal_discovery_pop_sim(causes,effects,min_lag,max_lag,lag_interval,data,self.level,joint_ids,log_path,sig_level,new_ids = None)
                    self.SK[var_group] = result_df

            else:
                old_ids = [str(self.level-1) + "_" + str(i) for i in id_store]
                print(f"old ids: {old_ids}")
                if any(item in old_ids for item in augmented_datasets):
                    non_recon_ids = [int(id.split("_")[1]) for id in old_ids if id not in augmented_datasets] #list of dataset number that were not reconstructed
                    
                    if len(non_recon_ids) == 0:
                        #ids were reconstructed but it is their first time being reconstricted
                        non_recon_ids = [int(id.split("_")[1]) for id in old_ids]
                    
                    non_recon_ids.sort()
                    non_recon_ids = "-".join(map(str,non_recon_ids))
                    print(f"Non-recon ids: {non_recon_ids}")

                    recon_and_non_recon_ids = [int(id.split("_")[1]) for id in old_ids]
                    recon_and_non_recon_ids.sort()
                    recon_and_non_recon_ids = "-".join(map(str,recon_and_non_recon_ids))
                    print(f"recon ids: {recon_and_non_recon_ids}")

                    #at least one of these ids have had missing variable reconstructed, so we re-run inference acorss all datasets in var_group
                    print(f"Running Sk in level {self.level} for var group {var_group} with ids {dataset_ids}")
                    result_df = run_causal_discovery_pop_sim(causes,effects,min_lag,max_lag,lag_interval,data,self.level,non_recon_ids,log_path,sig_level,new_ids = recon_and_non_recon_ids)
                    self.SK[var_group] = result_df

                else:
                    #these dataset ids have been unchanged
                    print(f"Skipping dataset_ids {dataset_ids}, as they were not modified in the previous iteration")
                    old_SK_result = CMC_stages_log[self.level-1]["SK"][var_group]
                    self.SK[var_group] = old_SK_result
                             
    '''Clustering Related Functions'''
    
    def get_variable_clusters(self):
        for var_group, dataset_ids in self.variable_groups_and_id.items():
            print(f"Variable group: {var_group}, ids: {dataset_ids}")
        return self.variable_groups_and_id
    
    def cluster_variable_sets(self,sig_level):
        clusters = self.get_variable_clusters()

        #For each cluster, perform wfisher and optionally correct for multiple hypothesis
        for var_group,dataset_ids in clusters.items():
            print(f'Summarizing Relationships for: {var_group} with {len(dataset_ids)} datasets')
            data = pd.DataFrame()
            for id in dataset_ids:
                data = pd.concat([data,self.D[id]], ignore_index = True)

            
            summary_df = self.SK[var_group].copy(deep = True)
            causes = [cause for cause in var_group if cause not in self.excluded_causes]
            
            if len(causes) < 2: #causes tested is a single variable
                self.SK_sig[var_group] = pd.DataFrame()
                self.create_local_causal_graph(pd.DataFrame(),var_group)
                

            elif len(summary_df) == 0:
                self.SK_sig[var_group] = pd.DataFrame()
                self.create_local_causal_graph(pd.DataFrame(),var_group)
                
            else:
                summary_df.sort_values(by = ["c-e","p-value"],ascending=[True,True],ignore_index= True,inplace = True)
                summary_df.drop_duplicates(subset=["c-e"], keep = "first", ignore_index = True, inplace = True)
                summary_df["c-e-ws-we"] = summary_df[["c-e","ws-we"]].agg("-".join, axis = 1)
                sig_relationships = summary_df[["c-e-ws-we","p-value","epsilon-sign"]].copy(deep = True)
                sig_relationships.rename(columns = {"p-value":"combined-p-val"}, inplace = True)
                
                if fdr_ind == "true":
                    #mod_sig_level = wFisher(np.asarray([sig_level]*len(dataset_ids)),is_onetail = True)[0] - when relas are tpically very significant?
                    mod_sig_level = sig_level
                    sig_relationships = fdr_bh(sig_relationships,mod_sig_level)
                    sig_relationships.reset_index(drop = True, inplace = True)
                else:
                    sig_relationships = sig_relationships[sig_relationships["combined-p-val"] < sig_level].copy(deep = True)
                
                #get the edge probabailitties
                edge_prob = []
                for _,row in sig_relationships.iterrows():
                    c,e,ws,we = row["c-e-ws-we"].split("-")
                    sgn = int(row["epsilon-sign"])
                    edge_prob.append(get_edge_probabilities(c,e,float(ws)*lag_interval,float(we)*lag_interval,data,sgn))
                sig_relationships["edge-prob"] = edge_prob

                self.SK_sig[var_group]=sig_relationships
                #create causal graphs using significant relationsjops
                self.create_local_causal_graph(sig_relationships,var_group)
                print("\n")

        if len(clusters) == 1:
            self.end = True

    def create_local_causal_graph(self,sig_relationships,var_group):
        cG = nx.DiGraph()
        relationships = {}
        for _,row in sig_relationships.iterrows():
            rela = row["c-e-ws-we"]
            c,e,ws,we = rela.split("-")
            current_pval = row["combined-p-val"]
            
            if (c,e) in relationships:
                if current_pval < relationships[(c,e)][2]:
                    #replace previous relationship with new version
                    relationships[(c,e)] = [ws,we,current_pval,row["epsilon-sign"],row["edge-prob"]]
            else:
                relationships[(c,e)] = [ws,we,current_pval,row["epsilon-sign"],row["edge-prob"]]

        for key,val in relationships.items():
            
            if np.round(val[4],2) >= probability_modulator:
                print(f"Adding relationship {key[0],key[1]}, with direction of {val[3]}, edge probabaility of {val[4]} and window {[int(val[0]),int(val[1])]}")
                cG.add_nodes_from([key[0],key[1]])
                cG.add_edge(key[0],key[1],window = [int(val[0]),int(val[1])],sgn = val[3],p = val[2],probs = val[4])
            else:
                print(f"Excluding relationship {key[0],key[1]}, with direction of {val[3]}, edge probabaility of {val[4]} lower than set threshold of {probability_modulator}")

        variables_with_relationship = set(cG.nodes)
        no_causal_rela_vars = list(set(var_group) - variables_with_relationship)
        print(f'Nodes with no edges: {no_causal_rela_vars}')
        cG.add_nodes_from(no_causal_rela_vars) #add variables with no relationships as nodes in graph
        self.CG[var_group] = cG

    '''Ranking and Scoring Related Functions'''
    def rank_and_score(self,total_num_vars,weights=[1,1]):
        print(f"=============Ranking and Scoring=============")

        result_scoring = pd.DataFrame(columns=["Base","K","p_subset","len","nat","Score","Latent-vars"])
        variable_groups = list(self.variable_groups_and_id.keys())
        
        for var_group in variable_groups:
            #print(var_group)
            if len(var_group) == total_num_vars: #dataset has no missing variables
                continue
            else:
                score,K,H,S1,S2,S0 = 0,None,[],0,0,0
                PairGenerator = ((var_group,other_var_group) for other_var_group in variable_groups if other_var_group != var_group)

                for x,y in PairGenerator:
                    if len(y) >= len(x):
                        latent_variables = list(set(y) - set(x))
                        #print(f"Base:{x}, K:{y}, Latent:{latent_variables}")

                        if self.preconditions_for_recon_met(latent_variables,y,x) == True:
                            s0 = 1 if is_proper_subset(x,y,self.CG) == True else 0
                            s1 = mod_tricube(len(latent_variables),total_num_vars) #fn for the number of latent variables
                            s2 = score_latent_variables(latent_variables,self.CG[y]) #fn for the nature of latent variables
                            tmp_score = s0 + (weights[0] * s1) + (weights[1] * s2)

                            if tmp_score > score:
                                K,score,H,S1,S2,S0 = y,tmp_score,latent_variables,s1,s2,s0

                if K is not None:
                    result_scoring.loc[len(result_scoring)] = [var_group,K,S0,S1,S2,score,H]

        return result_scoring
    
    def preconditions_for_recon_met(self,latent_vars,modelk,modelD):
        '''This functions check for latent variable validity before reconstructing
            relationships
        '''
        if len(latent_vars) == 0:
            return False

        LinkedVarsGroup = self.LinkedVarsGroup
        
        AllLinkedVars = [var for group in LinkedVarsGroup for var in group]
        observed_vars = list(self.CG[modelD].nodes())
        non_linked_vars = [var for var in latent_vars if var not in AllLinkedVars]
        linked_vars = [var for var in latent_vars if var in AllLinkedVars]
        
        '''Conditions for non-linked variables, if all valid, then we can focus on linked vars'''
        #Main Condition 1: Non-linked Latent variable should all have causal relationships
        #Note that is not necessary for linked_vars
        vars_in_y_with_no_edges = list(nx.isolates(self.CG[modelk]))
        latent_vars_with_no_edges = [v for v in non_linked_vars if v in vars_in_y_with_no_edges]
        if len(latent_vars_with_no_edges) != 0:
            return False
        
        for var in non_linked_vars:
            checker = self.preconditions(var,latent_vars,observed_vars,modelk)
            if checker == 0:
                #print(f"Latent variabl")
                return False
        
        '''Conditions for linked latent variables'''
        if len(linked_vars) == 0:
            return True
        
        #get the linked groups the linked latent variables belong to
        latent_linked_groups = [group for group in LinkedVarsGroup if any(item in group for item in linked_vars)]
        
        #for each group in latent_linked_groups, we need to make sure at least one variables in it passes our preconditions
        for latent_group in latent_linked_groups:
            group_res = []
            for var in latent_group:
                group_res.append(self.preconditions(var,latent_vars,observed_vars,modelk))
                res = np.sum(group_res)
                if res == 0: #all linked vars in this latent group were not valid for reconstruction
                    return False
                
        return True

    def preconditions(self,var,latent_vars,observed_vars,modelk):
        causes = list(self.CG[modelk].predecessors(var))
        effects = list(self.CG[modelk].successors(var))
        
        #Condition 1: A latent variable is only in a feedback loop with another latent var
        if len(causes) == 1 and len(effects) == 1:
            if (causes[0] == effects[0]) and (effects[0] in latent_vars):
                return 0
            
        #Condition 2: A latent variable's relationship is only a self-loop
        if len(causes) == 1:
            if causes[0] == var:
                return 0
            
        #Condition 3: Latent variable should have causal relationships with observed variables in D
        connections = causes + effects
        connected_to_D = [v for v in observed_vars if v in connections]
        if len(connected_to_D) == 0:
            return 0

        #Condition 4: If a latent variable (c) with no parent node (pa(c)) is the only parent of an effect (e), ensure that P(c) > P(e)
        #Else e has more than c as a cause. If reconstruction of c takes place, we will be over estimating the number of times it is active

        if len(causes) == 0: #no parent node
            for evar in effects:
                causes_of_evar = list(self.CG[modelk].predecessors(evar)) 
                if len(causes_of_evar) == 1: #evar has only one cause - the current latent var
                    temp = {}
                    datasets_K = self.variable_groups_and_id[modelk]
                    
                    #check if p(c) > p(e)
                    for v in [var,evar]:
                        probability_of_ori_cause,num_of_nonnan_samples = [],[]
                        for idx in datasets_K:
                            data = self.D[idx].copy(deep = True)

                            try:
                                l = data[v].notnull().sum()
                                num_of_nonnan_samples.append(l)
                                #number of 1/total number of non-nan entries (0 and 1)
                                probability_of_ori_cause.append(np.count_nonzero(data[v] == 1)/l)
                            except Exception as e:
                                print(f"An unexpected exception occurred: {str(e)}")
                                
                        temp[v] = np.ma.average(probability_of_ori_cause,weights = num_of_nonnan_samples)

                    if temp[var] < temp[evar]:
                        return 0
                    
                else:
                    pass

        return 1

    '''Reconstruction Related Functions'''
    def perform_reconstruction(self,id_base,id_K,latent_variables,proper_subset_ind):

        print("\n")
        print(f"Reconstructing {id_base} with {id_K}")
        print(f"latent vars: {latent_variables}")

        augmented_datasets,models_to_rereconstruct = [],{}

        #STEP 1: Create K
        K, variable_occurence_probability = self.create_K(id_K)
        print(K)
        print(variable_occurence_probability)

        #STEP 2: To maintain consistent relationships
        if (self.level != 0) and (proper_subset_ind == 1):
            models_to_rereconstruct = self.truth_maintenance(id_base,id_K,latent_variables)
            print(f"Models to re-reconstruct: {models_to_rereconstruct}")

        
        #STEP 3; Get the datasets Id to be reconstructed
        base_datasets = self.variable_groups_and_id[id_base]
        prev_dataset_base = []
        prev_latent_vars = {}
        
        for key,val in models_to_rereconstruct.items():
            prev_dataset_base.extend(val[0]) #get the datasets ids to reconstruct
            latent_vars = val[1]
            for idx in val[0]:
                prev_latent_vars[idx] = latent_vars
                
        prev_dataset_base_id = {x.split("_")[1]:(x.split("_")[0],x) for x in prev_dataset_base}

        #STEP 4: Reconstruct datasets
        for dataset_id in base_datasets:
            #change id to old level version and use to check if it should be re-reconstructed
            idx = dataset_id.split("_")[1]

            if idx not in prev_dataset_base_id.keys():
                #Information about linked variables
                latent_linked_vars_in_K,latent_linked_vars_notin_K,latent_linked_groups,latent_linked_groups_vars_only_in_K = self.linked_latent_variables(K,latent_variables)
                
                print(f"Reconsructing dataset: {dataset_id}")
                raw_D = self.D[dataset_id].copy(deep = True)
                inferred_D = pd.DataFrame()
                uniq_pids = list(raw_D["pID"].unique())

                recon_info = (latent_variables,K,variable_occurence_probability,self.CG[id_K],
                              latent_linked_vars_notin_K,lag_interval)
                latent_info = (latent_variables, latent_linked_vars_in_K,latent_linked_groups,latent_linked_groups_vars_only_in_K)
                results = Parallel(n_jobs=150)(delayed(self.run_recon)(pid,raw_D,recon_info,latent_info) for pid in uniq_pids)

                for res in results:
                    inferred_D = pd.concat([inferred_D,res], ignore_index = True)

                augmented_datasets.append((dataset_id,inferred_D))
            
            else:
                print(f"Correcting previously reconstructed dataset: {idx}")
                nid = prev_dataset_base_id[idx]
                raw_D = self.D[dataset_id].copy(deep = True)
                #we drop var columns in raw_D that are to be reconstructed from prev_latent_vars
                raw_D.drop(columns = prev_latent_vars[nid[1]],inplace = True)
                new_latent_vars = latent_variables + prev_latent_vars[nid[1]]
                print(f"new latent variables: {new_latent_vars}")

                #Information about linked variables
                latent_linked_vars_in_K,latent_linked_vars_notin_K,latent_linked_groups,latent_linked_groups_vars_only_in_K = self.linked_latent_variables(K,new_latent_vars)
                inferred_D = pd.DataFrame()
                uniq_pids = list(raw_D["pID"].unique())

                recon_info = (new_latent_vars,K,variable_occurence_probability,self.CG[id_K],
                              latent_linked_vars_notin_K,lag_interval)
                latent_info = (new_latent_vars, latent_linked_vars_in_K,latent_linked_groups,latent_linked_groups_vars_only_in_K)
                results = Parallel(n_jobs=100)(delayed(self.run_recon)(pid,raw_D,recon_info,latent_info) for pid in uniq_pids)

                for res in results:
                    inferred_D = pd.concat([inferred_D,res], ignore_index = True)

                augmented_datasets.append((dataset_id,inferred_D))

        #STEP 5: #save information that has been reconstricted
        self.save_reconstruction(K,id_K,id_base,base_datasets,latent_variables,proper_subset_ind)

        return augmented_datasets
        
    def create_K(self,id_K):
        '''Function to create the knowledge base K'''

        causal_model = nx.DiGraph()
        causal_model.add_edges_from(self.CG[id_K].edges(data=True)) #getting a copy so we dont change original model
        datasets_K = self.variable_groups_and_id[id_K]
        vars_requiring_complement = [] #causes with negative relationships, we use their complement so all causes are positive
        K = {"relations":{}, "time-windows":{}, "probs":{}, "direction":{}}
        i = 1
        
        variable_occurence_probability = self.get_occurence_probability(causal_model,datasets_K)

        #Create K using updated causal model with the complement of negative causes
        for r in causal_model.edges.data():
            K["relations"][i] = [r[0],r[1]]
            K["time-windows"][i] = [int(r[2]["window"][0]*lag_interval),int(r[2]["window"][1]*lag_interval)]
            ## note that the probability with the negative cause has already been precomputed when combining causal models
            K["probs"][i] = min(r[2]["probs"],0.90)
            K["direction"][i] = r[2]["sgn"]
            i += 1
        
        return K,variable_occurence_probability

    def get_occurence_probability(self,causal_model,datasets_K):

            #get occurence probabaility of origin causes
            variable_occurence_probability = {}
            all_nodes = list(causal_model.nodes)
            origin_causes = []

            for node in all_nodes:
                pred = list(causal_model.predecessors(node))
                if len(pred) == 0:
                    origin_causes.append(node)
                elif len(pred) == 1:
                    if node == pred[0]: #self loop with only itself as an edge , consider as a cause since
                        origin_causes.append(node)
                else:
                    pass

            #for each origin cause, get the weighted average of the probabaility of occurence (using number of samples) acorss all datasets with id_K
            for cause in origin_causes:
                probability_of_ori_cause,num_of_nonnan_samples = [],[]
                for idx in datasets_K:
                    data = self.D[idx].copy(deep = True)

                    
                    try:
                        l = data[cause].notnull().sum()
                        num_of_nonnan_samples.append(l)
                        #number of 1/total number of non-nan entries (0 and 1)
                        probability_of_ori_cause.append(np.count_nonzero(data[cause] == 1)/l)
                    except Exception as e:
                        print(f"An unexpected exception occurred: {str(e)}")
                        
                variable_occurence_probability[cause] = np.ma.average(probability_of_ori_cause,weights = num_of_nonnan_samples)

            #for all other nodes, assign a occurence probabaility of 0.0
            for other_nodes in all_nodes:
                if other_nodes not in variable_occurence_probability:
                    variable_occurence_probability[other_nodes] = 0.0

            return variable_occurence_probability

    def truth_maintenance(self,id_base,id_K,current_latent_variables):

        def check_for_chain_or_common_cause(latent_vars,edges,model,var_being_checked):
            '''Function to check if edge exists due to missing latent variables thats a common cause or in a chain'''
            for edge in edges:
                for lvar in latent_vars:
                    #check for chain:
                    if (model.has_edge(edge[0],lvar) == True) and (model.has_edge(lvar,edge[1]) == True):
                        return True 
                    #check for common cause
                    if (model.has_edge(lvar,edge[0]) == True) and (model.has_edge(lvar,edge[1]) == True):
                        return True
                        
            return False

        CG_K = self.CG[id_K]
        to_recon = {}

        #step 1 check if id base was previsoulsy used as K in prev level
        if id_base in self.reconstructed_info:
            print(f"Var group {id_base} previsouly used as K, checking for differences in causal relationships")
            prior_K = self.reconstructed_info[id_base]

            #step 2. For each previous id_base, get the previous latent variables and check with current K
            for prev_id_base,value in prior_K.items():
               
                if value[3] == 1:
                    prev_latent_vars = value[1]
                    prev_id_base_K = list(value[2]["relations"].values())
                    print(f"Prev id base is {prev_id_base} with latent vars: {prev_latent_vars}")
                    check,latent_recon = False,[]

                    for var in prev_latent_vars: 
                        #step 3. For each latent var, get the edges in prior K and current K
                        prev_edges_temp = [tuple(edge) for edge in prev_id_base_K if var in edge]
                        K_edges = list(CG_K.in_edges(var)) + list(CG_K.out_edges(var))
                        print(f"edges for {var} in prior K were :{prev_edges_temp}; and in current K: {K_edges}")

                        #step 4. check if all its edges in prior K are in current K
                        edges_in_K, edges_not_in_K = [],[]
                        for edge in prev_edges_temp:
                            if edge in K_edges:
                                edges_in_K.append(edge)
                            else:
                                edges_not_in_K.append(edge)
                        
                        #step 5: if all edges are in current K, no need to re-reconstruct
                        if len(edges_not_in_K) == 0:
                            print(f"All the edges of {var} are in both the prior and current K. No need to re-reconstruct this prior base model")
                        else:
                            check = check_for_chain_or_common_cause(current_latent_variables,edges_not_in_K,CG_K,var)
                            if check == True:
                                print(f"A diff relationships (likely chain or common cause) involving a prior latent variable {var} has been found")
                                latent_recon.append(var)

                    if len(latent_recon) != 0:
                        to_recon[prev_id_base] = (value[0],latent_recon)

                else:
                    print(f"Var group {prev_id_base} was not a proper subset of {id_base}. Skipping") 
        else:
            print(f"Var group {id_base} has not been used as K. Moving on to normal reconstruction")          
        return to_recon

    def linked_latent_variables(self,K,latent_variables):
        '''Function to unpack latent variables and get the linked variables w/o relations in K'''

        LinkedVarsGroup = self.LinkedVarsGroup
        AllLinkedVars = [var for group in LinkedVarsGroup for var in group]
        latent_linked_vars = [var for var in latent_variables if var in AllLinkedVars] #linked vars in latent variables

        #Goal here to is get list of linked vars both in/not in K
        relas = K["relations"]
        vars_in_K = [var for values in relas.values() for var in values]
        latent_linked_vars_in_K = [var for var in latent_linked_vars if var in vars_in_K] #list of only latent linked vars in K
        latent_linked_vars_notin_K = [var for var in latent_linked_vars if var not in vars_in_K] #list of only latent linked vars not in K

        #Goal here is to get a list of tuples where each tuple contains all the linked forms
        #and another where the tuple only contains the linked forms in K

        #get the linked groups the linked latent variables belong to
        latent_linked_groups = [group for group in LinkedVarsGroup if any(item in group for item in latent_linked_vars)]
        latent_linked_groups_vars_only_in_K = []
        for group in latent_linked_groups:
            temp = [var for var in group if var in vars_in_K]
            latent_linked_groups_vars_only_in_K.append(tuple(temp))
        

        return latent_linked_vars_in_K,latent_linked_vars_notin_K,latent_linked_groups,latent_linked_groups_vars_only_in_K

    def run_recon(self,pid,df,recon_info,latent_info):
        latent_vars,K,variable_occurence_probability,CG_K,latent_linked_vars_notin_K,lag_sample = recon_info
        latent_vars,latent_linked_vars_in_K,latent_linked_groups,latent_linked_groups_vars_only_in_K = latent_info

        df_pid = df[df["pID"] == pid].copy(deep = True)
        df_pid.reset_index(drop = True, inplace = True)

        df_pid_inferred = reconstruct_latent_variables_CMC(latent_vars,K,df_pid,variable_occurence_probability,CG_K,
                                                          latent_linked_vars_notin_K,lag_sample)
        
        df_pid_inferred = latent_var_post_processing(df_pid_inferred,latent_vars,latent_linked_vars_in_K,latent_linked_groups,latent_linked_groups_vars_only_in_K)
        return df_pid_inferred
    
    def save_reconstruction(self,K,id_K,id_base,dataset_base,latent_variables,ind):
        #at each level, we want to store the K - relatons and time windows, the latent variables to be reconstrictred,dataset-ids to be reconstructed, K variable group
        if id_K not in self.reconstructed_info:
            self.reconstructed_info[id_K] = {}
        
        print(f"Saving Reconstruction History of {id_base} using {id_K}")
        self.reconstructed_info[id_K][id_base] = [dataset_base,latent_variables,K,ind] #.update({id_base=value})

    
