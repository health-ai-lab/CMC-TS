
import itertools
import numpy as np
import math
import sys
import copy
import pandas as pd
#for clustering
import networkx as nx
from collections import Counter
import operator
import random

#for latent variable reconstruction
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import scipy.stats as ss
from scipy.special import softmax
import operator
from itertools import combinations

np.random.seed(0)


'''Code for Reconstructing latent variables'''

def get_keys_by_value(dictionary, search_value):
    keys_with_value = []
    for key, value in dictionary.items():
        if value == search_value:
            keys_with_value.append(key)
    return keys_with_value

def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

def label_edges(causal_relations,variables_in_K,G):

    '''Function used to label the unrolled edges of a network'''
    '''Inputs is the dictionary of causal relationships'''

    def setup(causal_relations,all_nodes,G):
        '''This function is used to change the order of edges in the list. We place variables with the most neighbours first'''
        #all_nodes = list(G.nodes())
        edges = [(val[0],val[1]) for _,val in causal_relations.items()]

        connection_frequency = {}
        for node in all_nodes:
            connection_frequency[node] = len(list(nx.all_neighbors(G,node)))
        connection_frequency = dict(sorted(connection_frequency.items(), key=operator.itemgetter(1),reverse=True))

        order = []
        for key,_ in connection_frequency.items():
            for edge in edges:
                if key in edge:
                    order.append(edge)
        operation_order = unique(order)
        return operation_order

    operation_order = setup(causal_relations,variables_in_K,G)
    #print(f"operation order: {operation_order}")
    edges, all_var_labels,in_use,var_edges = [],[],[],[]

    for rela in operation_order:
        cause,effect = rela
        if cause == effect:
            loop = True
        else:
            loop = False

        if len(edges) == 0:
            all_var_labels.extend([cause + "|1",effect + "|2"])
            edges.append((cause + "|1",effect + "|2"))
            in_use.extend([cause,effect])
            var_edges.append((cause,effect))

        else:
            if (cause in in_use) and (effect not in in_use):
                #print("1")
                cause_labels = [x.split("|")[1] for x in all_var_labels if x.split("|")[0] == cause]
                cause_label = min(cause_labels)
                cause_and_label = cause + "|" + cause_label
                effect_and_label = effect + "|" + str(int(cause_label)+1)

            elif (cause not in in_use) and (effect in in_use):
                #print("2")
                effect_labels = [x.split("|")[1] for x in all_var_labels if x.split("|")[0] == effect]
                effect_label = min(effect_labels)
                cause_and_label = cause + "|" + str(int(effect_label)-1)
                effect_and_label = effect + "|" + effect_label

            elif (cause not in in_use) and (effect not in in_use):
                #print("3")
                #hasnt occured before and no current combination
                cause_and_label = cause + "|1"
                effect_and_label = effect + "|2"

            elif (cause in in_use) and (effect in in_use):
                #print("4")

                if loop:
                #i.e a self loop is present: ["A","A"]
                #first check if the variable has been used
                    if cause in in_use:
                        all_cause_labels = []
                        for edge in edges:
                            if cause in edge[0]:
                                all_cause_labels.append(int(edge[0].split("|")[1]))
                            if cause in edge[1]:
                                all_cause_labels.append(int(edge[1].split("|")[1]))

                        min_cause_label = min(all_cause_labels) #e.g [1,3]
                        cause_and_label = cause + "|" + str(min_cause_label) #A1
                        #if len(all_cause_labels) > 1:
                        effect_and_label = effect + "|" + str(all_cause_labels[-1] + 1) #A4
                    
                    else:
                        #if variable has not been used in any capacity
                        cause_and_label = cause + "|1"
                        effect_and_label = cause + "|2"

                else:
                    #check if nodes have appeared together in the same edge pair
                    ind = [edge for edge in var_edges if (cause in edge) and (effect in edge)]

                    if len(ind) > 0:
                        #this means a feedback loop has occured . Such as: H->D, D->H
                        assert cause == ind[0][1] #the current cause was previoulsy the effect in this feedack loop
                        #get the index of the edge in var_edges list to get the index in edges list
                        idx  = [index for (index, item) in enumerate(var_edges) if item == ind[0]][0]
                        current_edge_labels = edges[idx] #H1->D2
                        #cause_label = current_edge_labels[1].split("|")[1] #2
                        cause_and_label = current_edge_labels[1] #D2
                        cause_labels = [x.split("|")[1] for x in all_var_labels if x.split("|")[0] == effect]
                        cause_label = max(cause_labels)
                        effect_and_label = effect + "|" + str(int(cause_label)+1) #H3

                    else: #cause and effect not together in any current edges e.g "H","D" and "D","H"
                        cause_labels = [x.split("|")[1] for x in all_var_labels if x.split("|")[0] == cause]
                        cause_label = min(cause_labels)
                        effect_labels = [x.split("|")[1] for x in all_var_labels if x.split("|")[0] == effect]
                        effect_label = max(effect_labels)
                        cause_and_label = cause + "|" + cause_label
                        effect_and_label = effect + "|" + str(int(effect_label)+1)
                
            all_var_labels.extend([cause_and_label,effect_and_label])
            all_var_labels = list(set(all_var_labels))
            edges.append((cause_and_label,effect_and_label))
            in_use.extend([cause,effect])
            var_edges.append((cause,effect))
            #print(edges)

    return edges, all_var_labels

def CPD_effects(variable_layer_labels,origin_causes,edges,causal_relations,causal_relations_prob,network):
    
    #define CPDs for the effect variables
    effect_variables = [var for var in variable_layer_labels if var not in origin_causes]
    #print(f"effect vars {effect_variables}")

    for effect_var in effect_variables:
        #print(effect_var)
        causes_of_effect = [c for c,e in edges if e == effect_var] #store causes of effect
        #print(f"causes of effect {causes_of_effect}")

        relation_index = [] #index for cause and effect in relations dictionary
        relation_prob = [] #probability for the relations index from above
        cards = []

        for cause in causes_of_effect:
            cards.append(2)
            index = [key for key,val in causal_relations.items() if (val[0] == cause.split("|")[0]) and (val[1] == effect_var.split("|")[0])][0]
            relation_index.append(index)
            relation_prob.append(causal_relations_prob[index])

        #compute all possible combinations of states. If two cause -> [[0,0],[0,1],[1,0],[1,1]] | if three causes -> [[0,0,0],[0,0,1],[0,1,0]...[1,1,1]]
        all_combinations = [list(i) for i in itertools.product([0, 1], repeat=len(causes_of_effect))]
        combination_probs = []
        #print(all_combinations)
        
        #compute conditional probabaility
        for combination in all_combinations:
            if sum(combination) == 0: #all zeros states
                probs = [1.0,0.0]   #zero state is prob of 1, one state is 0
            else:
                zero_prob = 1.0
                for i in range(0,len(combination)):
                    if combination[i] == 1: #if the ith cause is true
                        #get the causal relationship's probability when it does not occur, i.e zero state
                        zero_prob = zero_prob * (1.0 - causal_relations_prob[relation_index[i]])
                probs = [zero_prob,1.0-zero_prob]

            combination_probs.append(probs)

        #print(combination_probs)
        combination_probs = (np.asmatrix(combination_probs)).transpose()
        
        cpd = TabularCPD(variable=effect_var,variable_card=2,values=combination_probs,evidence=causes_of_effect,evidence_card=cards)
        network.add_cpds(cpd)

    return network

def inference(list_of_prob_and_evidence,infer_data,var,H_edgs,causal_model_dic,latent_vars,markov_vars,H_invars,
              H_effect_causes,H_effects,no_noise_data,causal_window_dic,var_infer,timestamp,sampling_interval,causal_relations_dir,causal_relations_prima_dir):

    #latent_ts = (infer_data[var]).tolist()
    latent_ts = {it:0 for it in timestamp}

    #Create a causal model of only variables markov blanket
    h_causal_model_dic = {}  #e.g. {1:(A_1,B_2),2:....}
    for edge in H_edgs:
        for index,(cause,effect) in causal_model_dic.items():
            if cause in edge[0] and effect in edge[1]:
                h_causal_model_dic[index]= edge


    #Get only the latent variables inluded in *current* variables markov blanket
    h_latent_variables = []
    for h_itv in latent_vars:
        for h_iitv in markov_vars:
            if h_itv in h_iitv:
                h_latent_variables.append(h_iitv)

    observed_activated_timepoints = {} #{{var_i:{"causes":{cause:[]}}}}

    for var_i in H_invars:
        #initialze the list_of_prob_and_evidence
        list_of_prob_and_evidence[var_i]=[]
        observed_activated_timepoints[var_i] = {}
    
    
    for it in timestamp:
    
        #inferring for each different latetn for var e.g. H_1,H_2,... H_3 the probability result for each latent H_1,H_2,....
        h_prob_result_dic = {}  #e.g. {H_1:0.3,H_2:0.4,...}
       
        for var_i in H_invars:
            #print(f"var_i  is {var_i}")
            h_prob_result_dic[var_i] = 0.0 #initilize the h_prob_result_dic
            #print(h_prob_result_dic)
            
            h_other_latent_vars = [other_latent_var for other_latent_var in h_latent_variables if other_latent_var != var_i] #get all other latent varialbes except the one being inferred
            current_evidence = [] #available evidences for current inference [("A_3",1),("B_4",1),..]

            #get the relevant relationships associated with the variant of the latet variable 
            #TO DO; add parent of child of causal relations
            relas_to_consider = []
            for _,(cause,effect) in h_causal_model_dic.items():
                if cause in [var_i] or effect in [var_i]:
                    relas_to_consider.append((cause,effect))
            #print(f"relas to consider : {relas_to_consider}")

            #find all relationships that connected to the covered vars
            for rela_index,(cause,effect) in h_causal_model_dic.items():
                
                #the the target variable is a cause of the covered vars and the target variable is not a latent varialbe iteself
                #except the current inferring one. This can happend when we have multiple latent varialbes
                #if effect in h_covered_vars.keys() and cause in h_covered_vars.keys() and rela_index not in checked_relas.keys():
                #    checked_relas[rela_index] = (cause,effect)
                
                '''The target variable is a cause of the latent variable'''
                if effect == var_i:
                    if it == timestamp[0]:
                        observed_activated_timepoints[var_i]["causes"] = {}
                    
                    if cause not in h_other_latent_vars: #cause is a latent variable
                        var_name = (cause.split("|"))[0]
                        var_ts = no_noise_data.set_index('timestamp')[var_name].to_dict()

                        if cause not in observed_activated_timepoints[var_i]["causes"]:
                            observed_activated_timepoints[var_i]["causes"][cause] = []

                        h_obs_flag,h_has_ob_fl,var_active = False,False,[]

                        for t in range(max(0,int(it) - causal_window_dic[rela_index][1]),max(0,int(it) - causal_window_dic[rela_index][0]+sampling_interval),sampling_interval):
                            h_has_ob_fl = True
                            try:
                                if var_ts[t] == 1:
                                    if t not in observed_activated_timepoints[var_i]["causes"][cause]:
                                        h_obs_flag = True
                                        var_active.append(t)

                                #randomly select a t in var_active. This ensures we do not reuse in the same cause (at t) to activate multiple effect timepoints
                                observed_activated_timepoints[var_i]["causes"][cause].append(random.choice(var_active))
                            except:
                                pass
                        
                        #add observations
                        if h_obs_flag == True:
                            if (cause,1) not in current_evidence:
                                if (causal_relations_dir[rela_index] == -1) and (causal_relations_prima_dir[rela_index] == 1):
                                    current_evidence.append((cause,0))
                                else:
                                    current_evidence.append((cause,1))

                        if h_obs_flag == False and h_has_ob_fl == True:
                            if (cause,0) not in current_evidence:
                                if (causal_relations_dir[rela_index] == -1) and (causal_relations_prima_dir[rela_index] == 1):
                                    current_evidence.append((cause,1))
                                else:
                                    current_evidence.append((cause,0))

                '''The target variable is an effect of the latent variable'''
                if cause == var_i:
                    if it == timestamp[0]:
                        observed_activated_timepoints[var_i]["effects"] = {}

                    if effect not in h_other_latent_vars: #effect is not a latent variable
                        #get the list for effect
                        var_name = (effect.split("|"))[0]
                        var_ts = no_noise_data.set_index('timestamp')[var_name].to_dict()
                        h_obs_flag, h_has_ob_fl,var_active = False,False,[]

                        if effect not in observed_activated_timepoints[var_i]["effects"]:
                            observed_activated_timepoints[var_i]["effects"][effect] = []

                        for t in range(min(int(timestamp[-1]),int(it) + causal_window_dic[rela_index][0]),min(int(timestamp[-1]),int(it) + causal_window_dic[rela_index][1]+sampling_interval),sampling_interval):
                            h_has_ob_fl = True
                            try:
                                if var_ts[t] == 1:
                                    if t not in observed_activated_timepoints[var_i]["effects"][effect]:
                                        h_obs_flag = True
                                        var_active.append(t)

                                #randomly select a t in var_active. This ensures we do not reuse in the same effect (at t) to activate multiple cause timepoints
                                observed_activated_timepoints[var_i]["effects"][effect].append(random.choice(var_active))
                            except:
                                pass
                                    
                        #add observations
                        if h_obs_flag == True:
                            if (effect,1) not in current_evidence:
                                if (causal_relations_dir[rela_index] == -1) and (causal_relations_prima_dir[rela_index] == 1):
                                    current_evidence.append((effect,0))
                                else:
                                    current_evidence.append((effect,1))

                        if h_obs_flag == False and h_has_ob_fl ==True:
                            if (effect,0) not in current_evidence:
                                if (causal_relations_dir[rela_index] == -1) and (causal_relations_prima_dir[rela_index] == 1):
                                    current_evidence.append((effect,1))
                                else:
                                    current_evidence.append((effect,0))

                '''The target variable is a parent of a child of the latent variable'''
                if (cause != var_i) and (cause in H_effect_causes) and (effect != var_i) and (effect in H_effects) and (cause not in H_invars):
                    if (cause not in h_other_latent_vars) and (effect not in h_other_latent_vars):
                        var_name_cause = (cause.split("|"))[0]
                        #var_ts_cause = (no_noise_data[var_name_cause]).tolist()
                        var_ts_cause = no_noise_data.set_index('timestamp')[var_name_cause].to_dict()
                        var_name_effect = (effect.split("|"))[0]
                        #var_ts_effect = (no_noise_data[var_name_effect]).tolist()
                        var_ts_effect = no_noise_data.set_index('timestamp')[var_name_effect].to_dict()
                        h_obs_flag, h_has_ob_fl,effect_occurs,var_active,effect_active = False,False,False,[],[]

                        '''step 1: find out if the effect occurs relative to var_i''' #(var_i,effect)
                        #print(h_causal_model_dic)
                        #print(f"details: {var_i}, {cause}, {effect}")
                        index = get_keys_by_value(h_causal_model_dic,(var_i,effect))
                        if len(index) != 0:
                            index = index[0]
                        else:
                            continue
                        
                        for t in range(min(int(timestamp[-1]), int(it) + causal_window_dic[index][0]), min(int(timestamp[-1]), int(it) + causal_window_dic[index][1]+sampling_interval),sampling_interval):
                            try:
                                if var_ts_effect[t] == 1:
                                    effect_occurs = True
                                    effect_active.append(t)
                            except:
                                pass
   
                        if effect_occurs == True:
                            '''Step 2a: if effect occurs, check if cause could have caused it'''
                            for curr_t in effect_active:
                                for t in range(max(0,curr_t - causal_window_dic[rela_index][1]), max(0, curr_t - causal_window_dic[rela_index][0]+sampling_interval),sampling_interval):
                                    h_has_ob_fl = True
                                    try:
                                        if var_ts_cause[t] == 1:
                                            h_obs_flag = True
                                            var_active.append(1)
                                    except:
                                        pass
                        

                        if h_obs_flag == True:
                            if (cause,1) not in current_evidence:
                                current_evidence.append((cause,1))

                        if h_obs_flag == False and h_has_ob_fl == True:
                            if (cause,0) not in current_evidence:
                                current_evidence.append((cause,0))

            
            fina_pk,previously_computed_flag = 0.0,False
            #check whether we alredy compuyted the probability
            for prob_and_evidence in list_of_prob_and_evidence[var_i]:
                for prob_val,prior_evidence in prob_and_evidence.items():
                    if set(prior_evidence) == set(current_evidence):
                        previously_computed_flag = True
                        fina_pk = prob_val

            if previously_computed_flag == True:
                h_prob_result_dic[var_i] = fina_pk
            else:
                h_eviden_dic = {var:int(val) for var,val in current_evidence} #compute the pk from available evidence
                h_result = (var_infer.query([var_i],evidence = h_eviden_dic,joint = False, show_progress = False))[var_i]
                fina_pk = 0.0 if math.isnan((h_result.values[0])) == True else h_result.values[1]
                h_prob_result_dic[var_i] = fina_pk
                list_of_prob_and_evidence[var_i].append({fina_pk: current_evidence}) #looks like [{0.0: [(A,1),(B,0)]},{0.25: [(A,0),(B,0)]}], all possible prob values and their evidences

       
        fina_zero = 1.0  #decide the final pk
        for _,prob_val in h_prob_result_dic.items():
            fina_zero = fina_zero * (1.0 - prob_val)
        fina_one = 1.0 - fina_zero
        

        '''
        deci = ss.bernoulli.rvs(fina_one,0)
        if deci==1:
            #new: louis check if origin cause
            #if (H_invars[0] not in f_causes) and (H_invars[0] not in f_effects) and (len(H_invars) == 1):
            #if (len(f_causes) == 0) and (len(H_invars) == 1) and (len(f_effects) == 1) and (len(f_effect_causes) == 1):
            #    r = H_invars[0].split("|")[0]
            #    fina_one = occ_prob[r]
            #    effect_prev = infer_data[f_effects[0].split("|")[0]].value_counts(1)[1]
            #    if fina_one < effect_prev:
            #        deci = ss.bernoulli.rvs(fina_one/effect_prev,0)
            #    else:
            #        print("error in bottom of recon")
            #        exit()
                
            latent_ts[it]=deci
            '''
        
        #added for linked variables - 08/07
        latent_ts[it] = fina_one #ss.bernoulli.rvs(fina_one,0)#fina_one

    #update the whole series
    #infer_data[var]=list(latent_ts.values())
    temp = []
    for _,val in infer_data.iterrows():
        curr_time = val["timestamp"]
        temp.append(latent_ts[curr_time])
    infer_data[var] = temp

    return infer_data,list_of_prob_and_evidence

def reconstruct_latent_variables_min_old(latent_variables,knowledge_base_K,D,occurence_probabaility_D,max_layer):

    #causal_relations, causal_relations_layers, causal_relations_windows,causal_relations_windows_probs,max_layer = knowledge_base_K
    causal_relations = knowledge_base_K["relations"]
    causal_relations_layers = knowledge_base_K["layers"]
    causal_relations_windows = knowledge_base_K["time-windows"]
    causal_relations_windows_probs = knowledge_base_K["probs"]

    inferred_data = D.copy(deep = True)
    #set the series needs inferring to list filled with zeros
    for it in latent_variables:
        inferred_data[it]=(np.zeros(len(D),dtype=int)).tolist()

    variables_in_K = []
    for _,v in causal_relations.items():
        variables_in_K.extend(v)
    variables_in_K = list(set(variables_in_K))

    #technically should not need this
    not_noise_vars = [var for var in variables_in_K if var not in latent_variables]
    no_noise_data = D[not_noise_vars].copy(deep = True)

    #label all variables after unrolling the network
    #edges, variable_layer_labels = label_edges(causal_relations,variables_in_K)
    
    #label all variables accoridng to layer number
    edges, variable_layer_labels = label_variables(causal_relations,causal_relations_layers)
    
    #define BN and populate for the knowledge base
    whole_bayes = BayesianNetwork()
    inferred_network = define_baysian_network(whole_bayes,edges,occurence_probabaility_D,variables_in_K,
                                        max_layer,variable_layer_labels,causal_relations,
                                        causal_relations_windows_probs)

    #infer latent variables here
    inferred_data = infer_latent_variables(latent_variables,edges,variable_layer_labels,inferred_network,
                                        causal_relations,causal_relations_windows_probs,inferred_data,no_noise_data,
                                        causal_relations_windows)

    return inferred_data

def reconstruct_latent_variables_CMC(latent_variables,knowledge_base_K,D,occurence_probabaility_D,K_model,linked_vars_not_K,sampling_interval):#,variable_type):

    #causal_relations, causal_relations_layers, causal_relations_windows,causal_relations_windows_probs,max_layer = knowledge_base_K
    causal_relations = knowledge_base_K["relations"]
    #causal_relations_layers = knowledge_base_K["layers"]
    causal_relations_windows = knowledge_base_K["time-windows"]
    causal_relations_windows_probs = knowledge_base_K["probs"]
    causal_relations_directions = knowledge_base_K["direction"]
    causal_relations_prima_directions = knowledge_base_K["prima-direction"]

    #print(D)
    #print(f"len of D: {len(D)}")
    inferred_data = D.copy(deep = True)
    #set the series needs inferring to list filled with zeros
    for it in latent_variables:
        inferred_data[it]=(np.zeros(len(D),dtype=int)).tolist()

    variables_in_K = []
    for _,v in causal_relations.items():
        variables_in_K.extend(v)
    
    variables_in_K = list(set(variables_in_K))

    #technically should not need this
    not_noise_vars = [var for var in variables_in_K if var not in latent_variables]
    not_noise_vars.append("timestamp")
    no_noise_data = D[not_noise_vars].copy(deep = True)
    timestamp = no_noise_data["timestamp"].to_list()
    #print(f"len of timestamp: {len(timestamp)}")

    #label all variables after unrolling the network
    edges, variable_layer_labels = label_edges(causal_relations,variables_in_K,K_model)
    print(f"edges: {edges}, var layer label: {variable_layer_labels}")
    
    #define BN and populate for the knowledge base
    whole_bayes = BayesianNetwork()

    inferred_network = define_baysian_network_CMC(whole_bayes,edges,occurence_probabaility_D,variables_in_K,
                                        variable_layer_labels,causal_relations, causal_relations_windows_probs)
    
    #remove latent linked vars not in K
    latent_variables = [x for x in latent_variables if x not in linked_vars_not_K]
    
    #infer latent variables here
    inferred_data = infer_latent_variables(latent_variables,edges,variable_layer_labels,inferred_network,causal_relations,causal_relations_windows_probs,inferred_data,no_noise_data,causal_relations_windows,occurence_probabaility_D,timestamp,sampling_interval,causal_relations_directions,causal_relations_prima_directions)
                       
    
    return inferred_data
    
def label_variables(causal_relations, causal_relations_layers):

    edges, variable_layer_labels = [],[]
    #for each relationship, assign layers num to cause and layer num + 1 to effect
    for id,relation in causal_relations.items(): #1,[A,B] for example
        c_label = causal_relations_layers[id]
        c_label = relation[0] + "|" + str(c_label) #A_1
        e_label = causal_relations_layers[id] + 1
        e_label = relation[1] + "|" + str(e_label) #B_2
        edges.append((c_label,e_label))
        variable_layer_labels.extend([c_label,e_label])

    variable_layer_labels = list(set(variable_layer_labels))
    return edges,variable_layer_labels

def define_baysian_network(network,edges,occ_probs,variables_in_K,max_layer,variable_layer_labels,causal_relations,
                            causal_relations_prob):
    '''
    Inputs:
        network: defined bayesian netwrok
        edges: a list of tuples where each tuple is a cause and effect
        occurence_probabaility: a dict of all varaibales in D as keys with the occurence 
                                probability as values
        variables_in_K: variables in the knowledge base
        variable_layer_labels: a list of all variables and their layer numbers | [A_1,B_2...]
        causal_relations: dict with key as id and val list of relationship
        causal_relations_prob: probabaility of each relation , dict
    
    '''
    #add edges to network
    for cause,effect in edges:
        network.add_edge(cause,effect) #cause,effects look like A_1,B_2

    #define CPD for origin causes i.e with no parents
    origin_causes = []
    for variablex,prob in occ_probs.items():
        if prob != 0 and variablex in variables_in_K:
            for cause,_ in edges:
                cause_variable,num = cause.split("|")
                if (str(cause_variable) == variablex) and (int(num) <= max_layer+1):
                    layer = int(num)

            #add the CPD
            cpd = TabularCPD(variable = variablex+"|"+str(layer), variable_card=2, values=[[1-prob],[prob]])
            network.add_cpds(cpd)
            origin_causes.append(variablex+"|"+str(layer))

    #define CPDs for the effect variables
    network = CPD_effects(variable_layer_labels,origin_causes,edges,causal_relations,causal_relations_prob,network)

    network.check_model()
    inference_network = VariableElimination(network)

    return inference_network

def define_baysian_network_CMC(network,edges,occ_probs,variables_in_K,variable_layer_labels,causal_relations,
                            causal_relations_prob):
    '''
    Inputs:
        network: defined bayesian netwrok
        edges: a list of tuples where each tuple is a cause and effect
        occurence_probabaility: a dict of all varaibales in D as keys with the occurence 
                                probability as values
        variables_in_K: variables in the knowledge base
        variable_layer_labels: a list of all variables and their layer numbers | [A_1,B_2...]
        causal_relations: dict with key as id and val list of relationship
        causal_relations_prob: probabaility of each relation , dict
    
    '''
    
    #add edges to network
    for cause,effect in edges:
        try:
            network.add_edge(cause,effect) #cause,effects look like A_1,B_2
        except:
            pass

    #print(variables_in_K)
    #print(f"edges: {edges}")

    #define CPD for origin causes i.e with no parents
    origin_causes = []
    for variablex,prob in occ_probs.items():
        if prob != 0 and variablex in variables_in_K:
            i = 0
            #print(variablex)
            for cause,_ in edges:
                cause_variable,num = cause.split("|")
                if (str(cause_variable) == variablex):
                    if i == 0:
                        layer = int(num)
                        i = i + 1
                    else:
                        if int(num) < layer:
                            layer = int(num)
                
            #add the CPD
            cpd = TabularCPD(variable = variablex+"|"+str(layer), variable_card=2, values=[[1-prob],[prob]])
            network.add_cpds(cpd)
            origin_causes.append(variablex+"|"+str(layer))

    #define CPDs for the effect variables
    network = CPD_effects(variable_layer_labels,origin_causes,edges,causal_relations,causal_relations_prob,network)
    network.check_model()
    #print(origin_causes)
    inference_network = VariableElimination(network)

    return inference_network

def infer_latent_variables(latent_variables,edges,variable_layer_labels,D_inference_network,
                            causal_relations,causal_relations_prob,infer_data,no_noise_data,causal_relations_windows,occurence_probabaility_D,timestamp,sampling_interval,causal_relations_directions,causal_relations_prima_directions):
                            

    list_of_prob_and_evidence = {}
    for_later = []
    #print(len(timestamp))
    #construct the bayes model within the Markov blanket,Let H denotes the latent variable
    #H's markovs blanket includes H, H's causes, H's effect and H's effect's causes
    
    for var in latent_variables:
        #print(f"latent: {var}")
        var_bayes = BayesianNetwork()

        H_causes = [c for c,e in edges if var == e.split("|")[0]]
        #print(f"hcauses {H_causes}")
        H_effects = [e for c,e in edges if var == c.split("|")[0]]
        #print(f"heffects {H_effects}")
        H_effect_causes = [c for c,e in edges if e in H_effects]
        #print(f"heffectscause {H_effect_causes}")
        H_invars = [i for i in variable_layer_labels if var == i.split("|")[0]]
        #print(f"hinvars {H_invars}")
        markov_vars = H_causes + H_effect_causes + H_effects + H_invars
        markov_vars = list(set(markov_vars))
        H_edges = [edge for edge in edges if (edge[0] in markov_vars) and (edge[1] in markov_vars)]

        #Get all causes and all effects in the markov blanket
        H_all_causes = [cause for cause,_ in H_edges]
        H_all_causes = list(set(H_all_causes))
        H_all_effects = [effect for _,effect in H_edges]
        H_all_effects = list(set(H_all_effects))

        latent_inference_network = define_bayesian_network_markov(var_bayes,H_edges,markov_vars,H_all_causes,
                                            H_all_effects,H_invars,D_inference_network,causal_relations,
                                            causal_relations_prob,var)
        
        
        #added by Louis to handle root node that is a latent variable with a single effect
        #if (len(H_causes) == 0) and (len(H_invars) == 1) and (len(H_effects) == 1) and (len(H_effect_causes) == 1): #and (var in variable_type["single"]):
        #    for_later.append(var)
        #else:
        infer_data,list_of_prob_and_evidence = inference(list_of_prob_and_evidence,infer_data,var,H_edges,causal_relations,latent_variables,
                                                    markov_vars,H_invars,H_effect_causes,H_effects,no_noise_data,causal_relations_windows,latent_inference_network,
                                                    timestamp,sampling_interval,causal_relations_directions,causal_relations_prima_directions)

    for var in for_later:
        H_effects = [e for c,e in edges if var == c.split("|")[0]]
        latent_ts = {it:0 for it in timestamp}
        effect_occurence = infer_data[H_effects[0].split("|")[0]].value_counts(1)[1]
        var_occurence = occurence_probabaility_D[var]
        print(var)
        print(causal_relations)
        print(causal_relations_prob)
        tag = None
        for key,rela in causal_relations.items():
            if rela[0] == var and rela[1] == H_effects[0].split("|")[0]:
                tag = key

        #get the edge probabaility and time window
        edge_prob = causal_relations_prob[tag]
        twindow = causal_relations_windows[tag]
        prob_effect_for_cause = (var_occurence * edge_prob)/effect_occurence
        var_name = H_effects[0].split("|")[0]
        var_ts = no_noise_data.set_index('timestamp')[var_name].to_dict()
        observed_activated_timepoints = []

        for it in timestamp:
            var_active = []
            for t in range(min(timestamp[-1],it + twindow[0]),min(timestamp[-1],it + twindow[1]+sampling_interval),sampling_interval):         
                try:
                    if var_ts[t] == 1:
                        if t not in observed_activated_timepoints:
                            var_active.append(t)

                    observed_activated_timepoints.append(random.choice(var_active))
                except:
                    pass
                
                

            if len(var_active) > 0:
                if causal_relations_directions[tag] == -1:
                    latent_ts[it] = 1-prob_effect_for_cause
                else:
                    latent_ts[it] = prob_effect_for_cause

        infer_data[var]= list(latent_ts.values())
    return infer_data

def define_bayesian_network_markov(network,edges,markov_variables,causes,effects,invars,D_infer,
                                    causal_relations,causal_relations_prob,var_to_recreate):

    #add edges to network
    for cause,effect in edges:
        network.add_edge(cause,effect) #cause,effects look like A_1,B_2
    
    #define CPD for origin causes i.e with no parents
    origin_causes = []
    #print(f"markov vars {markov_variables}")
    #print(f"causes {causes}")
    #print(f"effects {effects}")
    #print(f"invars {invars}")
    #print(var_to_recreate)
    for var in markov_variables:
        #check whether it is an origin cause and not a latent variable
        #if (var in causes) and (var not in effects) and (var not in invars):
        ori_ind = False
        if (var in causes) and (var not in effects):
            ori_ind = True
            if (var_to_recreate == var.split("|")[0]):
                if len(invars) == 1:
                    ori_ind = True
                else:
                    ori_ind = False

        if ori_ind == True:
            #get its probabaility
            probs = []
            temp = (D_infer.query([var],joint=False,show_progress=False))[var]

            if math.isnan((temp.values[0])):
                probs = [1.0,0.0]
                print("error:  shouldn't happen: ")
                sys.exit(1)
            else:
                probs = [temp.values[0],temp.values[1]]
                #print("ori")
                #print(probs)
                cpd =TabularCPD(variable=var,variable_card=2,values=[[probs[0]],[probs[1]]])
                network.add_cpds(cpd)
                origin_causes.append(var)
    
    #define CPDs for the effect variables
    #print(f"origin causes {origin_causes}")
    network = CPD_effects(markov_variables,origin_causes,edges,causal_relations,causal_relations_prob,network)
    
    network.check_model()
    inference_network = VariableElimination(network)
    #re = (inference_network.query(['V4_1'],joint = False,evidence={'V1_2':0}))['V4_1'] #evidence={'V2_2':0,'V1_1':0}
    #print(re)
    

    #print(network.get_cpds())
    
    return inference_network
     
def test_latent():
    causal_model_dic = {1:["A","H"],2:["H","B"],3:["H","C"]}#,4:["D","C"]}  #relationships pairs
    causal_model_layer= {1:1, 2:2, 3:2}#, 4:2}                              #layer of each realationship
    causal_window_dic = {1:[1,2], 2:[3,4], 3:[4,5]}#, 4:[2,3]}              #time window of each relationship
    causal_window_pro_dic = {1:0.9, 2:0.9, 3:0.9}#, 4:0.9}
    latent_vars = ["H"]
    var_occur_lis = {"A":0.15,"B":0.00,"C":0.0,"H":0.00}#,"D":0.15,"E":0.01,"F":0.01}
    max_layer = 2

    raw_series = "/data/code/lgomez/Causal_Inference/Causal-Model-Combination/Simulated_Data/latentB_H.csv"
    D = pd.read_csv(raw_series)
    causal_model = {"relations":causal_model_dic, "layers":causal_model_layer, "time-windows":causal_window_dic, "probs":causal_window_pro_dic}
    #K = (causal_model_dic,causal_model_layer,causal_window_dic,causal_window_pro_dic,max_layer)
    infered_series = reconstruct_latent_variables(latent_vars,causal_model,D,var_occur_lis,max_layer)

    infered_series.to_csv("/data/code/lgomez/Causal_Inference/Causal-Model-Combination/Simulated_Data/latentB_K2.csv",index=0)

def CMC_test():
    raw_series = "/data/code/lgomez/Causal_Inference/cmc/DMITRI/race_min.csv"
    #raw_series = "/data/code/lgomez/Causal_Inference/Causal-Model-Combination/D7_infer.csv"
    raw_D = pd.read_csv(raw_series,index_col = 0)
    #pids = list(raw_D["pid"].unique())
    #raw_D = raw_D[raw_D["pid"] == pids[0]].copy(deep = True)
    #raw_D = raw_D.drop_duplicates(subset=['timestamp'], keep='first')
    latent_variables = ['int_exe',"rest","mod_exe"] #list of all latent vars - both non-linked and linked
    linked_vars_K = ['int_exe',"rest"] #list of only linked vars with relations
    linked_vars_not_K = ["mod_exe"] #list of linked vars not in K
    linked_vars_group = [('int_exe',"rest","mod_exe")] # a list of tuples where each tuple contains the associated linked latent variables
    linked_vars_group_with_rela = [('int_exe',"rest")] # a list of tuples where each tuple contains only associated linked latent variables in K
    sampling_interval = 900


    K = {'relations': {1: ['int_exe', 'glu_hyper'], 2:["rest", "glu_normal"]}, 
            'time-windows': {1: [900, 3600], 2:[900, 1800]}, 
            'probs': {1: 0.93, 2: 0.53}}

    variable_occurence_probability = {'int_exe': 0.019352, 'mod_exe': 0.15, 'rest':0.82, 'glu_hyper':0.0, "glu_normal":0.0}
    causal_model = nx.DiGraph()  
    #causal_model.add_edges_from([('V4','V1'),("V6","V2"),("V5","V2")])
    causal_model.add_edges_from([('int_exe', 'glu_hyper'),("rest", "glu_normal")])

    #print(raw_D)
    pids = list(raw_D["pID"].unique())
    df_prob,df_bin = pd.DataFrame(),pd.DataFrame()
    for pid in pids:
        print(pid)
        raw_pid = raw_D[raw_D["pID"] == pid].copy(deep = True)
        raw_pid.reset_index(drop = True, inplace = True)
        inferred_series = reconstruct_latent_variables_CMC(latent_variables,K,raw_pid,variable_occurence_probability,causal_model,linked_vars_not_K,sampling_interval)
        df_prob = pd.concat([df_prob,inferred_series], ignore_index = True)
        inferred_series = latent_var_post_processing(inferred_series,latent_variables, linked_vars_K,linked_vars_group,linked_vars_group_with_rela)
        df_bin = pd.concat([df_bin,inferred_series], ignore_index = True)

    df_prob.to_csv("/data/code/lgomez/Causal_Inference/cmc/sim_data/race_recon_prob.csv",index = 0)
    df_bin.to_csv("/data/code/lgomez/Causal_Inference/cmc/sim_data/race_recon.csv",index = 0)

    #print(inferred_series["meal_true"].value_counts())

def latent_var_post_processing(inferred_data,latent_variables,linked_vars,linked_vars_group,linked_vars_group_with_rela):
    '''Inputs:
    inferred_data: the outputted dataframe from reconstruction
    latent_variables: a list latent variables including non-linked and linked variables!
    linked_vars: list of all possible linked variables
    linked_vars_group: a list of tuples where each tuple contains the connected linked variables
    linked_vars_group-with_rela: a list of tuples where each tuple contains the connected linked
    variables with relationships in K
    '''

    used = []

    for lat_var in latent_variables:
        #1. non-linked vars
        if lat_var not in linked_vars:
            lat_ts = inferred_data[lat_var].to_numpy()
            updated_lat_ts = []

            for prob in lat_ts:
                updated_lat_ts.append(ss.bernoulli.rvs(prob,0))

            inferred_data[lat_var] = updated_lat_ts
            used.append(lat_var)

        else:
            if lat_var not in used:
                linked_group = [x for x in linked_vars_group if lat_var in x][0]
                linked_group_with_rela = [x for x in linked_vars_group_with_rela if lat_var in x][0]
                df_linked = inferred_data[list(linked_group)].copy(deep = True)
                
                if len(linked_group) == len(linked_group_with_rela):
                    #1. all linked vars have relationship
                    active_forms = df_linked.apply(resolve_linked,args=(1,), axis = 1)
                else:
                    #2. not all linked vars have a relationship
                    #print("in here")
                    active_forms = df_linked.apply(resolve_linked,args=(2,), axis = 1)
                    #print(active_forms)

                df_linked = df_linked * 0
                for i in range(0,len(active_forms)):
                    if active_forms[i] == 0:
                        pass
                    else:
                        df_linked.at[i,active_forms[i]] = 1

                for var in linked_group:
                    inferred_data[var] = df_linked[var]
                    used.append(var)
                    
    return inferred_data

def resolve_linked(row,cond):
    names = list(row.index)
    values = list(row.values)
    
    if cond == 1:
        values = list(softmax(values))
        max_value = np.nanmax(values)
        max_index = values.index(max_value)
        max_var = names[max_index]
        return max_var
    
    elif cond == 2:
        #get the max values
        #print("here too")
        #print(values)
        max_value = np.nanmax(values)
        #print(max_value)
        if max_value > 0.5:
            #assumptions here is that only two linked form are non-zero and one should be greater than 0.5
            max_index = values.index(max_value)
            max_var = names[max_index]
            return max_var
        else:
            return 0


if __name__ == "__main__":
    CMC_test()



            

            


    


