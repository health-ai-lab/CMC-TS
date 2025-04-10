import networkx as nx

def is_proper_subset(vars1,vars2,CG):
    '''Function to check if model1 is a subset of model2'''
    '''Inputs:
            vars1,vars2: variable groups of two causal models, vars 2 should be bigger
            
      Ouptut: Ind: bool; ouputs true or false depending on validity of statement
    
    '''
   
    latent_variables = list(set(vars2) - set(vars1))
    '''Using set operations'''
    if len(vars2) > len(vars1):
        Ind = set(vars1) < set(vars2)
        if Ind == True:
            #check to see if all latent variables have relationships. if so then a causal proper subset, if not false
            vars_in_y_with_no_edges = list(nx.isolates(CG[vars2]))
            latent_vars_with_no_edges = [v for v in latent_variables if v in vars_in_y_with_no_edges]
            if len(latent_vars_with_no_edges) != 0:
                Ind = False
            else:
                Ind = True
        else:
            ind = False
    else:
        Ind = False
              
    return Ind

def mod_tricube(num_latent_vars,num_of_variables):
    '''Function to give weight based on the number of latent varaibles to reconstruct'''
    '''Inputs:
            num_latent_vars: the number of latent variables
        
       Outputs: the weight in the range [0,1.0]
    '''
    u = num_latent_vars/num_of_variables
    k_u = (70.0/81.0)*((1-abs(u)**3)**3) #original code for tricube kernel
    rescaled_k_u = (81.0/70.0) * k_u
    return rescaled_k_u

def score_latent_variables(latent_vars,model2):
    '''Function to give a score between 0 to 0.5 for the nature of latent variables; We weight parents nodes higher than child nodes'''
    '''Inputs:
            latent_vars: the causal model we are reconstructing latent variables for
            model 2: the causal model used for reconstructing as the knowledge base
       Outputs: weight in the range [0 to 0.5]
    '''
    num_H = len(latent_vars)
    score = 0
    max_score = num_H * 5

    for var in latent_vars:
        pred = list(model2.predecessors(var))
        succ = list(model2.successors(var))
        
        if len(pred) != 0 and len(succ) == 0: #leaf node
            score += 1

        elif len(succ) >= 1: #a parent node
            if len(pred) == 0 and len(succ) == 1: #origin cause with only one child
                score += 3
            elif len(pred) == 0 and len(succ) >= 2: #orign cause and common cause for at least two variables
                score += 4
            elif len(pred) != 0 and len(succ) == 1: #non-orogin node with one child
                score += 2
            elif len(pred) != 0 and len(succ) >= 2: #non-origin node and common cause
                score += 5
        
    #num. of latent varibales that are parents nodes in model1/total number of latent variables
    final_score = score/max_score
    return final_score