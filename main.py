import sys
from pathlib import Path
import glob as glob
import pandas as pd
import pickle

from CMC_TS import CMC

if __name__ == "__main__":

    dataset_path = sys.argv[1]
    sig_level = float(sys.argv[2])
    fdr_ind = sys.argv[3] #whether to perform multiple hypothesis correction
    min_lag = int(sys.argv[4])
    max_lag = int(sys.argv[5])
    lag_interval = int(sys.argv[6]) #interval of time in seconds - diabetes is 5minutes = 5* 60 here it is 60
    probability_modulator = float(sys.argv[7])
    max_level = int(sys.argv[8])
    num_datasets = int(sys.argv[9])
    effects_to_test = sys.argv[10]
    causes_to_exclude = sys.argv[11]
    linkedvarsgroup = sys.argv[12]


    global CMC_stages_log
    CMC_stages_log = {}
    

    Path(dataset_path + "/results/").mkdir(parents=True, exist_ok=True)
    results_path = dataset_path + "/results/"
    list_of_datasets, list_of_variables_measured,all_vars = [],[],[]
    all_datasets = glob.glob(dataset_path + "/data/" + "*.csv")
    
    for dataset in all_datasets:
        df = pd.read_csv(dataset,index_col = 0)
        #print(dataset)
        variables_measured = list(df.columns)
        print(variables_measured)
        variables_measured.remove("pID")
        variables_measured.remove("timestamp")

        list_of_datasets.append(df)
        list_of_variables_measured.append(variables_measured)
        all_vars.extend(variables_measured)


    all_vars = list(set(all_vars))
    total_number_of_variables = len(all_vars)
    linkedvarsgroup = [tuple(x.split("|")) for x in linkedvarsgroup.split("#")]
    effects_to_test = effects_to_test.split("|")
    causes_to_exclude = causes_to_exclude.split("|")

    pre_load = False
    level = 0
    stage_end = False
    start_level = 0
    level = start_level

    while stage_end == False:
        print(f"=============Level {level}=============")
        print("\n")
        
        
        stage = CMC(level,total_number_of_variables,effects_to_test,causes_to_exclude,linkedvarsgroup) #define the class object
        if level == 0 and pre_load == False:
            #add datasets and variables measured
            stage.add_datasets(list_of_datasets)
            stage.add_variable_sets(list_of_variables_measured)

        '''Step 1: Find Causal relationships for each dataset'''  
        if pre_load == False:
            if level == 0:
                stage.find_causal_relationships(min_lag,max_lag,lag_interval,results_path)
            else:
                #pass in augmented dataset list, so that we do not re-run Sk for datasets that have not been augmented to save run time
                stage.update_CMC(updated_datasets,updated_variables,updated_recon_info)
                stage.find_causal_relationships(min_lag,max_lag,lag_interval,results_path,augmented_datasets)

            #save current progress - temp
            res = {"dataset": stage.D, "variables": stage.V, "SK": stage.SK, "SK_sig": stage.SK_sig, "CG": stage.CG, 
                "Var-group-ids": stage.variable_groups_and_id, "Recon-info":stage.reconstructed_info}
            CMC_stages_log[level] = res
            with open(results_path + "CMC_pre-stage_" + str(level) + ".pickle",'wb') as f:
                pickle.dump(res,f,pickle.HIGHEST_PROTOCOL)
                
        else:
            if start_level == level:
                #Load saved progress at the specified "start_level"
                stage_info = pd.read_pickle(results_path + "CMC_pre-stage_" + str(start_level) + ".pickle")
                stage.update_CMC(stage_info["dataset"],stage_info["variables"],stage_info["Recon-info"])
                stage.SK = stage_info["SK"]
                stage.CG = stage_info["CG"]       
                stage.variable_groups_and_id = stage_info["Var-group-ids"] 
                stage.SK_sig = stage_info["SK_sig"]
                stage.reconstructed_info = stage_info["Recon-info"]
                stage.LinkedVarsGroup = linkedvarsgroup
            
            else:
                #pass in augmented dataset list, so that we do not re-run Sk for datasets that have not been augmented to save run time
                stage.update_CMC(updated_datasets,updated_variables,updated_recon_info)
                stage.find_causal_relationships(min_lag,max_lag,lag_interval,results_path,augmented_datasets)
                #save current progress
                res = {"dataset": stage.D, "variables": stage.V, "SK": stage.SK, "SK_sig": stage.SK_sig, "CG": stage.CG, 
                    "Var-group-ids": stage.variable_groups_and_id, "Recon-info":stage.reconstructed_info}
                CMC_stages_log[level] = res
                with open(results_path + "CMC_pre-stage_" + str(level) + ".pickle",'wb') as f:
                    pickle.dump(res,f,pickle.HIGHEST_PROTOCOL)
                
        
        
        '''Step 2: Cluster Variable Sets, perform wfisher and create graphs for each variable group'''
        stage.cluster_variable_sets(sig_level)

        '''Step 3. Check if we are at the end of CMC's run before processing forward'''
        if (stage.end == True) or (level == max_level):
            print("CMC Completed")
            print(f"Final causal relationships are: {stage.SK_sig}")
            stage_end = True
        
        else:
            '''Step 4: Rank and Score causal models'''
            rank_and_score_results = stage.rank_and_score(total_number_of_variables)
            try:
                rank_and_score_results.sort_values(by=["Score"],ascending = False, inplace = True,ignore_index = True)
            except:
                pass
            
            rank_and_score_results.to_csv(results_path + "rank_and_score_level_" + str(level) + ".csv")
            

            if len(rank_and_score_results) == 0:
                print("No other combination is possible. Final Causal relationships are: ")
                stage_end = True
            else:
                print("\n")
                print(rank_and_score_results)
                print("\n")
                recon_results,used_as_K,used_as_base = [],[],[]

                '''Step 5: Reconstruct Missing Variables'''
                #If there a proper subset, reconstruct those, else reconstruct only the top scoring combination
                if rank_and_score_results["p_subset"].sum() > 0:
                    rank_and_score_results = rank_and_score_results[rank_and_score_results["p_subset"] == 1].copy(deep = True)
                else:
                    rank_and_score_results = pd.DataFrame(rank_and_score_results.iloc[0]).transpose()
               
                for row,val in rank_and_score_results.iterrows():
                    #we dont want to use a var group as K and also reconstruct it in the same stage
                    if val["Base"] not in used_as_K:
                        if val["K"] not in used_as_base:
                            recon_results.extend(stage.perform_reconstruction(val["Base"],val["K"],val["Latent-vars"],val["p_subset"]))
                            used_as_K.append(val["K"])
                            used_as_base.append(val["Base"])

                print("Reconstruction Complete")
                print("\n")

                ids_and_recon_datasets = {d_id:d for d_id,d in recon_results}
                augmented_datasets = [d_id for d_id,_ in recon_results]
                current_datasets = stage.get_datasets()
                current_variables = stage.get_variable_sets()
                updated_datasets, updated_variables= {},{}
                updated_recon_info = stage.reconstructed_info

                #save all progress
                res = {"dataset": stage.D, "variables": stage.V, "SK": stage.SK, "SK_sig": stage.SK_sig, "CG": stage.CG, 
                    "Var-group-ids": stage.variable_groups_and_id, "Recon-info":updated_recon_info}
                CMC_stages_log[level] = res
                with open(results_path + "CMC_post-stage_" + str(level) + ".pickle",'wb') as f:
                    pickle.dump(res,f,pickle.HIGHEST_PROTOCOL)

                level = level + 1

                #for each id in the stage, check if it is to be updated
                for dataset_id,dataset in current_datasets.items():
                    i = dataset_id.split("_")[1] #level_i | here we get the i

                    if dataset_id in ids_and_recon_datasets.keys():
                        updated_datasets[str(level) + "_" + i] = ids_and_recon_datasets[dataset_id] #this is a dictionary
                        new_columns = list(ids_and_recon_datasets[dataset_id].columns)
                        new_columns.remove("pID")
                        new_columns.remove("timestamp")
                        new_columns.remove("dID")
                        new = sorted(new_columns)
                        print(new)
                        updated_variables[str(level) + "_" + i] = new
                    else:
                        #carry over the current datasets and variables but update the ids
                        updated_datasets[str(level) + "_" + i] = dataset
                        updated_variables[str(level) + "_" + i] = current_variables[dataset_id]



    #Final Step
    all_df = pd.DataFrame()
    for var_group,df in stage.SK_sig.items():
        df = df[df["edge-prob"] >= probability_modulator]
        all_df = pd.concat([all_df,df],ignore_index = True)
    all_df.sort_values(by=["c-e-ws-we","combined-p-val"],ascending=[True,True],ignore_index= True,inplace = True)
    #all_df.drop_duplicates(subset=["c-e"], keep = "first", ignore_index = True, inplace = True)
    all_df.to_csv(results_path + "CMC_relationships.csv")