import optimize_params_reg, main
import experiments_reg
from IPython.lib.display import exists
import multiprocessing
import os
import time
"This file serves to run bulk executions based on different configurations of PPM models."

"For running the Optimize_params module, use 'param' as the input, and for running the experiment module, use 'performance' as the input."
Execution_mode = 'performance' # input: 'param' 'performance'
if Execution_mode == 'param':
    start = time.perf_counter()
    processes = []
    for classification_method in ["svm", "DecisionTree", "rf" , "xgboost"]:  # input: "DecisionTree", "rf" "xgboost", "svm",
        for encoding_method in ['agg', "laststate","combined", "previous", "index"]:  # input: 'agg', "laststate","combined", "previous","index",
            for bucket_method in ["prefix", "cluster" , "single", "state"]:  #  input: "prefix" ,"cluster" , "single" ,"state",
                if (exists(os.path.join(main.logs_dir, "param_optim_all_trials_%s_%s_%s.csv" % (
                        classification_method, f"{main.filenames[:-4]}_final", "%s_%s_%s_%s" % (bucket_method, encoding_method,main.preprossing_mod, main.inter_intra_mode))))):
                    print(
                        f"param_optim_all_trials_{classification_method}_{main.filenames[:-4]}_final_{bucket_method}_{encoding_method}_{main.preprossing_mod}.csv  is exist :\ ")

                else:
                    try:
                        print(
                            f"optimization_param model is running using classification method: {classification_method}, encoding method: {encoding_method} and bucketig method: {bucket_method} ....")
                        p = multiprocessing.Process(target= optimize_params_reg.running_func, args=[bucket_method,encoding_method, classification_method, 30, main.inter_intra_mode])
                        p.start()
                        processes.append(p)
                    except:
                        print(
                            f"optimization_param model using classification method: {classification_method}, encoding method: {encoding_method} and bucketig method: {bucket_method} has not been completely run :(")
                        pass

    for process in processes:
        process.join()

    finish=time.perf_counter()
    print(f"final time with multi processing: {finish-start}")
if Execution_mode == 'performance':
    processes2 = []
    for classification_method in ["svm","DecisionTree" ,"rf","xgboost"]: # input: "DecisionTree" ,"rf","xgboost",,"svm"
        for bucket_method in ["prefix","state", "single", "cluster"]: #input:"prefix","state", "single" , "cluster"
            for encoding_method in ["agg", "combined", "previous", "laststate", "index"]:  #input: "index"  "agg", "combined", "previous", "laststate",

                if (exists(os.path.join(main.logs_dir, "performance_results_%s_%s_%s_%s_%s_gap%s.csv" % (
                classification_method, f"{main.filenames[:-4]}_final", "%s_%s" % (bucket_method, encoding_method),main.preprossing_mod, main.inter_intra_mode, main.gap)))):
                    print(
                        f"performance_results_{classification_method}_{main.filenames[:-4]}_final_{bucket_method}-{encoding_method}_gap{main.gap}.csv is exist :\ ")
                else:
                    try:
                        print(
                            f"experiment model is running using classification method: {classification_method}, encoding method: {encoding_method} and bucketig method: {bucket_method} ....")
                        experiments_reg.running_func(bucket_method_in=bucket_method, cls_method_in=classification_method,
                                                 cls_encoding_in=encoding_method, inter_intra_mode_in=main.inter_intra_mode)
                        p = multiprocessing.Process(target=experiments_reg.running_func,
                                                    args=[bucket_method, encoding_method, classification_method,
                                                          main.gap, 1, main.inter_intra_mode])
                        p.start()
                        processes2.append(p)
                        print(
                            f"experiment model is running using classification method: {classification_method}, encoding method: {encoding_method} and bucketig method: {bucket_method} and mode {main.inter_intra_mode}: is completed :)")
                    except:
                        print(
                            f"experiment model using classification method: {classification_method}, encoding method: {encoding_method} and bucketig method: {bucket_method} and mode: {main.inter_intra_mode} has not been completely run :(")
                        pass


    for process in processes2:
        process.join()