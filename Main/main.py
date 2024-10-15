import os.path

import pandas as pd

"** This file serves as the control unit for conducting experiments and configuring models.**"


"For considering both inter and intra case features, use 'inter+intra'; for intra-only features, use 'intra'"
inter_intra_mode = 'inter+intra'  # 'intra' 'inter+intra'

if inter_intra_mode == 'inter+intra':
    input_data_folder = "For inter+intra case features, assign the location as 'C:/Users/...'"
if inter_intra_mode == 'intra':
    input_data_folder = "For intra-only features, assign the location as 'C:/Users/...'."
output_data_folder = input_data_folder
params_dir = input_data_folder
"Uncomment the dataset you want to use and comment out the other"
# filenames= ["BPIC17_Accepted.csv"][0]
# filenames = ["BPIC17_Canceled.csv"][0]
filenames = ["BPIC15_%s.csv"%(municipality) for municipality in range(1,2)][0]
# filenames = ["BPIC15_%s.csv"%(municipality) for municipality in range(2,3)][0]
# filenames = ["BPIC15_%s.csv"%(municipality) for municipality in range(3,4)][0]
# filenames = ["BPIC15_%s.csv"%(municipality) for municipality in range(4,5)][0]
# filenames = ["BPIC15_%s.csv"%(municipality) for municipality in range(5,6)][0]
# filenames = ["BPIC11.csv"][0]


preprossing_mod = "regression"

# dataset_ref
if ("BPIC17" in filenames):
    dataset_ref = "BPIC17"
if ("BPIC15" in filenames):
    dataset_ref = "BPIC15"
if ("BPIC11" in filenames):
    dataset_ref = "BPIC11"
if ("BPIC12" in filenames):
    dataset_ref = "BPIC12"

# where result of parameter optimization will be stored
results_dir = params_dir
# where logs are stored
logs_dir = params_dir


# The following configuration is for manually running Param Optimal or Experiment files.
# For bulk runs, adjust the settings in the Running Program Module.

# which bucket method from [cluster,prefix ,state ,prefix , single]
bucket_method = "prefix"
# which encoding method from [laststate, agg, index,combined ,previous] # "previous" is my extension
cls_encoding = "agg"
# which classification method from [rf, xgboost, logit, svm] #DecisionTree
cls_method = 'xgboost'
# number of gap in prefix length bucketing
gap = 2
# number of iteration in parameter optimization
n_iter = 1



# resource experiment parameters
other_resource_experience = ['n_tasks', 'n_cases', 'n_acts', 'n_handoffs', 'ent_act',
                             'ent_case', 'ent_handoff', 'ratio_act_case', 'busyness',
                             'n_current_case', 'n_current_act', 'n_current_handoff',
                             'ratio_current_case', 'ratio_current_act', 'ratio_current_handoff']

# defining categorical an numerical columns
if ("BPIC17" in filenames):
    dynamic_cat_cols = ["Activity", 'org:resource', 'EventOrigin', 'lifecycle:transition',
                        "Accepted", "Selected"]  # i.e. event attributes
    static_cat_cols = ['ApplicationType', 'LoanGoal']  # i.e. case attributes that are known from the start
    if inter_intra_mode == 'inter+intra':
        dynamic_num_cols = ['FirstWithdrawalAmount', 'MonthlyCost', 'NumberOfTerms', 'OfferedAmount', 'CreditScore',
                        "timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr", "month", "weekday",
                        "hour", "open_cases", "resource_multitasking"] + other_resource_experience
    if inter_intra_mode == 'intra':
        dynamic_num_cols = ['FirstWithdrawalAmount', 'MonthlyCost', 'NumberOfTerms', 'OfferedAmount', 'CreditScore',
                            "timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr", "month",
                            "weekday","hour"]

    static_num_cols_classification = ['RequestedAmount', "TotalTimeOFCase"]
    static_num_cols_regression = ['RequestedAmount']

if ("BPIC15" in filenames):
    dynamic_cat_cols = ["Activity", "monitoringResource", "question", "org:resource"]  # i.e. event attributes
    static_cat_cols = ["case:Responsible_actor"]  # i.e. case attributes that are known from the start
    if inter_intra_mode == 'inter+intra':
        dynamic_num_cols = ["timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr", "month",
                            "weekday",
                            "hour", "open_cases", "resource_multitasking"] + other_resource_experience
    if inter_intra_mode == 'intra':
        dynamic_num_cols = ["timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr", "month", "weekday",
                        "hour"]  #
    static_num_cols_classification = ['case:SUMleges']  #, "TotalTimeOFCase" ,"Aanleg (Uitvoeren werk of werkzaamheid)"	,"Bouw", "Brandveilig gebruik (melding)",	"Brandveilig gebruik (vergunning)",	"Gebiedsbescherming",	"Handelen in strijd met regels RO",	"Inrit/Uitweg",	"Kap",	"Milieu (melding)"	"Milieu (neutraal wijziging)",	"Milieu (omgevingsvergunning beperkte milieutoets)",	"Milieu (vergunning)",	"Monument",	"Reclame",	"Sloop"

    static_num_cols_regression = ['case:SUMleges']

if ("BPIC11" in filenames):
    dynamic_cat_cols = ["Activity code", "Producer code", "Section", "Specialism code", "org:group"]
    static_cat_cols = ["case:Diagnosis", "case:Treatment code", "case:Diagnosis code", "case:Specialism code",
                       "case:Diagnosis Treatment Combination ID"]
    if inter_intra_mode == 'inter+intra':
        dynamic_num_cols = ["Number of executions", "timesincelastevent", 'timesincecasestart', "timesincemidnight",
                            "month", 'weekday', 'hour', "open_cases",
                            "resource_multitasking"] + other_resource_experience  # ,
    if inter_intra_mode == 'intra':
        dynamic_num_cols = ["Number of executions", "timesincelastevent", 'timesincecasestart', "timesincemidnight",
                            "month", 'weekday', 'hour']
    static_num_cols_classification = ["case:Age"]
    static_num_cols_regression = ["case:Age"]

# number of bunch in discretizing for reducing the size of dataset if it is needed
number_of_bunch0 = 10  # for initial labeling in data preprocessing
dataset_ref_to_datasets = {
    "BPIC11": [f"{filenames[:-4]}_final"],
    "BPIC15": [f"{filenames[:-4]}_final"],
    "insurance": ["insurance_activity", "insurance_followup"],
    "sepsis_cases": ["sepsis_cases_1", "sepsis_cases_2", "sepsis_cases_4"],
    "BPIC17": [f"{filenames[:-4]}_final"],
}

# dataset_confs parameter
if ((preprossing_mod == "regression") & ("BPIC17" in filenames)):
    bpic2017_dict = {"BPIC17_Accepted_final": "BPIC17_Accepted_Final_Data_File_Reg.csv",
                     "BPIC17_Canceled_final": "BPIC17_Canceled_Final_Data_File_Reg.csv"}  # my extension



if ((preprossing_mod == "regression") & ("BPIC15" in filenames)):
    bpic2017_dict = {"BPIC15_1_final": "BPIC15_1_Final_Data_File_Reg.csv",
                     "BPIC15_2_final": "BPIC15_2_Final_Data_File_Reg.csv",
                     "BPIC15_3_final": "BPIC15_3_Final_Data_File_Reg.csv",
                     "BPIC15_4_final": "BPIC15_4_Final_Data_File_Reg.csv",
                     "BPIC15_5_final": "BPIC15_5_Final_Data_File_Reg.csv"}  # my extension


if ((preprossing_mod == "regression") & ("BPIC11" in filenames)):
    bpic2017_dict = {"BPIC11_final": "BPIC11_Final_Data_File_Reg.csv"}






"For selecting the optimal parameters in prefix length-based bucketing to conduct experiments"
def optimal_prefixbased_param(cls_method_in=cls_method, dataset_ref_in=dataset_ref, bucket_method_in=bucket_method,
                              cls_encoding_in=cls_encoding):
    method_name = "%s_%s" % (bucket_method_in, cls_encoding_in)
    dataset_name = [dataset_ref_in] if dataset_ref_in not in dataset_ref_to_datasets else dataset_ref_to_datasets[
        dataset_ref_in]
    inputfile = os.path.join(params_dir, "param_optim_all_trials_%s_%s_%s_%s_%s.csv" % (
    cls_method_in, dataset_name[0], method_name, preprossing_mod, inter_intra_mode))
    data_in = pd.read_csv(inputfile, sep=";")
    data_in.drop(data_in[data_in['param'] == "processing_time"].index, inplace=True)
    data_in = data_in.astype(str)
    data_in = data_in.astype({"nr_events": int, "value": float, 'MAE': float})
    param = data_in["param"].unique()

    data2 = data_in.groupby("nr_events", as_index=False)[["nr_events", 'param', "value", 'MAE']].apply(
        lambda x: x.sort_values("MAE", ascending=True, kind="mergesort").head(len(param))
    )
    data2.sort_values("nr_events")
    nr_of_event = data2["nr_events"].unique()
    optimal_params = {
        i: {data2["param"].loc[index1:index1].values[0]: data2["value"].loc[index1:index1].values[0] for index1 in
            data2[
                data2["nr_events"] == i].index} for i in nr_of_event}

    return optimal_params

