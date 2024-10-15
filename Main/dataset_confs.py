import os

case_id_col = {}
activity_col = {}
resource_col = {}
timestamp_col = {}
label_col = {}


dynamic_cat_cols = {}
static_cat_cols = {}
dynamic_num_cols = {}
static_num_cols = {}
filename = {}

resource_multitasking_col = {}
totaltimeofcase_col = {}

import main

logs_dir = main.logs_dir



bpic2017_dict=main.bpic2017_dict

for dataset, fname in bpic2017_dict.items():

    filename[dataset] = os.path.join(logs_dir, fname)

    case_id_col[dataset] = "Case ID"
    activity_col[dataset] = "Activity"
    resource_col[dataset] = 'org:resource'
    if ("BPIC11" in dataset):
        activity_col[dataset] = "Activity code"
        resource_col[dataset] = "Producer code"
    timestamp_col[dataset] = 'time:timestamp'
    label_col[dataset] = "label"

    resource_multitasking_col[dataset] = "resource_multitasking"
    totaltimeofcase_col[dataset] = "TotalTimeOFCase"


    dynamic_cat_cols[dataset] = main.dynamic_cat_cols
    static_cat_cols[dataset] = main.static_cat_cols
    dynamic_num_cols[dataset] = main.dynamic_num_cols
    if (main.preprossing_mod == "regression"):
        static_num_cols[dataset] = main.static_num_cols_regression

    if (main.preprossing_mod == "classification"):
        static_num_cols[dataset] = main.static_num_cols_classification