import pandas as pd
import numpy as np
import os
import sys
from numpy import average
import main
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from scipy import stats
import time
from datetime import datetime

"This file savers as preprocessing for prepairing the BPIC 2017 dataset"
input_data_folder = main.input_data_folder
output_data_folder = main.output_data_folder
filenames = main.filenames
if type(filenames)==str:
    filenames=[filenames]

case_id_col = "Case ID"
activity_col = "Activity"
resource_col = 'org:resource'
timestamp_col = 'time:timestamp'
label_col = "label"
pos_label = "deviant"
neg_label = "regular"

resource_multitasking_col = "resource_multitasking"
totaltimeofcase_col = "TotalTimeOFCase"



if (main.preprossing_mod=="regression"):
    relevant_offer_events = ["Final_Data_File_Reg"]




resource_freq_threshold = 10
max_category_levels = 10

# features for classifier

dynamic_cat_cols = [activity_col, resource_col, 'EventOrigin', 'lifecycle:transition',
                    "Accepted", "Selected"]  # i.e. event attributes
static_cat_cols = ['ApplicationType', 'LoanGoal']  # i.e. case attributes that are known from the start
dynamic_num_cols = ['FirstWithdrawalAmount', 'MonthlyCost', 'NumberOfTerms', 'OfferedAmount', 'CreditScore',
                    "timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr", "month", "weekday",
                    "hour"]

if (main.preprossing_mod=="regression"):
    static_num_cols = ['RequestedAmount']

if (main.preprossing_mod=="classification"):
    static_num_cols = ['RequestedAmount', totaltimeofcase_col]

static_cols = static_cat_cols + static_num_cols + [case_id_col, label_col]
dynamic_cols = dynamic_cat_cols + dynamic_num_cols + [timestamp_col]
cat_cols = dynamic_cat_cols + static_cat_cols



def extract_timestamp_features(group):

    group = group.sort_values(timestamp_col, ascending=False, kind='mergesort')

    tmp = group[timestamp_col] - group[timestamp_col].shift(-1)
    tmp = tmp.fillna("0 days 00:00:00.000000")
    group["timesincelastevent"] = tmp.apply(lambda x: float(x / np.timedelta64(1, 'm')))  # m is for minutes

    tmp = group[timestamp_col] - group[timestamp_col].iloc[-1]
    tmp = tmp.fillna("0")
    group["timesincecasestart"] = tmp.apply(lambda x: float(x / np.timedelta64(1, 'm')))  # m is for minutes

    # calculating total time of cases
    temp = group[timestamp_col].iloc[0] - group[timestamp_col].iloc[-1]
    group["TotalTimeOFCase"] = (temp / np.timedelta64(1, 'm'))

    group = group.sort_values(timestamp_col, ascending=True, kind='mergesort')
    group["event_nr"] = range(1, len(group) + 1)


    return group


def get_open_cases(date):
    return sum((dt_first_last_timestamps["start_time"] <= date) & (dt_first_last_timestamps["end_time"] > date))


def get_multitasking_recources(input):
    temp3 = temp1.loc[temp1['resource'] == input.iloc[0]]

    return sum((temp3['start_working'] <= input['time:timestamp']) & (
            temp3['end_working'] >= input['time:timestamp']))

def detect_and_remove_outlier(dataset_in):

    data = dataset_in
    Q1 = np.percentile(data["TotalTimeOFCase"], 25,
                       interpolation='midpoint')

    Q3 = np.percentile(data["TotalTimeOFCase"], 75,
                       interpolation='midpoint')
    IQR = Q3 - Q1
    # Upper bound
    upper = np.where(data["TotalTimeOFCase"] >= (Q3 + 1.5 * IQR))
    # Lower bound
    lower = np.where(data["TotalTimeOFCase"] <= (Q1 - 1.5 * IQR))

    data2=data.drop(upper[0])
    data2=data2.drop(lower[0])

    return data2


def ent(data, col):
    p_data = 1.0 * data[col].value_counts() / len(data)  # calculates the probabilities
    entropy = stats.entropy(p_data)  # input probabilities to get the entropy
    return entropy


def get_prev_resource_and_event_nr(group):
    group[handoff_col] = group[resource_col].shift(1)  # This means group[handoff_col][t] = group[resource_col][t-1]
    group["event_nr"] = range(1, len(group) + 1)
    group["is_last_event"] = False
    group["is_last_event"].iloc[-1] = True
    return (group)


def extract_experience(gr):
    group = gr.copy()
    group = group.reset_index()
    group["n_tasks"] = 0
    group["n_cases"] = 0
    group["n_acts"] = 0
    group["n_handoffs"] = 0
    group["ent_act"] = 0
    group["ent_case"] = 0
    group["ent_handoff"] = 0
    group["ratio_act_case"] = 0
    group["n_current_case"] = 0
    group["n_current_act"] = 0
    group["n_current_handoff"] = 0
    group["ratio_current_case"] = 0
    group["ratio_current_act"] = 0
    group["ratio_current_handoff"] = 0
    # group["polarity_current_act"] = 0
    group["busyness"] = 0
    for col in act_freq_cols_sum:
        group[col] = 0
    for col in act_freq_cols_ratio:
        group[col] = 0
    for col in handoff_freq_cols_sum:
        group[col] = 0
    for col in handoff_freq_cols_ratio:
        group[col] = 0

    if recent_days is not None:
        group["n_tasks_recent"] = 0
        group["n_cases_recent"] = 0
        group["n_acts_recent"] = 0
        group["n_handoffs_recent"] = 0
        group["ent_act_recent"] = 0
        group["ent_case_recent"] = 0
        group["ent_handoff_recent"] = 0
        group["ratio_act_case_recent"] = 0
        group["n_current_case_recent"] = 0
        group["n_current_act_recent"] = 0
        group["n_current_handoff_recent"] = 0
        group["ratio_current_case_recent"] = 0
        group["ratio_current_act_recent"] = 0
        group["ratio_current_handoff_recent"] = 0
        # group["polarity_current_act_recent"] = 0
        group["busyness_recent"] = 0

    start_idx = 0
    start_time = group.iloc[0][timestamp_col]
    idx = 0

    for _, row in group.iterrows():  # The row here means each row in the group data.

        all_prev_exp = group.iloc[:(idx + 1)]  # Extracts 1th row to (idx+1)th row of group data
        if recent_days is not None:
            while (row[timestamp_col] - start_time).days > recent_days:
                start_idx += 1
                start_time = group.iloc[start_idx][timestamp_col]
            recent_prev_exp = group.iloc[start_idx:(idx + 1)]

        n_tasks = len(all_prev_exp)
        n_cases = len(all_prev_exp[case_id_col].unique())
        n_acts = len(all_prev_exp[activity_col].unique())
        n_handoffs = len(all_prev_exp[handoff_col].unique())
        ent_act = ent(all_prev_exp, activity_col)
        ent_case = ent(all_prev_exp, case_id_col)
        ent_handoff = ent(all_prev_exp, handoff_col)
        d = (all_prev_exp[timestamp_col].max() - all_prev_exp[
            timestamp_col].min()).days  # Extract number of days between the max timestamp and min timestamp
        busyness = n_tasks / d if d else 0

        if row[case_id_col] == 2771451:
            print(row[timestamp_col])
            print(n_tasks)



        ratio_act_case = n_tasks / n_cases

        n_current_case = len(all_prev_exp[all_prev_exp[case_id_col] == row[case_id_col]])
        n_current_act = len(all_prev_exp[all_prev_exp[activity_col] == row[activity_col]])
        n_current_handoff = len(all_prev_exp[all_prev_exp[handoff_col] == row[handoff_col]])
        ratio_current_case = n_current_case / n_tasks
        ratio_current_act = n_current_act / n_tasks
        ratio_current_handoff = n_current_handoff / n_tasks

        group.loc[idx, 'n_tasks'] = n_tasks
        group.loc[idx, 'n_cases'] = n_cases
        group.loc[idx, 'n_acts'] = n_acts
        group.loc[idx, 'n_handoffs'] = n_handoffs
        group.loc[idx, 'ent_act'] = ent_act
        group.loc[idx, 'ent_case'] = ent_case
        group.loc[idx, 'ent_handoff'] = ent_handoff
        group.loc[idx, 'ratio_act_case'] = ratio_act_case
        group.loc[idx, 'busyness'] = busyness

        group.loc[idx, 'n_current_case'] = n_current_case
        group.loc[idx, 'n_current_act'] = n_current_act
        group.loc[idx, 'n_current_handoff'] = n_current_handoff
        group.loc[idx, 'ratio_current_case'] = ratio_current_case
        group.loc[idx, 'ratio_current_act'] = ratio_current_act
        group.loc[idx, 'ratio_current_handoff'] = ratio_current_handoff

        # add frequencies of all activities and handoffs (not just the current one)
        dt_act_freqs = all_prev_exp[act_freq_cols]
        dt_act_freqs = dt_act_freqs.sum()
        dt_act_freqs.columns = act_freq_cols_sum
        group.loc[idx, act_freq_cols_sum] = dt_act_freqs.values
        dt_act_freqs = dt_act_freqs / np.sum(dt_act_freqs)
        dt_act_freqs.columns = act_freq_cols_ratio
        group.loc[idx, act_freq_cols_ratio] = dt_act_freqs.values

        dt_handoff_freqs = all_prev_exp[handoff_freq_cols]
        dt_handoff_freqs = dt_handoff_freqs.sum()
        dt_handoff_freqs.columns = handoff_freq_cols_sum
        group.loc[idx, handoff_freq_cols_sum] = dt_handoff_freqs.values
        dt_handoff_freqs = dt_handoff_freqs / np.sum(dt_handoff_freqs)
        dt_handoff_freqs.columns = handoff_freq_cols_ratio
        group.loc[idx, handoff_freq_cols_ratio] = dt_handoff_freqs.values
        if recent_days is not None:
            n_tasks_recent = len(recent_prev_exp)
            n_cases_recent = len(recent_prev_exp[case_id_col].unique())
            n_acts_recent = len(recent_prev_exp[activity_col].unique())
            n_handoffs_recent = len(recent_prev_exp[handoff_col].unique())
            ent_act_recent = ent(recent_prev_exp, activity_col)
            ent_case_recent = ent(recent_prev_exp, case_id_col)
            ent_handoff_recent = ent(recent_prev_exp, handoff_col)
            d = (recent_prev_exp[timestamp_col].max() - recent_prev_exp[timestamp_col].min()).days
            busyness_recent = n_tasks_recent / d if d else 0

            ratio_act_case_recent = n_tasks_recent / n_cases_recent

            n_current_case_recent = len(recent_prev_exp[recent_prev_exp[case_id_col] == row[case_id_col]])
            n_current_act_recent = len(recent_prev_exp[recent_prev_exp[activity_col] == row[activity_col]])
            n_current_handoff_recent = len(recent_prev_exp[recent_prev_exp[handoff_col] == row[handoff_col]])
            ratio_current_case_recent = n_current_case_recent / n_tasks_recent
            ratio_current_act_recent = n_current_act_recent / n_tasks_recent
            ratio_current_handoff_recent = n_current_handoff_recent / n_tasks_recent

            group.loc[idx, "n_tasks_recent"] = n_tasks_recent
            group.loc[idx, "n_cases_recent"] = n_cases_recent
            group.loc[idx, "n_acts_recent"] = n_acts_recent
            group.loc[idx, "n_handoffs_recent"] = n_handoffs_recent
            group.loc[idx, "ent_act_recent"] = ent_act_recent
            group.loc[idx, "ent_case_recent"] = ent_case_recent
            group.loc[idx, "ent_handoff_recent"] = ent_handoff_recent
            group.loc[idx, "ratio_act_case_recent"] = ratio_act_case_recent
            group.loc[idx, "busyness_recent"] = busyness_recent
            group.loc[idx, "n_current_case_recent"] = n_current_case_recent
            group.loc[idx, "n_current_act_recent"] = n_current_act_recent
            group.loc[idx, "n_current_handoff_recent"] = n_current_handoff_recent
            group.loc[idx, "ratio_current_case_recent"] = ratio_current_case_recent
            group.loc[idx, "ratio_current_act_recent"] = ratio_current_act_recent
            group.loc[idx, "ratio_current_handoff_recent"] = ratio_current_handoff_recent

        idx += 1

    return group


for filename in filenames:
    start_time = time.time()
    # Get the current time
    now = datetime.now()

    # Format the time in 12-hour format
    time_string = now.strftime("%I:%M %p")

    data = pd.read_csv(os.path.join(input_data_folder, filename), sep=";")

    data[timestamp_col] = pd.to_datetime(data[timestamp_col],  format='mixed')
    data["timesincemidnight"] = data[timestamp_col].dt.hour * 60 + data[timestamp_col].dt.minute
    data["month"] = data[timestamp_col].dt.month
    data["weekday"] = data[timestamp_col].dt.weekday
    data["hour"] = data[timestamp_col].dt.hour

    # add features extracted from timestamp
    print("Extracting timestamp features...")
    sys.stdout.flush()

    data = data.groupby(case_id_col).apply(extract_timestamp_features)

    # removing outlier
    data=detect_and_remove_outlier(data)


    # Assign the last 'O' activity to reduce the data based on the results of the application.
    print("Assigning last O event...")
    data = data.reset_index(drop=True)
    sys.stdout.flush()
    last_o_events = \
        data[data['EventOrigin'] == "Offer"].sort_values(timestamp_col, ascending=True, kind='mergesort').groupby(
            case_id_col).last()["concept:name"]
    last_o_events = pd.DataFrame(last_o_events)
    last_o_events.columns = ["last_o_activity"]
    data = data.merge(last_o_events, left_on=case_id_col, right_index=True)
    print("all last o event set is** befor filtering:", data["last_o_activity"].value_counts())
    " For BPIC17_Accepted, place 'O_Accepted' in the following section, and for BPIC17_Cancelled, place 'O_Cancelled' in the following section."
    data = data[data.last_o_activity.isin(["O_Cancelled"])]  # inputs: "O_Accepted"  "O_Cancelled"
    all_last_O_activity = pd.unique(data["last_o_activity"]).tolist()
    # print()
    print("all last o event set is** after filtering:", data["last_o_activity"].value_counts())

    #  Discretizing and labeling for data reduction based on the following sample reduction.
    print("labeling 1 ..... ")
    sys.stdout.flush()
    number_of_bunch = main.number_of_bunch0
    a = pd.DataFrame(data['TotalTimeOFCase'])
    maxnum = a.values.max()
    minmum = a.values.min()
    boundaries = np.arange(minmum, maxnum, (maxnum - minmum) / number_of_bunch)
    boundaries = sorted({minmum, maxnum + 1} | set(boundaries))
    a_discretized = pd.cut(a["TotalTimeOFCase"], bins=boundaries, labels=range(len(boundaries) - 1), right=False)
    a_discretized_autolabeled = LabelEncoder().fit_transform(a_discretized)
    transformation = set((i, j) for i, j in zip(a_discretized_autolabeled, a_discretized))
    transformationdic = {itera[0]: itera[1] for itera in transformation}
    intervals = [pd.Interval(left=boundaries[i], right=boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    intervals_dict = {i: intervals[i] for i in range(len(intervals))}
    data[label_col] = a_discretized_autolabeled


    sys.stdout.flush()

    "Select a certain amount of data based on the real proportions of labels in the original dataset for conducting experiments. If this is not needed, make it a comment."
    data = data.sort_values(["Case ID", "timesincelastevent"], ascending=False, kind='mergesort')
    tatal_data = 40000
    percent_of_label = data["label"].value_counts()
    percent_of_label = percent_of_label.apply(lambda x: x / percent_of_label.sum())
    number_of_data_per_labe = percent_of_label.apply(lambda x: int(x * tatal_data))
    data = data.sort_values(["Case ID", "TotalTimeOFCase"], ascending=False, kind="mergsort")
    data2 = data.groupby("label").apply(lambda x: x.head(number_of_data_per_labe[x["label"].iloc[0]])).reset_index(
        drop=True)
    case_ids_al = defaultdict(int)
    for i in data["label"].unique():
        case_ids = data.groupby("label").nth(number_of_data_per_labe[i])["Case ID"]
        if not case_ids.empty:
            case_ids_al[i] = case_ids.iloc[0]  # Get the first element
        else:
            case_ids_al[i] = None
    for value in case_ids_al:
        data2.drop(data2[data2["Case ID"] == case_ids_al[value]].index, inplace=True)
    data=data2
    del(data2)

    # deleting labels
    data.drop('label', inplace=True, axis=1)


    if (main.preprossing_mod == "regression"):
        data[label_col]=data["TotalTimeOFCase"]

    "extracting other inter-case features"
    if main.inter_intra_mode == 'inter+intra':
        print('Extracting resource experience')
        recent_days = None
        handoff_col = "prev_resource"
        resource_col_index = data.columns.get_loc(resource_col)
        "extracting previous resources"
        data = data.sort_values(timestamp_col, ascending=True, kind="mergesort").groupby(case_id_col).apply(
            get_prev_resource_and_event_nr)
        data[handoff_col] = data[handoff_col].fillna("first")
        data = pd.concat([data, pd.get_dummies(data[activity_col], prefix="act_freq")],
                         axis=1)  # Making dummies of activities
        data = pd.concat([data, pd.get_dummies(data[handoff_col], prefix="handoff_freq")],
                         axis=1)  # Making dummies of handoffs(prev_resources)data = pd.concat([data, pd.get_dummies(data[activity_col], prefix="act_freq")], axis=1)     #Making dummies of activities

        data.to_csv((os.path.join(main.output_data_folder, 'data_after_dummy.csv')), sep=";", index=False)
        act_freq_cols = [col for col in data.columns if col.startswith("act_freq")]
        act_freq_cols_df = pd.DataFrame(act_freq_cols)
        handoff_freq_cols = [col for col in data.columns if col.startswith("handoff_freq")]
        handoff_freq_cols_df = pd.DataFrame(handoff_freq_cols)
        act_freq_cols_sum = ["sum_%s" % col for col in act_freq_cols]
        act_freq_cols_ratio = ["ratio_%s" % col for col in act_freq_cols]
        handoff_freq_cols_sum = ["sum_%s" % col for col in handoff_freq_cols]
        handoff_freq_cols_ratio = ["ratio_%s" % col for col in handoff_freq_cols]
        "extracting resource experience"
        data = data.reset_index(drop=True)
        data = data.sort_values(timestamp_col, ascending=True, kind="mergesort").groupby(resource_col).apply(
            extract_experience)
        other_resource_experience = ['n_tasks', 'n_cases', 'n_acts', 'n_handoffs', 'ent_act',
                                     'ent_case', 'ent_handoff', 'ratio_act_case', 'busyness',
                                     'n_current_case', 'n_current_act', 'n_current_handoff',
                                     'ratio_current_case', 'ratio_current_act', 'ratio_current_handoff']
        other_resource_experience_recent = ["n_tasks_recent", "n_cases_recent", "n_acts_recent",
                                            "n_handoffs_recent", "ent_act_recent", "ent_case_recent",
                                            "ent_handoff_recent",
                                            "ratio_act_case_recent", "n_current_case_recent", "n_current_act_recent",
                                            "n_current_handoff_recent",
                                            "ratio_current_case_recent", "ratio_current_act_recent",
                                            "ratio_current_handoff_recent", "busyness_recent"]
        data = data.drop(act_freq_cols + handoff_freq_cols, axis=1)
        data.to_csv((os.path.join(main.output_data_folder, 'data_after_dummy.1.csv')), sep=";", index=False)

        # Extracting Open Cases
        print("Extracting open cases...")
        sys.stdout.flush()
        data = data.sort_values([timestamp_col], ascending=True, kind='mergesort')

        dt_first_last_timestamps = data.groupby(case_id_col)[timestamp_col].agg([min, max])
        dt_first_last_timestamps.columns = ["start_time", "end_time"]
        data["open_cases"] = data[timestamp_col].apply(get_open_cases)


        # Resource multitasking Extraction
        print("Extracting multitasking of resources...")
        sys.stdout.flush()
        data2 = data.copy()
        data2 = data2.sort_index(ascending=True, kind='mergesort')
        data2 = data2.sort_values([case_id_col, timestamp_col], ascending=True, kind='mergesort')
        temp2 = data2.copy()
        del (data2)
        temp1 = pd.DataFrame(columns=['resource', 'start_working', 'end_working', 'case_id'])
        temp1._set_value(0, 'resource', temp2[resource_col].iloc[0])
        temp1._set_value(0, 'start_working', temp2[timestamp_col].iloc[0])
        temp1._set_value(0, 'case_id', temp2[case_id_col].iloc[0])
        i1 = 0

        for i2 in range(0, len(temp2[resource_col])):
            if (i2 < len(temp2[resource_col]) - 1):
                if ((temp2[resource_col].iloc[i2 + 1] == temp1['resource'].iloc[i1]) & (
                        temp2[case_id_col].iloc[i2 + 1] == temp1['case_id'].iloc[i1])):
                    pass

                else:

                    temp1._set_value(i1, 'end_working', temp2[timestamp_col].iloc[i2])
                    temp1._set_value(i1 + 1, 'resource', temp2[resource_col].iloc[i2 + 1])
                    temp1._set_value(i1 + 1, 'start_working', temp2[timestamp_col].iloc[i2 + 1])
                    temp1._set_value(i1 + 1, 'case_id', temp2[case_id_col].iloc[i2 + 1])
                    i1 = i1 + 1
            elif (i2 == len(temp2[resource_col]) - 1):
                if ((temp2[resource_col].iloc[i2] == temp1['resource'].iloc[i1]) & (
                        temp2[case_id_col].iloc[i2] == temp1['case_id'].iloc[i1])):
                    temp1._set_value(i1, 'end_working', temp2[timestamp_col].iloc[i2])
                    temp1._set_value(i1, 'resource', temp2[resource_col].iloc[i2])
                    temp1._set_value(i1, 'case_id', temp2[case_id_col].iloc[i2])
                else:
                    temp1._set_value(i1, 'end_working', temp2[timestamp_col].iloc[i2 - 1])
                    temp1._set_value(i1, 'resource', temp2[resource_col].iloc[i2 - 1])
                    temp1._set_value(i1, 'case_id', temp2[case_id_col].iloc[i2 - 1])
                    temp1._set_value(i1 + 1, 'resource', temp2[resource_col].iloc[i2])
                    temp1._set_value(i1 + 1, 'start_working', temp2[timestamp_col].iloc[i2])
                    temp1._set_value(i1 + 1, 'end_working', temp2[timestamp_col].iloc[i2])
                    temp1._set_value(i1 + 1, 'case_id', temp2[case_id_col].iloc[i2])
        data["resource_multitasking"] = data[[resource_col, case_id_col, timestamp_col]].apply(get_multitasking_recources,
                                                                                               axis=1)
        del (temp2)
        del (temp1)



for activity in relevant_offer_events:
    print("Finishing dataset for activity ", activity)
    sys.stdout.flush()
    dt_labeled = data.copy()

    # impute missing values
    grouped = dt_labeled.sort_values(timestamp_col, ascending=True, kind='mergesort').groupby(case_id_col)
    for col in static_cols + dynamic_cols:
        dt_labeled[col] = grouped[col].transform(lambda grp: grp.fillna(method='ffill'))

    dt_labeled[cat_cols] = dt_labeled[cat_cols].fillna('missing')
    dt_labeled = dt_labeled.fillna(0)

    if main.inter_intra_mode == 'inter+intra':
        all_columns = static_cols + dynamic_cols + [ "open_cases", 'resource_multitasking',
                                                ] + other_resource_experience
    if main.inter_intra_mode == 'intra':
        all_columns = static_cols + dynamic_cols


    dt_labeled = dt_labeled.loc[:, all_columns]


    dt_labeled.to_csv(os.path.join(output_data_folder, "%s_%s.csv" % (filename[:-4], activity)), sep=";",
                      index=False)

    print('Execution time', time.time() - start_time)
    now2 = datetime.now()

    time_finishing = now2.strftime("%I:%M %p")
    print('starting time is :', time_string)
    print('finishing time is :', time_finishing)
