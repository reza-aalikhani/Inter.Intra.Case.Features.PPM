import EncoderFactory
import numpy
from DatasetManager import DatasetManager
import BucketFactory

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor


import time
import os

import pickle

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR

import main
"*** This file serves to conduct experiments using the optimal parameters determined by the optimize_params module."


"The running_function serves to run multiple models using the Running_Program module."
def running_func(bucket_method_in=main.bucket_method, cls_encoding_in=main.cls_encoding, cls_method_in=main.cls_method,
                 gap_in=main.gap, n_iter_in=main.n_iter, inter_intra_mode_in=main.inter_intra_mode):
    dataset_ref = main.dataset_ref
    params_dir = main.params_dir
    results_dir = main.results_dir
    bucket_method = bucket_method_in
    cls_encoding = cls_encoding_in
    cls_method = cls_method_in
    gap = gap_in
    n_iter = n_iter_in

    if bucket_method == "state":
        bucket_encoding = "last"
    else:
        bucket_encoding = "agg"

    method_name = "%s_%s" % (bucket_method, cls_encoding)

    dataset_ref_to_datasets = main.dataset_ref_to_datasets


    encoding_dict = {
        "laststate": ["static", "last"],
        "agg": ["static", "agg"],
        "index": ["static", "index"],
        "combined": ["static", "last", "agg"],
        "previous": ["static", "previous"]

    }

    datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]
    methods = encoding_dict[cls_encoding]

    train_ratio = 0.8
    random_state = 22

    # create results directory
    if not os.path.exists(os.path.join(params_dir)):
        os.makedirs(os.path.join(params_dir))

    for dataset_name in datasets:

        # load optimal params
        optimal_params_filename = os.path.join(params_dir,
                                               "optimal_params_%s_%s_%s_%s_%s.pickle" % (
                                                   cls_method, dataset_name, method_name, main.preprossing_mod,
                                                   inter_intra_mode_in))
        if not os.path.isfile(optimal_params_filename) or os.path.getsize(optimal_params_filename) <= 0:
            continue

        with open(optimal_params_filename, "rb") as fin:
            args = pickle.load(fin)

        # read the data
        dataset_manager = DatasetManager(dataset_name)
        data = dataset_manager.read_dataset()
        cls_encoder_args = {'case_id_col': dataset_manager.case_id_col,
                            'static_cat_cols': dataset_manager.static_cat_cols,
                            'static_num_cols': dataset_manager.static_num_cols,
                            'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                            'dynamic_num_cols': dataset_manager.dynamic_num_cols,
                            'fillna': True}


        # determine min and max (truncated) prefix lengths
        min_prefix_length = 1
        if "traffic_fines" in dataset_name:
            max_prefix_length = 10
        elif "bpic2017" in dataset_name:
            max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))

        else:
            max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))

        # split into training and test
        train, test = dataset_manager.split_data_strict(data, train_ratio, split="temporal")

        if gap > 1:
            outfile = os.path.join(results_dir,
                                   "performance_results_%s_%s_%s_%s_%s_gap%s.csv" % (
                                       cls_method, dataset_name, method_name, main.preprossing_mod, inter_intra_mode_in,
                                       gap))
        else:
            outfile = os.path.join(results_dir,
                                   "performance_results_%s_%s_%s_%s_%s.csv" % (
                                   cls_method, dataset_name, method_name, main.preprossing_mod, inter_intra_mode_in))

        start_test_prefix_generation = time.time()
        dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)
        test_prefix_generation_time = time.time() - start_test_prefix_generation

        offline_total_times = []
        online_event_times = []
        train_prefix_generation_times = []
        for ii in range(n_iter):
            # create prefix logs
            start_train_prefix_generation = time.time()
            dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length, gap)
            train_prefix_generation_time = time.time() - start_train_prefix_generation
            train_prefix_generation_times.append(train_prefix_generation_time)

            # Bucketing prefixes based on control flow
            bucketer_args = {'encoding_method': bucket_encoding,
                             'case_id_col': dataset_manager.case_id_col,
                             'cat_cols': [dataset_manager.activity_col],
                             'num_cols': [],
                             'random_state': random_state}
            if bucket_method == "cluster":
                bucketer_args["n_clusters"] = int(args["n_clusters"])
            bucketer = BucketFactory.get_bucketer(bucket_method, **bucketer_args)

            start_offline_time_bucket = time.time()
            bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)
            offline_time_bucket = time.time() - start_offline_time_bucket

            bucket_assignments_test = bucketer.predict(dt_test_prefixes)

            preds_all = []
            test_y_all = []
            nr_events_all = []
            offline_time_fit = 0
            current_online_event_times = []
            for bucket in set(bucket_assignments_test):
                if bucket_method == "prefix":
                    optimal_param = main.optimal_prefixbased_param(cls_method_in=cls_method_in,
                                                                   bucket_method_in=bucket_method_in,
                                                                   cls_encoding_in=cls_encoding_in)
                    current_args = optimal_param[bucket]
                else:
                    current_args = args
                relevant_train_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[
                    bucket_assignments_train == bucket]
                relevant_test_cases_bucket = dataset_manager.get_indexes(dt_test_prefixes)[
                    bucket_assignments_test == bucket]
                dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes,
                                                                              relevant_test_cases_bucket)
                nr_events_all.extend(list(dataset_manager.get_prefix_lengths(dt_test_bucket)))
                if len(relevant_train_cases_bucket) == 0:
                    preds = [dataset_manager.get_class_ratio(train)] * len(relevant_test_cases_bucket)
                    test_y_all.extend(dataset_manager.get_label_numeric_reg(dt_test_bucket))
                    current_online_event_times.extend([0] * len(preds))
                else:
                    dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes,
                                                                                   relevant_train_cases_bucket)  # one row per event
                    train_y = dataset_manager.get_label_numeric_reg(dt_train_bucket)

                    if len(set(train_y)) < 2:
                        preds = [train_y[0]] * len(relevant_test_cases_bucket)
                        current_online_event_times.extend([0] * len(preds))
                        test_y_all.extend(dataset_manager.get_label_numeric_reg(dt_test_bucket))
                    else:
                        start_offline_time_fit = time.time()
                        feature_combiner = FeatureUnion(
                            [(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])

                        if cls_method == "rf":
                            cls = RandomForestRegressor(n_estimators=500,
                                                        max_features=current_args['max_features'],
                                                        random_state=random_state)

                        elif cls_method == "xgboost":
                            cls = xgb.XGBRegressor(
                                n_estimators=500,
                                learning_rate=args['learning_rate'],
                                subsample=args['subsample'],
                                max_depth=int(args['max_depth']),
                                colsample_bytree=args['colsample_bytree'],
                                min_child_weight=int(args['min_child_weight']),
                                seed=random_state)


                        elif cls_method == "svm":

                            cls = SVR(C=2 ** args['C'],
                                      gamma=2 ** args['gamma'],
                                      epsilon=args['epsilon'])
                        elif cls_method == "DecisionTree":
                            cls = DecisionTreeRegressor(max_depth=args["max_depth"],
                                                        min_samples_leaf=args["min_samples_leaf"],
                                                        min_weight_fraction_leaf=args["min_weight_fraction_leaf"],
                                                        max_leaf_nodes=args["max_leaf_nodes"])


                        if cls_method == "svm" :
                            pipeline = Pipeline(
                                [('encoder', feature_combiner), ('scaler', StandardScaler()), ('cls', cls)])

                        else:
                            pipeline = Pipeline([('encoder', feature_combiner), ('cls', cls)])

                        pipeline.fit(dt_train_bucket, train_y)
                        offline_time_fit += time.time() - start_offline_time_fit

                        # predict separately for each prefix case
                        preds = []
                        test_all_grouped = dt_test_bucket.groupby(dataset_manager.case_id_col)
                        for _, group in test_all_grouped:


                            test_y_all.extend(dataset_manager.get_label_numeric_reg(group))
                            start = time.time()
                            _ = bucketer.predict(group)
                            if cls_method == "svm":
                                pred = pipeline.predict(group)

                            if cls_method == "xgboost":
                                pred0 = pipeline.predict(group)
                                pred = pred0.clip(min=0)


                            if cls_method == "rf":
                                pred = pipeline.predict(group)

                            if cls_method == "DecisionTree":
                                pred = pipeline.predict(group)


                            pipeline_pred_time = time.time() - start
                            current_online_event_times.append(pipeline_pred_time / len(group))
                            preds.extend(pred)
                preds_all.extend(preds)

            offline_total_time = offline_time_bucket + offline_time_fit + train_prefix_generation_time
            offline_total_times.append(offline_total_time)
            online_event_times.append(current_online_event_times)

        with open(outfile, 'w') as fout:
            fout.write("%s;%s;%s;%s;%s;%s;%s" % ("dataset", "method", "cls", "nr_events", "n_iter", "metric", "score"))
            fout.write("\n")
            fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
                dataset_name, method_name, cls_method, -1, -1, "test_prefix_generation_time",
                test_prefix_generation_time))

            for ii in range(len(offline_total_times)):
                fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
                    dataset_name, method_name, cls_method, -1, ii, "train_prefix_generation_time",
                    train_prefix_generation_times[ii]))
                fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
                    dataset_name, method_name, cls_method, -1, ii, "offline_time_total", offline_total_times[ii]))
                fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
                    dataset_name, method_name, cls_method, -1, ii, "online_time_avg", np.mean(online_event_times[ii])))
                fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
                    dataset_name, method_name, cls_method, -1, ii, "online_time_std", np.std(online_event_times[ii])))

            preds_all_time = preds_all
            dt_results = pd.DataFrame({"actual": test_y_all, "predicted": preds_all_time, "nr_events": nr_events_all})

            for nr_events, group in dt_results.groupby("nr_events"):

                if len(set(group.actual)) < 2:

                    fout.write(
                        "%s;%s;%s;%s;%s;%s;%s\n" % (
                            dataset_name, method_name, cls_method, nr_events, -1, "MAE", np.nan))
                else:

                    fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, nr_events, -1, "MAE",
                                                           mean_absolute_error(group.actual, group.predicted)))

            fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
                dataset_name, method_name, cls_method, -1, -1, "MAE",
                mean_absolute_error(dt_results.actual, dt_results.predicted)))

            online_event_times_flat = [t for iter_online_event_times in online_event_times for t in
                                       iter_online_event_times]
            fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
                dataset_name, method_name, cls_method, -1, -1, "online_time_avg", np.mean(online_event_times_flat)))
            fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
                dataset_name, method_name, cls_method, -1, -1, "online_time_std", np.std(online_event_times_flat)))
            fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
                dataset_name, method_name, cls_method, -1, -1, "offline_time_total_avg", np.mean(offline_total_times)))
            fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
                dataset_name, method_name, cls_method, -1, -1, "offline_time_total_std", np.std(offline_total_times)))


# ----------------------------------------------------------------- Separate into running function and manual running ###########
# Manual running using the configuration in the main file.
if __name__ == "__main__":
    dataset_ref = main.dataset_ref
    params_dir = main.params_dir
    results_dir = main.results_dir
    bucket_method = main.bucket_method
    cls_encoding = main.cls_encoding
    cls_method = main.cls_method
    gap = main.gap
    n_iter = main.n_iter

    if bucket_method == "state":
        bucket_encoding = "last"
    else:
        bucket_encoding = "agg"

    method_name = "%s_%s" % (bucket_method, cls_encoding)

    dataset_ref_to_datasets = main.dataset_ref_to_datasets
    encoding_dict = {
        "laststate": ["static", "last"],
        "agg": ["static", "agg"],
        "index": ["static", "index"],
        "combined": ["static", "last", "agg"]
        , "previous": ["static", "previous"]

    }

    datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]
    methods = encoding_dict[cls_encoding]

    train_ratio = 0.8
    random_state = 22

    # create results directory
    if not os.path.exists(os.path.join(params_dir)):
        os.makedirs(os.path.join(params_dir))

    for dataset_name in datasets:

        # load optimal params
        optimal_params_filename = os.path.join(params_dir,
                                               "optimal_params_%s_%s_%s_%s_%s.pickle" % (
                                                   cls_method, dataset_name, method_name, main.preprossing_mod, main.inter_intra_mode))
        if not os.path.isfile(optimal_params_filename) or os.path.getsize(optimal_params_filename) <= 0:
            continue

        with open(optimal_params_filename, "rb") as fin:
            args = pickle.load(fin)

        # read the data
        dataset_manager = DatasetManager(dataset_name)
        data = dataset_manager.read_dataset()
        cls_encoder_args = {'case_id_col': dataset_manager.case_id_col,
                            'static_cat_cols': dataset_manager.static_cat_cols,
                            'static_num_cols': dataset_manager.static_num_cols,
                            'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                            'dynamic_num_cols': dataset_manager.dynamic_num_cols,
                            'fillna': True}

        number_of_bunch = len(data["label"].unique())

        # determine min and max (truncated) prefix lengths
        min_prefix_length = 1
        if "traffic_fines" in dataset_name:
            max_prefix_length = 10
        elif "bpic2017" in dataset_name:
            max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))

        else:
            max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))

        # split into training and test
        train, test = dataset_manager.split_data_strict(data, train_ratio, split="temporal")

        if gap > 1:
            outfile = os.path.join(results_dir,
                                   "performance_results_%s_%s_%s_%s_gap%s.csv" % (
                                       cls_method, dataset_name, method_name, main.preprossing_mod, gap))
        else:
            outfile = os.path.join(results_dir,
                                   "performance_results_%s_%s_%s_%s.csv" % (
                                   cls_method, dataset_name, method_name, main.preprossing_mod))

        start_test_prefix_generation = time.time()
        dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)
        test_prefix_generation_time = time.time() - start_test_prefix_generation

        offline_total_times = []
        online_event_times = []
        train_prefix_generation_times = []
        for ii in range(n_iter):
            print(f"{(ii / n_iter) * 100} % of iteration completeted")
            # create prefix logs
            start_train_prefix_generation = time.time()
            dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length, gap)
            train_prefix_generation_time = time.time() - start_train_prefix_generation
            train_prefix_generation_times.append(train_prefix_generation_time)

            # Bucketing prefixes based on control flow
            bucketer_args = {'encoding_method': bucket_encoding,
                             'case_id_col': dataset_manager.case_id_col,
                             'cat_cols': [dataset_manager.activity_col],
                             'num_cols': [],
                             'random_state': random_state}
            if bucket_method == "cluster":
                bucketer_args["n_clusters"] = int(args["n_clusters"])
            bucketer = BucketFactory.get_bucketer(bucket_method, **bucketer_args)

            start_offline_time_bucket = time.time()
            bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)
            offline_time_bucket = time.time() - start_offline_time_bucket

            bucket_assignments_test = bucketer.predict(dt_test_prefixes)

            preds_all = []
            test_y_all = []
            nr_events_all = []
            offline_time_fit = 0
            current_online_event_times = []
            for bucket in set(bucket_assignments_test):
                if bucket_method == "prefix":
                    optimal_param = main.optimal_prefixbased_param()
                    current_args = optimal_param[bucket]
                else:
                    current_args = args
                relevant_train_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[
                    bucket_assignments_train == bucket]
                relevant_test_cases_bucket = dataset_manager.get_indexes(dt_test_prefixes)[
                    bucket_assignments_test == bucket]
                dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes,
                                                                              relevant_test_cases_bucket)
                nr_events_all.extend(list(dataset_manager.get_prefix_lengths(dt_test_bucket)))
                if len(relevant_train_cases_bucket) == 0:
                    preds = [dataset_manager.get_class_ratio(train)] * len(relevant_test_cases_bucket)
                    test_y_all.extend(dataset_manager.get_label_numeric_reg(dt_test_bucket))
                    current_online_event_times.extend([0] * len(preds))
                else:
                    dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes,
                                                                                   relevant_train_cases_bucket)  # one row per event
                    train_y = dataset_manager.get_label_numeric_reg(dt_train_bucket)

                    if len(set(train_y)) < 2:  # my extension
                        preds = [train_y[0]] * len(relevant_test_cases_bucket)
                        current_online_event_times.extend([0] * len(preds))
                        test_y_all.extend(dataset_manager.get_label_numeric_reg(dt_test_bucket))

                    else:
                        start_offline_time_fit = time.time()
                        feature_combiner = FeatureUnion(
                            [(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])

                        if cls_method == "rf":
                            cls = RandomForestRegressor(n_estimators=500,
                                                        max_features=current_args['max_features'],
                                                        random_state=random_state)

                        elif cls_method == "xgboost":
                            cls = xgb.XGBRegressor(
                                n_estimators=500,
                                learning_rate=args['learning_rate'],
                                subsample=args['subsample'],
                                max_depth=int(args['max_depth']),
                                colsample_bytree=args['colsample_bytree'],
                                min_child_weight=int(args['min_child_weight']),
                                seed=random_state)


                        elif cls_method == "svm":

                            cls = SVR(C=2 ** args['C'],
                                      gamma=2 ** args['gamma'],
                                      epsilon=args['epsilon'])
                        elif cls_method == "DecisionTree":
                            cls = DecisionTreeRegressor(max_depth=args["max_depth"],
                                                        min_samples_leaf=args["min_samples_leaf"],
                                                        min_weight_fraction_leaf=args["min_weight_fraction_leaf"],
                                                        max_leaf_nodes=args["max_leaf_nodes"])


                        if cls_method == "svm" :
                            pipeline = Pipeline(
                                [('encoder', feature_combiner), ('scaler', StandardScaler()), ('cls', cls)])



                        else:
                            pipeline = Pipeline([('encoder', feature_combiner), ('cls', cls)])



                        pipeline.fit(dt_train_bucket, train_y)
                        offline_time_fit += time.time() - start_offline_time_fit

                        # predict separately for each prefix case
                        preds = []
                        test_all_grouped = dt_test_bucket.groupby(dataset_manager.case_id_col)
                        for _, group in test_all_grouped:


                            test_y_all.extend(dataset_manager.get_label_numeric_reg(group))
                            start = time.time()
                            _ = bucketer.predict(group)
                            if cls_method == "svm":
                                pred = pipeline.predict(group)

                            if cls_method == "xgboost":
                                pred0 = pipeline.predict(group)
                                pred = pred0.clip(min=0)

                            if cls_method == "rf":
                                pred = pipeline.predict(group)
                            if cls_method == "DecisionTree":
                                pred = pipeline.predict(group)

                            pipeline_pred_time = time.time() - start
                            current_online_event_times.append(pipeline_pred_time / len(group))
                            preds.extend(pred)
                preds_all.extend(preds)

            offline_total_time = offline_time_bucket + offline_time_fit + train_prefix_generation_time
            offline_total_times.append(offline_total_time)
            online_event_times.append(current_online_event_times)

        with open(outfile, 'w') as fout:
            fout.write("%s;%s;%s;%s;%s;%s;%s" % ("dataset", "method", "cls", "nr_events", "n_iter", "metric", "score"))

            fout.write("\n")
            fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
                dataset_name, method_name, cls_method, -1, -1, "test_prefix_generation_time",
                test_prefix_generation_time))

            for ii in range(len(offline_total_times)):
                fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
                    dataset_name, method_name, cls_method, -1, ii, "train_prefix_generation_time",
                    train_prefix_generation_times[ii]))
                fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
                    dataset_name, method_name, cls_method, -1, ii, "offline_time_total", offline_total_times[ii]))
                fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
                    dataset_name, method_name, cls_method, -1, ii, "online_time_avg", np.mean(online_event_times[ii])))
                fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
                    dataset_name, method_name, cls_method, -1, ii, "online_time_std", np.std(online_event_times[ii])))

            preds_all_time = preds_all

            dt_results = pd.DataFrame({"actual": test_y_all, "predicted": preds_all_time, "nr_events": nr_events_all})

            for nr_events, group in dt_results.groupby("nr_events"):

                if len(set(group.actual)) < 2:
                    fout.write(
                        "%s;%s;%s;%s;%s;%s;%s\n" % (
                            dataset_name, method_name, cls_method, nr_events, -1, "MAE", np.nan))
                else:

                    fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, nr_events, -1, "MAE",
                                                           mean_absolute_error(group.actual, group.predicted)))

            fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
                dataset_name, method_name, cls_method, -1, -1, "MAE",
                mean_absolute_error(dt_results.actual, dt_results.predicted)))

            online_event_times_flat = [t for iter_online_event_times in online_event_times for t in
                                       iter_online_event_times]
            fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
                dataset_name, method_name, cls_method, -1, -1, "online_time_avg", np.mean(online_event_times_flat)))
            fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
                dataset_name, method_name, cls_method, -1, -1, "online_time_std", np.std(online_event_times_flat)))
            fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
                dataset_name, method_name, cls_method, -1, -1, "offline_time_total_avg", np.mean(offline_total_times)))
            fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (
                dataset_name, method_name, cls_method, -1, -1, "offline_time_total_std", np.std(offline_total_times)))
