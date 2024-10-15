import EncoderFactory
from DatasetManager import DatasetManager
import BucketFactory

import pandas as pd
# import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler



import time
import os

import pickle
from collections import defaultdict

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.tree  import DecisionTreeRegressor

from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import hyperopt
from hyperopt.pyll.base import scope

import main


def running_func(bucket_method_in, cls_encoding_in, cls_method_in,
                 n_iter_in, inter_intra_mode_in):
    def create_and_evaluate_model(args):
        global trial_nr
        trial_nr += 1

        start = time.time()
        score = 0
        for cv_iter in range(n_splits):

            dt_test_prefixes = dt_prefixes[cv_iter]
            dt_train_prefixes = pd.DataFrame()
            for cv_train_iter in range(n_splits):
                if cv_train_iter != cv_iter:
                    dt_train_prefixes = pd.concat([dt_train_prefixes, dt_prefixes[cv_train_iter]], axis=0)

            # Bucketing prefixes based on control flow
            bucketer_args = {'encoding_method': bucket_encoding,
                             'case_id_col': dataset_manager.case_id_col,
                             'cat_cols': [dataset_manager.activity_col],
                             'num_cols': [],
                             'random_state': random_state}

            if bucket_method == "cluster":
                bucketer_args["n_clusters"] = args["n_clusters"]
            bucketer = BucketFactory.get_bucketer(bucket_method, **bucketer_args)
            bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)
            bucket_assignments_test = bucketer.predict(dt_test_prefixes)
            preds_all = []
            test_y_all = []
            if "prefix" in method_name:
                scores = defaultdict(int)

            for bucket in set(bucket_assignments_test):

                relevant_train_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[
                    bucket_assignments_train == bucket]

                relevant_test_cases_bucket = dataset_manager.get_indexes(dt_test_prefixes)[
                    bucket_assignments_test == bucket]

                dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes,
                                                                              relevant_test_cases_bucket)

                test_y = dataset_manager.get_label_numeric_reg(dt_test_bucket)

                if len(relevant_train_cases_bucket) == 0:
                    preds = [class_ratios[cv_iter]] * len(relevant_test_cases_bucket)

                else:

                    dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes,
                                                                                   relevant_train_cases_bucket)  # one row per event
                    train_y = dataset_manager.get_label_numeric_reg(dt_train_bucket)
                    if len(set(train_y)) < 2:
                        preds = [train_y[0]] * len(relevant_test_cases_bucket)

                    else:
                        feature_combiner = FeatureUnion(
                            [(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])
                        if cls_method == "rf":
                            cls = RandomForestRegressor(n_estimators=500,
                                                        max_features=args['max_features'],
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
                                      epsilon=args['epsilon']
                                      )
                        elif cls_method == "DecisionTree":
                            cls=DecisionTreeRegressor(max_depth=args["max_depth"],
                                                      min_samples_leaf=args["min_samples_leaf"],
                                                      min_weight_fraction_leaf=args["min_weight_fraction_leaf"],
                                                      max_leaf_nodes=args["max_leaf_nodes"])

                        if cls_method == "svm":
                            pipeline = Pipeline(
                                [('encoder', feature_combiner), ('scaler', StandardScaler()), ('cls', cls)])
                        else:
                            pipeline = Pipeline([('encoder', feature_combiner), ('cls', cls)])


                        pipeline.fit(dt_train_bucket, train_y)

                        if cls_method == "svm":
                            preds = pipeline.predict(dt_test_bucket)

                        if cls_method == "xgboost":
                            preds0 = pipeline.predict(dt_test_bucket)
                            preds = preds0.clip(min=0)

                        if cls_method == "rf":
                            preds = pipeline.predict(dt_test_bucket)

                        if cls_method == "DecisionTree":
                                preds = pipeline.predict(dt_test_bucket)

                if "prefix" in method_name:
                    AME = 0
                    if len(set(test_y)) > 1:

                        AME = mean_absolute_error(dt_test_bucket.groupby("Case ID").first()["label"],preds)

                    if bucket in scores.keys():
                        scores[bucket] = [x + y for x, y in zip(scores[bucket], AME)]
                    else:
                        scores[bucket] = AME

                preds_all.extend(preds)
                test_y_all.extend(test_y)

            score += mean_absolute_error(test_y_all, preds_all)

        if "prefix" in method_name:
            for k, v in args.items():
                for bucket, bucket_score in scores.items():
                    fout_all.write("%s;%s;%s;%s;%s;%s;%s;%s\n" % (
                        trial_nr, dataset_name, cls_method, method_name, bucket, k, v, bucket_score / n_splits))

            fout_all.write("%s;%s;%s;%s;%s;%s;%s;%s\n" % (
                trial_nr, dataset_name, cls_method, method_name, 0, "processing_time", time.time() - start, 0))
        else:
            for k, v in args.items():
                fout_all.write(
                    "%s;%s;%s;%s;%s;%s;%s\n" % (
                    trial_nr, dataset_name, cls_method, method_name, k, v, score / n_splits))

            fout_all.write("%s;%s;%s;%s;%s;%s;%s\n" % (
                trial_nr, dataset_name, cls_method, method_name, "processing_time", time.time() - start, 0))
        fout_all.flush()

        return {'loss': (score / n_splits)/float(max_TotalTimeOFCase) , 'status': STATUS_OK, 'model': cls}




    dataset_ref = main.dataset_ref
    params_dir = main.params_dir
    n_iter = n_iter_in
    bucket_method = bucket_method_in
    cls_encoding = cls_encoding_in
    cls_method = cls_method_in

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
    n_splits = 3
    random_state = 22

    # create results directory
    if not os.path.exists(os.path.join(params_dir)):
        os.makedirs(os.path.join(params_dir))

    for dataset_name in datasets:
        start_time = time.time()
        # read the data
        dataset_manager = DatasetManager(dataset_name)
        # time recording
        sec1_time = time.time() - start_time
        print(f"the time of (sec1_time) is:", sec1_time)
        data = dataset_manager.read_dataset()

        max_TotalTimeOFCase = data["label"].max()

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

        train, _ = dataset_manager.split_data_strict(data, train_ratio, split="temporal")

        # prepare chunks for CV
        dt_prefixes = []
        class_ratios = []
        for train_chunk, test_chunk in dataset_manager.get_stratified_split_generator_reg(train, n_splits=n_splits):
            class_ratios.append(dataset_manager.get_class_ratio(train_chunk))
            # generate data where each prefix is a separate instance
            dt_prefixes.append(dataset_manager.generate_prefix_data(test_chunk, min_prefix_length, max_prefix_length))
        del train

        # set up search space
        if cls_method == "rf":
            space = {'max_features': hp.uniform('max_features', 0, 1)}
        elif cls_method == "xgboost":
            space = {'learning_rate': hp.uniform("learning_rate", 0, 1),
                     'subsample': hp.uniform("subsample", 0.5, 1),
                     'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
                     'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 1),
                     'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 6, 1))}

        elif cls_method == "svm":
            space = {'C': hp.uniform('C', -15, 15),
                     'gamma': hp.uniform('gamma', -15, 15),
                     "epsilon":hp.uniform('epsilon', 0, 1)}
        elif cls_method == "DecisionTree":
            space = {"max_depth" : scope.int(hp.quniform('max_depth', 1, 15, 2)),
           "min_samples_leaf":scope.int(hp.quniform('min_samples_leaf', 1, 10, 1)),
           "min_weight_fraction_leaf":hp.uniform("min_weight_fraction_leaf", 0, 0.5),
           "max_leaf_nodes": scope.int(hp.quniform('max_leaf_nodes', 10, 100, 10)) }

        if bucket_method == "cluster":
            space['n_clusters'] = scope.int(hp.quniform('n_clusters', 2, 50, 1))

        # optimize parameters
        global trial_nr
        trial_nr = 1
        trials = Trials()

        fout_all = open(
            os.path.join(params_dir, "param_optim_all_trials_%s_%s_%s_%s_%s.csv" % (
            cls_method, dataset_name, method_name, main.preprossing_mod, inter_intra_mode_in)),
            "w")
        if "prefix" in method_name:
            fout_all.write(
                "%s;%s;%s;%s;%s;%s;%s;%s" % ("iter", "dataset", "cls", "method", "nr_events", "param", "value", "MAE"))

            fout_all.write("\n")
        else:
            fout_all.write("%s;%s;%s;%s;%s;%s;%s" % ("iter", "dataset", "cls", "method", "param", "value", "MAE"))

            fout_all.write("\n")

        best = fmin(create_and_evaluate_model, space, algo=tpe.suggest, max_evals=n_iter, trials=trials)
        fout_all.close()

        # write the best parameters
        best_params = hyperopt.space_eval(space, best)
        outfile = os.path.join(params_dir, "optimal_params_%s_%s_%s_%s_%s.pickle" % (
        cls_method, dataset_name, method_name, main.preprossing_mod, inter_intra_mode_in))
        # write to file
        with open(outfile, "wb") as fout:
            pickle.dump(best_params, fout)

        print("finished one")


# ******************************************************* Separate into running function and manual running****************

if __name__ == "__main__":
    def create_and_evaluate_model(args):

        global trial_nr
        trial_nr += 1

        start = time.time()
        score = 0
        for cv_iter in range(n_splits):

            dt_test_prefixes = dt_prefixes[cv_iter]
            dt_train_prefixes = pd.DataFrame()
            for cv_train_iter in range(n_splits):
                if cv_train_iter != cv_iter:
                    dt_train_prefixes = pd.concat([dt_train_prefixes, dt_prefixes[cv_train_iter]], axis=0)

            # Bucketing prefixes based on control flow
            bucketer_args = {'encoding_method': bucket_encoding,
                             'case_id_col': dataset_manager.case_id_col,
                             'cat_cols': [dataset_manager.activity_col],
                             'num_cols': [],
                             'random_state': random_state}

            if bucket_method == "cluster":
                bucketer_args["n_clusters"] = args["n_clusters"]
            bucketer = BucketFactory.get_bucketer(bucket_method, **bucketer_args)
            bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)
            bucket_assignments_test = bucketer.predict(dt_test_prefixes)
            preds_all = []
            test_y_all = []
            if "prefix" in method_name:
                scores = defaultdict(int)

            for bucket in set(bucket_assignments_test):

                relevant_train_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[
                    bucket_assignments_train == bucket]

                relevant_test_cases_bucket = dataset_manager.get_indexes(dt_test_prefixes)[
                    bucket_assignments_test == bucket]

                dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes,
                                                                              relevant_test_cases_bucket)

                test_y = dataset_manager.get_label_numeric_reg(dt_test_bucket)

                if len(relevant_train_cases_bucket) == 0:
                    preds = [class_ratios[cv_iter]] * len(relevant_test_cases_bucket)

                else:

                    dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes,
                                                                                   relevant_train_cases_bucket)  # one row per event
                    train_y = dataset_manager.get_label_numeric_reg(dt_train_bucket)
                    if len(set(train_y)) < 2:
                        preds = [train_y[0]] * len(relevant_test_cases_bucket)

                    else:
                        feature_combiner = FeatureUnion(
                            [(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])
                        if cls_method == "rf":
                            cls = RandomForestRegressor(n_estimators=500,
                                                         max_features=args['max_features'],
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
                            cls=DecisionTreeRegressor(max_depth=args["max_depth"],
                                                      min_samples_leaf=args["min_samples_leaf"],
                                                      min_weight_fraction_leaf=args["min_weight_fraction_leaf"],
                                                      max_leaf_nodes=args["max_leaf_nodes"])


                        if cls_method == "svm":
                            pipeline = Pipeline(
                                [('encoder', feature_combiner), ('scaler', StandardScaler()), ('cls', cls)])

                        else:
                            pipeline = Pipeline([('encoder', feature_combiner), ('cls', cls)])


                        pipeline.fit(dt_train_bucket, train_y)

                        if cls_method == "svm":
                            preds=pipeline.predict(dt_test_bucket)

                        if cls_method == "xgboost":
                                preds0 = pipeline.predict(dt_test_bucket)
                                preds = preds0.clip(min=0)
                        if cls_method == "rf":
                                preds = pipeline.predict(dt_test_bucket)
                        if cls_method == "DecisionTree":
                                preds = pipeline.predict(dt_test_bucket)


                if "prefix" in method_name:
                    AME = 0
                    if len(set(test_y)) > 1:

                        AME = mean_absolute_error(dt_test_bucket.groupby("Case ID").first()["label"],preds)

                    if bucket in scores.keys():
                        scores[bucket] = [x + y for x, y in zip(scores[bucket], AME)]
                    else:
                        scores[bucket] = AME

                preds_all.extend(preds)
                test_y_all.extend(test_y)

            score += mean_absolute_error(test_y_all,preds_all)

        if "prefix" in method_name:
            for k, v in args.items():
                for bucket, bucket_score in scores.items():
                    fout_all.write("%s;%s;%s;%s;%s;%s;%s;%s\n" % (
                        trial_nr, dataset_name, cls_method, method_name, bucket, k, v, bucket_score / n_splits))

            fout_all.write("%s;%s;%s;%s;%s;%s;%s;%s\n" % (
                trial_nr, dataset_name, cls_method, method_name, 0, "processing_time", time.time() - start, 0))
        else:
            for k, v in args.items():

                fout_all.write(
                    "%s;%s;%s;%s;%s;%s;%s\n" % (trial_nr, dataset_name, cls_method, method_name, k, v, score / n_splits))

            fout_all.write("%s;%s;%s;%s;%s;%s;%s\n" % (
                trial_nr, dataset_name, cls_method, method_name, "processing_time", time.time() - start, 0))
        fout_all.flush()

        return {'loss': (score / n_splits)/float(max_TotalTimeOFCase) , 'status': STATUS_OK, 'model': cls}




    dataset_ref = main.dataset_ref
    params_dir = main.params_dir
    n_iter = main.n_iter
    bucket_method = main.bucket_method
    cls_encoding = main.cls_encoding
    cls_method = main.cls_method


    if bucket_method == "state":
        bucket_encoding = "last"
    else:
        bucket_encoding = "agg"

    method_name = "%s_%s" % (bucket_method, cls_encoding)

    dataset_ref_to_datasets=main.dataset_ref_to_datasets

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
    n_splits = 3
    random_state = 22

    # create results directory
    if not os.path.exists(os.path.join(params_dir)):
        os.makedirs(os.path.join(params_dir))

    for dataset_name in datasets:
        start_time = time.time()
        # read the data
        dataset_manager = DatasetManager(dataset_name)
        # time recording
        sec1_time = time.time()- start_time
        print(f"the time of (sec1_time) is:", sec1_time)
        data = dataset_manager.read_dataset()

        max_TotalTimeOFCase=data["label"].max()

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

        train, _ = dataset_manager.split_data_strict(data, train_ratio, split="temporal")

        # prepare chunks for CV
        dt_prefixes = []
        class_ratios = []
        for train_chunk, test_chunk in dataset_manager.get_stratified_split_generator_reg(train, n_splits=n_splits):
            class_ratios.append(dataset_manager.get_class_ratio(train_chunk))
            # generate data where each prefix is a separate instance
            dt_prefixes.append(dataset_manager.generate_prefix_data(test_chunk, min_prefix_length, max_prefix_length))
        del train

        # set up search space
        if cls_method == "rf":
            space = {'max_features': hp.uniform('max_features', 0, 1)}
        elif cls_method == "xgboost":
            space = {'learning_rate': hp.uniform("learning_rate", 0, 1),
                     'subsample': hp.uniform("subsample", 0.5, 1),
                     'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
                     'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 1),
                     'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 6, 1))}

        elif cls_method == "svm":
            space = {'C': hp.uniform('C', -15, 15),
                     'gamma': hp.uniform('gamma', -15, 15),
                     "epsilon":hp.uniform('epsilon', 0, 1)}
        elif cls_method == "DecisionTree":
            space = {
                "max_depth": scope.int(hp.quniform('max_depth', 1, 15, 2)),
                "min_samples_leaf": scope.int(hp.quniform('min_samples_leaf', 1, 10, 1)),
                "min_weight_fraction_leaf": hp.uniform("min_weight_fraction_leaf", 0, 0.5),
                "max_leaf_nodes": scope.int(hp.quniform('max_leaf_nodes', 10, 100, 10))}


        if bucket_method == "cluster":
            space['n_clusters'] = scope.int(hp.quniform('n_clusters', 2, 50, 1))

        # optimize parameters
        trial_nr = 1
        trials = Trials()

        fout_all = open(
            os.path.join(params_dir, "param_optim_all_trials_%s_%s_%s_%s.csv" % (cls_method, dataset_name, method_name,main.preprossing_mod)),
            "w")
        if "prefix" in method_name:
            fout_all.write(
                "%s;%s;%s;%s;%s;%s;%s;%s" % ("iter", "dataset", "cls", "method", "nr_events", "param", "value","MAE"))

            fout_all.write("\n")
        else:
            fout_all.write("%s;%s;%s;%s;%s;%s;%s" % ("iter", "dataset", "cls", "method", "param", "value","MAE"))

            fout_all.write("\n")

        best = fmin(create_and_evaluate_model, space, algo=tpe.suggest, max_evals=n_iter, trials=trials)
        fout_all.close()

        # write the best parameters
        best_params = hyperopt.space_eval(space, best)
        outfile = os.path.join(params_dir, "optimal_params_%s_%s_%s_%s.pickle" % (cls_method, dataset_name, method_name,main.preprossing_mod))
        # write to file
        with open(outfile, "wb") as fout:
            pickle.dump(best_params, fout)

