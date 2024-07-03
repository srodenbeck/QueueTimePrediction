from absl import app, flags
import torch
import torch.nn as nn
import torch.optim as optim
from optuna.samplers import TPESampler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import sys
import neptune
import neptune.integrations.optuna as npt_utils
import transformations
import imblearn.over_sampling
import optuna
from optuna.trial import TrialState
import pandas as pd

import config_file
import read_db

# flags.DEFINE_integer('batch_size', 32, 'Batch size')
# flags.DEFINE_boolean('shuffle', True, 'Shuffle training/validation set')
# flags.DEFINE_float('oversample', 0.4, 'Oversampling factor')
# flags.DEFINE_float('undersample', 0.8, 'Undersampling factor')
# flags.DEFINE_integer('epochs', 60, 'Number of epochs')
# flags.DEFINE_float('lr', 0.001, 'Learning rate')
# flags.DEFINE_enum('activ_fn', "leaky_relu", ['leaky_relu', 'relu'], 'Activation Function')
# flags.DEFINE_integer('n_jobs', 800_000, 'Number of jobs to run on')
# flags.DEFINE_boolean('condense_jobs', True, 'Condense jobs submitted by same user')
# FLAGS = flags.FLAGS

gl_df = None
gl_X = None
gl_y_one_hot = None
gl_feature_mapping_dict = None

gl_hyp_param = {
    "transformations": ["log", "min_max"],
    "n_layers_low": 2,
    "n_layers_high": 3,
    "layer_size_low": 16,
    "layer_size_high": 128,
    "dropout_low": -1,
    "dropout_high": -1,
    "features": ["austin_hypo_2", "queue_request", "user"]
}


def get_planned_target_index(df):
    return df.columns.get_loc('planned')


def get_feature_indices(df, feature_names):
    feature_indices = []
    for feature_name in feature_names:
        try:
            feature_indices.append(df.columns.get_loc(feature_name))
        except Exception as e:
            print(f"Error: Could not find '{feature_name}' in database\nExiting...")
            sys.exit(1)
    return feature_indices


def get_class_labels(y, threshold):
    np_array = np.where(y > threshold, 1, 0)
    return np_array


def to_one_hot(np_array, num_classes=2):
    one_hot = nn.functional.one_hot(torch.from_numpy(np_array), num_classes=num_classes)
    return one_hot


def create_dataloaders(X, y, transform):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        shuffle=FLAGS.shuffle)
    if transform == "log":
        X_train, X_test = transformations.scale_log(X_train, X_test)
    elif transform == "min_max":
        X_train, X_test = transformations.scale_min_max(X_train, X_test)

    # First step: converting to tensor
    x_train_to_tensor = torch.from_numpy(X_train).to(torch.float32)
    y_train_to_tensor = torch.from_numpy(y_train).to(torch.float32)
    x_test_to_tensor = torch.from_numpy(X_test).to(torch.float32)
    y_test_to_tensor = torch.from_numpy(y_test).to(torch.float32)

    # Second step: Creating TensorDataset for Dataloader
    train_dataset = TensorDataset(x_train_to_tensor, y_train_to_tensor)
    test_dataset = TensorDataset(x_test_to_tensor, y_test_to_tensor)

    train_dataloader = DataLoader(train_dataset, batch_size=FLAGS.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=FLAGS.batch_size)
    return train_dataloader, test_dataloader


def transform_test(X_test, y_test, transform):
    if transform == "log":
        X_test = transformations.scale_log_test(X_test)
    elif transform == "min_max":
        X_test = transformations.scale_min_max_test(X_test)

    # First step: converting to tensor
    x_test_to_tensor = torch.from_numpy(X_test).to(torch.float32)
    y_test_to_tensor = torch.from_numpy(y_test).to(torch.float32)

    return x_test_to_tensor, y_test_to_tensor


def count_classes(y):
    count = [0, 0]
    for i in range(y.shape[0]):
        count[y[i]] += 1
    return count


def balance_dataset(X, y):
    over = imblearn.over_sampling.SMOTE(sampling_strategy=FLAGS.oversample)
    under = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=FLAGS.undersample)
    steps = [('o', over), ('u', under)]
    pipeline = imblearn.pipeline.Pipeline(steps=steps)
    X, y = pipeline.fit_resample(X, y)
    return X, y


def model_performance(model, X, y):
    pred = model(X)
    correct_pred = [0, 0]
    total_pred = [0, 0]
    for i in range(pred.shape[0]):
        pred_class = torch.argmax(pred[i]).item()
        true_class = torch.argmax(y[i]).item()
        if pred_class == true_class:
            correct_pred[true_class] += 1
        total_pred[true_class] += 1
    return sum(correct_pred) / sum(total_pred)


def define_model(trial, num_features):
    in_features = num_features
    n_layers = trial.suggest_int("n_layers", gl_hyp_param["n_layers_low"], gl_hyp_param["n_layers_high"])
    activ_fn = FLAGS.activ_fn
    if activ_fn == "relu":
        activ_fn = nn.ReLU()
    elif activ_fn == "leaky_relu":
        activ_fn = nn.LeakyReLU()
    layers = []
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), gl_hyp_param["layer_size_low"],
                                         gl_hyp_param["layer_size_high"])
        layers.append(nn.Linear(in_features, out_features))
        layers.append(activ_fn)
        # p = trial.suggest_float("dropout_l{}".format(i), gl_hyp_param["dropout_low"], gl_hyp_param["dropout_high"])
        p = 0.225
        layers.append(nn.Dropout(p))
        in_features = out_features
    layers.append(nn.Linear(in_features, 2))
    return nn.Sequential(*layers)


def feature_options(features):
    if features == "queue":
        return ["jobs_ahead_queue", "cpus_ahead_queue", "memory_ahead_queue", "nodes_ahead_queue",
                "time_limit_ahead_queue"]
    elif features == "request":
        return ["priority", "time_limit_raw", "req_cpus", "req_mem", "req_nodes"]
    elif features == "running":
        return ["jobs_running", "cpus_running", "memory_running", "nodes_running", "time_limit_running"]
    elif features == "memory":
        return ["req_mem", "memory_ahead_queue", "memory_running"]
    elif features == "cpu":
        return ["req_cpus", "cpus_ahead_queue", "cpus_running"]
    elif features == "all":
        return ["jobs_ahead_queue", "cpus_ahead_queue", "memory_ahead_queue", "nodes_ahead_queue",
                "time_limit_ahead_queue", "priority", "time_limit_raw", "req_cpus", "req_mem", "req_nodes",
                "jobs_running", "cpus_running", "memory_running", "nodes_running", "time_limit_running"]
    elif features == "job_count":
        return ["priority", "jobs_ahead_queue", "jobs_running"]
    elif features == "queue_request":
        return ["jobs_ahead_queue", "cpus_ahead_queue", "memory_ahead_queue", "nodes_ahead_queue",
                "time_limit_ahead_queue",
                "priority", "time_limit_raw", "req_cpus", "req_mem", "req_nodes"]
    elif features == "user":
        return ["user_jobs_past_day", "user_cpus_past_day", "user_memory_past_day", "user_nodes_past_day", "user_time_limit_past_day"]
    elif features == "qos":
        out = []
        for feature in gl_df.columns:
            if "qos_" in feature:
                out.append(feature)
        return out
    elif features == "partition":
        out = ["par_jobs_running", "par_cpus_running", "par_memory_running", "par_nodes_running", "par_time_limit_running"]
        for feature in gl_df.columns:
            if "partition_" in feature:
                out.append(feature)
        return out
    elif features == "all_more":
        out = ["jobs_ahead_queue", "cpus_ahead_queue", "memory_ahead_queue", "nodes_ahead_queue",
                "time_limit_ahead_queue", "priority", "time_limit_raw", "req_cpus", "req_mem", "req_nodes",
                "jobs_running", "cpus_running", "memory_running", "nodes_running", "time_limit_running"]
        for feature in gl_df.columns:
            if "qos_" in feature:
                out.append(feature)
        for feature in gl_df.columns:
            if "partition_" in feature:
                out.append(feature)
        return out
    elif features == "austin_hypo":
        out = ["priority", "req_cpus", "req_mem", "user_id", "cpus_ahead_queue", "memory_ahead_queue", "par_jobs_running", "par_cpus_running", "user_memory_past_day", "user_cpus_past_day"]
        for feature in gl_df.columns:
            if "partition_" in feature:
                out.append(feature)
        return out
    elif features == "austin_hypo_2":
        return ["jobs_ahead_queue", "cpus_ahead_queue", "memory_ahead_queue", "nodes_ahead_queue",
                "time_limit_ahead_queue",
                "priority", "time_limit_raw", "req_cpus", "req_mem", "req_nodes",
                "user_jobs_past_day", "user_cpus_past_day", "user_memory_past_day", "user_nodes_past_day", "user_time_limit_past_day",
                "par_jobs_running", "par_cpus_running", "par_memory_running", "par_nodes_running", "par_time_limit_running"]
    elif features == "austin_hypo_3":
        return ["priority", "time_limit_raw", "req_cpus", "req_mem", "req_nodes", "partition",
                "par_nodes_available", "par_cpus_available", "par_memory_available",
                "par_nodes_available_running_queue_priority", "par_cpus_available_running_queue_priority",
                "par_memory_available_running_queue_priority"]
    elif features == "austin_hypo_4":
        return ["priority", "time_limit_raw", "req_cpus", "req_mem", "req_nodes", "partition",
                "par_nodes_available_running_queue_priority", "par_cpus_available_running_queue_priority",
                "par_memory_available_running_queue_priority"]
    elif features == "austin_hypo_5":
        return ["priority", "time_limit_raw", "req_cpus", "req_mem", "partition",
                "par_nodes_available_running_queue_priority", "user_time_limit_past_day"]
    elif features == "mid":
        return ["priority", "time_limit_raw", "req_cpus", "req_mem",
                "par_nodes_available_running_queue_priority", "user_time_limit_past_day"]



def train_model(trial, is_ret_model=False):
    X, y_one_hot, feature_mapping_dict = gl_X, gl_y_one_hot, gl_feature_mapping_dict
    features = trial.suggest_categorical("features", gl_hyp_param["features"])
    chosen_features = feature_options(features)
    num_features = len(chosen_features)
    feature_idxs = []
    for feature in chosen_features:
        feature_idxs.append(feature_mapping_dict[feature])
    X = X[:, feature_idxs]

    transform = trial.suggest_categorical("transform", ["none", "log", "min_max"])

    train_dataloader, test_dataloader = create_dataloaders(X, y_one_hot, transform)
    model = define_model(trial, num_features)
    optimizer = optim.Adam(params=model.parameters(), lr=FLAGS.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Run training loop
    train_loss_by_epoch = []
    test_loss_by_epoch = []
    for epoch in range(FLAGS.epochs):
        correct_pred = [0, 0]
        total_pred = [0, 0]
        train_loss = []
        test_loss = []
        model.train()
        for X, y in train_dataloader:
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(loss.item())

        model.eval()
        for X, y in test_dataloader:
            with torch.no_grad():
                pred = model(X)
                loss = loss_fn(pred, y)
                test_loss.append(loss.item())
                for i in range(pred.shape[0]):
                    pred_class = torch.argmax(pred[i]).item()
                    true_class = torch.argmax(y[i]).item()
                    if pred_class == true_class:
                        correct_pred[true_class] += 1
                    total_pred[true_class] += 1

        train_loss_by_epoch.append(np.mean(train_loss))
        test_loss_by_epoch.append(np.mean(test_loss))
        total_acc = sum(correct_pred) / sum(total_pred)
        trial.report(total_acc, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    if is_ret_model:
        return model, feature_idxs, transform
    else:
        return total_acc


def objective(trial):
    total_acc = train_model(trial, is_ret_model=False)
    return total_acc


def detailed_objective(trial, X_test, y_test, y_test_planned):
    model, feature_idxs, transform = train_model(trial, is_ret_model=True)

    classes = ["Under 10min", "Over10min"]
    X_test = X_test[:, feature_idxs]
    X_test, y_test = transform_test(X_test, y_test, transform)
    correct_pred = [0, 0]
    total_pred = [0, 0]
    over_hour_total = 0
    over_hour_correct = 0
    misses = 0
    marginal_misses = 0

    model.eval()
    for i in range(y_test.shape[0]):
        pred = model(X_test[i])
        true_class = torch.argmax(y_test[i]).item()
        pred_class = torch.argmax(pred).item()
        raw_mins = y_test_planned[i] / 60
        if raw_mins >= 60:
            over_hour_total += 1
        if pred_class == true_class:
            correct_pred[true_class] += 1
            if raw_mins >= 60:
                over_hour_correct += 1
        else:
            # print(f"pred={pred_class}, true={true_class}, raw={raw_mins}mins")
            misses += 1
            if raw_mins >= 3 and raw_mins <= 15:
                marginal_misses += 1
        total_pred[true_class] += 1

    total_acc = sum(correct_pred) / sum(total_pred)
    for i in range(len(correct_pred)):
        if total_pred[i] != 0:
            print(f"{classes[i]} accuracy: {correct_pred[i] / total_pred[i]}")
    print(f"Test Accuracy: {total_acc}")
    tmp = over_hour_correct / over_hour_total
    print(f"over60mins acc: {tmp}   ------ total over 60 min: {over_hour_total}")
    percent_misses_marginal = marginal_misses / misses
    print(f"percent of misses that are marginal: {percent_misses_marginal}    ------ total misses {misses}")

    torch.save(model.state_dict(), "classify_model.pt")

    return total_acc


def load_data():
    global gl_df
    global gl_X
    global gl_y_one_hot
    global gl_feature_mapping_dict
    num_jobs = FLAGS.n_jobs
    read_all = True if num_jobs == 0 else False
    gl_df = read_db.read_to_df(table="jobs_everything_tmp", read_all=read_all, jobs=num_jobs, order_by="eligible", condense_same_times=FLAGS.condense_jobs)
    print(len(gl_df.index))
    gl_df = gl_df.sort_values(by=["eligible"], ascending=True)
    ten_perc = int(num_jobs / 10)
    
    # Adding support for one hot
    gl_df = transformations.make_one_hot(gl_df, "partition")
    gl_df = transformations.make_one_hot(gl_df, "qos", 3)

    y = gl_df["planned"].to_numpy()
    y_test_planned = y[-ten_perc:]
    y = get_class_labels(y, threshold=600)
    gl_X = gl_df.drop(["planned"], axis=1)
    gl_X = gl_X._get_numeric_data()
    gl_feature_mapping_dict = {}
    for feature_name in gl_X.columns:
        gl_feature_mapping_dict[feature_name] = gl_X.columns.get_loc(feature_name)
    gl_X = gl_X.to_numpy().astype(np.float32)

    X_test = gl_X[-ten_perc:]
    y_test = y[-ten_perc:]
    gl_X = gl_X[:-ten_perc]
    y = y[:-ten_perc]

    gl_X, y = balance_dataset(gl_X, y)
    gl_y_one_hot = to_one_hot(y, num_classes=2).numpy()
    y_test = to_one_hot(y_test, num_classes=2).numpy()
    return X_test, y_test, y_test_planned


def start_trials():
    X_test, y_test, y_test_planned = load_data()
    run_study = neptune.init_run(
        project="queue/trout",
        api_token=config_file.neptune_api_token,
        tags=["classify"]
    )

    sampler = TPESampler(n_startup_trials=25)
    study = optuna.create_study(direction='maximize', study_name='namez', sampler=sampler)
    study.optimize(objective, n_trials=100)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    trial = study.best_trial

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("Best trial:")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    run_study["info/n_jobs"] = FLAGS.n_jobs
    run_study["valid/score"] = trial.value
    npt_utils.log_study_metadata(study, run_study)
    total_acc = detailed_objective(trial, X_test, y_test, y_test_planned)
    run_study["end_test/score"] = total_acc

    run_study.stop()


def main(argv):
    start_trials()


if __name__ == '__main__':
    app.run(main)
