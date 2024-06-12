import argparse
import numpy as np
import pandas as pd
from glob import glob
import torch
from torch.utils.data import DataLoader
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from lfp_response_dataset import create_train_and_validation_datasets


def my_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_trained_models", type=str, default="./trained-by-fish-zone-session")
    parser.add_argument("--save_filename", type=str, default="./filters-models-by-fish-zone-session.pkl")
    return parser


def load_trained_model_summary(path_to_version: str) -> pd.DataFrame:
    """Load summary of the trained model.

    Args:
        path_to_version (str): Path to the version directory for one trained model.
            E.g. `./trained_filters/fish_01-mz-0p05/version_0`.

    Returns:
        pd.Series: Summary of the trained model.
    """
    experiment_name = path_to_version.split("/")[-2].split("-")
    fish_id = experiment_name[0]
    zone = experiment_name[1]
    session_id = experiment_name[2]
    input_noise_std = float(experiment_name[3].replace("p", "."))
    model_id = int(path_to_version.split("/")[-1].split("_")[-1])

    version_files = glob(path_to_version + "/**", recursive=True)
    checkpoint = [x for x in version_files if ".ckpt" in x]
    assert len(checkpoint) == 1, "There must be a single checkpoint."
    checkpoint = checkpoint[0]
    events_file = [x for x in version_files if "tfevents" in x]
    assert len(events_file) == 1, "There must be a single events file."
    events_file = events_file[0]

    checkpoint = torch.load(checkpoint, map_location=torch.device("cpu"))
    model_filter = checkpoint["state_dict"]["model.conv_list.0.weight"].numpy().squeeze()
    model_bias = checkpoint["state_dict"]["model.conv_list.0.bias"].numpy()
    bn_weight = checkpoint["state_dict"]["model.bn.weight"].numpy()
    bn_bias = checkpoint["state_dict"]["model.bn.bias"].numpy()
    bn_mean = checkpoint["state_dict"]["model.bn.running_mean"].numpy()
    bn_var = checkpoint["state_dict"]["model.bn.running_var"].numpy()

    event_acc = EventAccumulator(events_file)
    event_acc.Reload()
    train_error = np.sqrt(np.mean([x.value for x in event_acc.Scalars("train_loss")[-10:]]))
    valid_error = np.sqrt(np.mean([x.value for x in event_acc.Scalars("val_loss")[-10:]]))

    return pd.DataFrame(
        dict(
            fish_id=fish_id,
            zone=zone,
            session_id=session_id,
            input_noise_std=input_noise_std,
            model_id=model_id,
            model_filter=(model_filter,),
            model_bias=model_bias,
            train_error=train_error,
            valid_error=valid_error,
            bn_weight=bn_weight,
            bn_bias=bn_bias,
            bn_mean=bn_mean,
            bn_var=bn_var,
        ),
        index=[0],
    )


if __name__ == "__main__":
    parser = my_parser()
    args = parser.parse_args()

    # load data to get the input length
    data_fname = "../data/lfp-abby/processed/single_trials.pkl"
    data = pd.read_pickle(data_fname)
    train_dataset, valid_dataset = create_train_and_validation_datasets(data, percent_train=0.8)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)

    path_to_versions = glob(args.path_to_trained_models + "/*/*")
    path_to_versions.sort()
    path_to_print = ""
    trained_models_summary = pd.DataFrame()
    for path_to_version in path_to_versions:
        temp_path_to_print = "/".join(path_to_version.split("/")[:-1])
        if temp_path_to_print != path_to_print:
            path_to_print = temp_path_to_print
            print(path_to_print)
        trained_models_summary = pd.concat(
            [trained_models_summary, load_trained_model_summary(path_to_version)], axis=0, ignore_index=True
        )
    print("Done.")

    # save the summary to .pkl
    trained_models_summary.to_pickle(args.save_filename)
