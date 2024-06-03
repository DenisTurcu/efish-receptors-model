import argparse
from os import path
import numpy as np
import pandas as pd
from glob import glob
from torch.utils.data import DataLoader
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from Convolutional_Mormyromast import ConvMormyromast
from Convolutional_Mormyromast_PL import ConvMormyromast_PL
from lfp_response_dataset import create_train_and_validation_datasets


def my_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_trained_models", type=str, default="./trained_filters")
    parser.add_argument("--save_filename", type=str, default="./filters-models.pkl")
    return parser


def load_trained_model_summary(path_to_version: str, base_model: ConvMormyromast) -> pd.DataFrame:
    """Load summary of the trained model.

    Args:
        path_to_version (str): Path to the version directory for one trained model.
        base_model (ConvMormyromast): The base model to load from checkpoint.

    Returns:
        pd.Series: Summary of the trained model.
    """
    experiment_name = path_to_version.split("/")[-2].split("-")
    fish_id = experiment_name[0]
    zone = experiment_name[1]
    input_noise_std = float(experiment_name[2].replace("p", "."))
    model_id = int(path_to_version.split("/")[-1].split("_")[-1])

    version_files = glob(path_to_version + "/**", recursive=True)
    checkpoint = [x for x in version_files if ".ckpt" in x]
    assert len(checkpoint) == 1, "There must be a single checkpoint."
    checkpoint = checkpoint[0]
    events_file = [x for x in version_files if "tfevents" in x]
    assert len(events_file) == 1, "There must be a single events file."
    events_file = events_file[0]

    model_PL = ConvMormyromast_PL.load_from_checkpoint(
        checkpoint, model=base_model, input_noise_std=input_noise_std, learning_rate=1e-3
    )
    model_filter = model_PL.model.conv_list[0].weight.detach().cpu().squeeze().numpy()
    model_bias = model_PL.model.conv_list[0].bias.detach().cpu().numpy()

    event_acc = EventAccumulator(events_file)
    event_acc.Reload()
    train_error = np.mean([x.value for x in event_acc.Scalars("train_loss")[-10:]])
    valid_error = np.mean([x.value for x in event_acc.Scalars("val_loss")[-10:]])

    return pd.DataFrame(
        dict(
            fish_id=fish_id,
            zone=zone,
            input_noise_std=input_noise_std,
            model_id=model_id,
            model_filter=(model_filter,),
            model_bias=model_bias,
            train_error=train_error,
            valid_error=valid_error,
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

    # define the base model of trained models to load from checkpoint
    base_model = ConvMormyromast(
        input_length=next(iter(train_loader))[0].shape[2],
        input_channels=1,
        conv_layer_fraction_widths=[1],
        conv_output_channels=1,
        conv_stride=25,
        N_receptors=1,
    )

    path_to_versions = glob(args.path_to_trained_models + "/*/*")
    path_to_versions.sort()
    path_to_print = ""
    trained_models_summary = pd.DataFrame()
    for path_to_version in path_to_versions:
        temp_path_to_print = '/'.join(path_to_version.split("/")[:-1])
        if temp_path_to_print != path_to_print:
            path_to_print = temp_path_to_print
            print(path_to_print)
        trained_models_summary = pd.concat(
            [trained_models_summary, load_trained_model_summary(path_to_version, base_model)], axis=0, ignore_index=True
        )
    print("Done.")

    # save the summary to .pkl
    trained_models_summary.to_pickle(args.save_filename)
