import numpy as np
import pandas as pd


def create_train_and_validation_datasets_for_regression(
    dataframe: pd.DataFrame,
    fish_id: str = "",
    zone: str = "mz",
    percent_train: float = 0.8,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Create train and validation datasets for linear regression models.

    Args:
        dataframe (pd.DataFrame): Dataframe containing the Python processed LFP responses.
        fish_id (str): The fish ID , e.g. `fish_01`. If empty (i.e. `''`), all fish IDs are considered.
        zone (str): Zone of the recorded data - `mz` or `dlz`
        percent_train (float, optional): Percent of stimuli to use for training. Defaults to 0.8.

    Returns:
        tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]: Train [X, Y] and validation [X, Y]
            datasets.
    """
    # below line allows to match all fish when `fish_id == ''` or `fish_id == 'fish'`
    fish_id_match_df_indices = dataframe["fish_id"].apply(lambda x: fish_id in x)
    zone_match_df_indices = dataframe["zone"] == zone

    dataframe = dataframe[zone_match_df_indices & fish_id_match_df_indices]

    stimulus_markers = dataframe["stimulus_marker"].unique()
    num_stimuli = len(stimulus_markers)
    num_trained_stimuli = int(num_stimuli * percent_train)
    stimuli_permutation = np.random.permutation(num_stimuli)
    train_stimuli = stimulus_markers[stimuli_permutation[:num_trained_stimuli]]
    valid_stimuli = stimulus_markers[stimuli_permutation[num_trained_stimuli:]]

    train_df = dataframe[dataframe["stimulus_marker"].apply(lambda x: x in train_stimuli)]
    valid_df = dataframe[dataframe["stimulus_marker"].apply(lambda x: x in valid_stimuli)]

    train_data = (
        np.vstack(train_df["waveform"]).astype(np.float32),  # type: ignore
        np.hstack(train_df["lfp_response_modulation"]).astype(np.float32),  # type: ignore
    )
    valid_data = (
        np.vstack(valid_df["waveform"]).astype(np.float32),  # type: ignore
        np.hstack(valid_df["lfp_response_modulation"]).astype(np.float32),  # type: ignore
    )
    return train_data, valid_data
