import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class LfpResponseDataset(Dataset):
    """Dataset class for the LFP responses."""

    def __init__(self, dataframe: pd.DataFrame, fish_id: str, zone: str, mean_response: bool = False):
        """Initialize the LFP response dataset.

        Args:
            dataframe (pd.DataFrame): Dataframe containing the Python processed LFP responses.
            fish_id (str): The fish ID , e.g. `fish_01`.
            zone (str): Zone of the recorded data - `mz` or `dlz`
            mean_response (bool, optional): Whether to use mean response instead of single trials. Defaults to False.
        """
        super(LfpResponseDataset, self).__init__()
        self.dataframe = dataframe
        self.fish_id = fish_id
        self.zone = zone
        self.mean_response = mean_response
        self._process_dataframe()

    def __len__(self):
        return self.response.shape[0]

    def __getitem__(self, index):
        return (self.stimuli[index].reshape(1, -1), self.response[index].reshape(-1))

    def _process_dataframe(self):
        """Process the Dataframe containing the Python processed LFP responses to
        extract the stimuli and corresponding responses."""
        df = self.dataframe[(self.dataframe["zone"] == self.zone) & (self.dataframe["fish_id"] == self.fish_id)]

        if self.mean_response:
            df = df.groupby("stimulus_marker").apply(lambda x: x[["waveform", "mean_lfp_response_modulation"]].iloc[0])
            self.stimuli = np.vstack(df["waveform"]).astype(np.float32)  # type: ignore
            self.response = np.hstack(df["mean_lfp_response_modulation"]).astype(np.float32)  # type: ignore
        else:
            self.stimuli = np.vstack(df["waveform"]).astype(np.float32)  # type: ignore
            self.response = np.hstack(df["lfp_response_modulation"]).astype(np.float32)  # type: ignore


def create_train_and_validation_datasets(
    dataframe: pd.DataFrame, fish_id: str, zone: str, mean_response: bool = False, percent_train: float = 0.8
) -> tuple[Dataset, Dataset]:
    """Create training and validation datasets from the processed LFP responses. Split the training and validation
    to ensure that validation data contains stimuli that are not present in the training data, as opposed to simply
    containing single trials from (possibly) the same stimuli from the training data.

    Args:
        dataframe (pd.DataFrame): Dataframe containing the Python processed LFP responses.
        fish_id (str): The fish ID , e.g. `fish_01`.
        zone (str): Zone of the recorded data - `mz` or `dlz`
        mean_response (bool, optional): Whether to use mean response instead of single trials. Defaults to False.
        percent_train (float, optional): Percent of stimuli to use for training. Defaults to 0.8.

    Returns:
        tuple[Dataset, Dataset]: Train and validation datasets.
    """
    dataframe = dataframe[(dataframe["zone"] == zone) & (dataframe["fish_id"] == fish_id)]

    stimulus_markers = dataframe["stimulus_marker"].unique()
    num_stimuli = len(stimulus_markers)
    num_trained_stimuli = int(num_stimuli * percent_train)
    stimuli_permutation = np.random.permutation(num_stimuli)
    train_stimuli = stimulus_markers[stimuli_permutation[:num_trained_stimuli]]
    valid_stimuli = stimulus_markers[stimuli_permutation[num_trained_stimuli:]]

    train_dataset = LfpResponseDataset(
        dataframe[dataframe["stimulus_marker"].apply(lambda x: x in train_stimuli)], fish_id, zone, mean_response
    )
    valid_dataset = LfpResponseDataset(
        dataframe[dataframe["stimulus_marker"].apply(lambda x: x in valid_stimuli)], fish_id, zone, mean_response
    )

    return train_dataset, valid_dataset
