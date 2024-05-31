import numpy as np
import pandas as pd
import scipy.io as sio
import mat73


def load_waveform(fname: str, fraction_of_max: float = 2e-2) -> pd.DataFrame:
    """Load waveform .mat file and return a DataFrame with the data.

    Args:
        fname (str): File name containing the waveform data.
        fraction_of_max (float, optional): Eliminate the 0-padding of the waveforms based on the fraction of maximum
            absolute value of the base EOD. Every sample smaller than this fraction of max(abs(*)) is considered 0;
            then, all preceding and trailing 0s are removed. Defaults to 2e-2.

    Returns:
        pd.DataFrame: DataFrame containing the waveform data.
    """
    data = sio.loadmat(fname)
    k = list(data.keys())[-1]
    data = data[k]

    # remove preceding and trailing 0s
    base_waveform = data["waveform"][0][-1].flatten()
    max_base_waveform = np.abs(base_waveform).max()
    ids_good = np.where(np.abs(base_waveform) > fraction_of_max * max_base_waveform)[0]

    waveforms = []
    for i in range(len(data["waveform"][0]) - 1):
        waveform = data["waveform"][0][i].flatten()
        waveform = waveform[ids_good[0] : ids_good[-1]]  # noqa: E203
        waveform = waveform / max_base_waveform
        waveform = waveform - waveform[0]
        waveforms.append(waveform)
    base_waveform = base_waveform[ids_good[0] : ids_good[-1]] / max_base_waveform  # noqa: E203
    base_waveform = base_waveform - base_waveform[0]

    return pd.DataFrame(
        dict(
            stimulus_fname=[x[0] for x in data["fname"][0]],
            stimulus_marker=[x[0][0] for x in data["marker"][0]],
            stimulus_sampling_rate=[x[0][0] for x in data["samprate"][0]],
            stimulus_resistance=[x[0][0] for x in data["RO"][0]],
            stimulus_capacitance=[x[0][0] for x in data["CO"][0]],
            stimulus_amplitude_modulation=[x[0][0] for x in data["amp_mod"][0]],
            stimulus_waveform_modulation=[x[0][0] for x in data["wav_mod"][0]],
            stimulus_value_max=[x[0][0] for x in data["maxv"][0]],
            stimulus_value_min=[x[0][0] for x in data["minv"][0]],
            waveform=waveforms + [base_waveform],
            base_waveform=[base_waveform] * len(data["waveform"][0]),
        )
    )


def load_lfp_data(fname: str, lfp_id_min: int = 301, lfp_id_max: int = 512) -> pd.DataFrame:
    """Load the LFP data from the .mat file and return a DataFrame with the data. The LFP traces are processed to
    extract single trial responses. The single trial responses are then used to compute the response modulation.
    The single trial responses are extracted by taking the minimum value of the LFP trace between the minimum and
    maximum indices provided. The response modulation is computed as the ratio of the single trial response to the
    average base response minus 1.

    Args:
        fname (str): File name containing the LFP data.

        The LFP data is a long trace and not all of it is relevant for extracting single trial responses. The following
        two variables specify the range of LFP array indices that are relevant for extracting the single trial response.
        The default values correspond to searching for the response between 1ms and 8ms after the stimulus onset.
            lfp_id_min (int, optional): Minimum index to start looking for the LFP response. Defaults to 301.
            lfp_id_max (int, optional): Maximum index to stop looking for the LFP response. Defaults to 512.

    Returns:
        pd.DataFrame: DataFrame containing the processed LFP data.
    """
    data_means = mat73.loadmat(fname)["LfpMeans"]
    experiment_date, session_id, zone, _ = fname.split("/")[-1].split("-")

    lfp_means_time = data_means["lfptime"][-1]
    lfp_sampling_rate = data_means["vdt"][-1]

    # extract the lfp traces
    lfp_trace = data_means["lfpNorm"][:-1]
    mean_lfp_trace = data_means["lfpMean"][:-1]
    base_lfp_trace = data_means["b1lfpNorm"][:-1]
    base_mean_lfp_trace = data_means["b1lfpMean"][:-1]

    def process_response(list_of_traces):
        return [x.T[lfp_id_min:lfp_id_max].min(axis=0) for x in list_of_traces]

    # compute the lfp responses for single trials
    lfp_response = [process_response(y) for y in lfp_trace]
    mean_lfp_response = [process_response(y.reshape(y.shape[0], -1).T) for y in mean_lfp_trace]
    base_lfp_response = [process_response(y) for y in base_lfp_trace]
    base_mean_lfp_response = [process_response(y.reshape(y.shape[0], -1).T) for y in base_mean_lfp_trace]

    def process_response_modulation(list_of_responses, base):
        return [list_of_responses[i] / base[i] - 1 for i in range(len(list_of_responses))]

    # compute the lfp response modulation for single trials
    lfp_response_modulation = [
        process_response_modulation(y, base) for y, base in zip(lfp_response, base_mean_lfp_response)
    ]

    # compute the lfp response modulation for the whole trial
    mean_lfp_response_modulation = [
        (np.array(mean_lfp_response[i]) / np.array(base_mean_lfp_response[i])).mean() - 1
        for i in range(len(mean_lfp_response))
    ]

    return pd.DataFrame(
        dict(
            stimulus_marker=[int(x) for x in data_means["marker"][:-1]],
            number_bouts=[int(x) for x in data_means["bouts"][:-1]],
            lfp_trace=lfp_trace,
            mean_lfp_trace=mean_lfp_trace,
            base_lfp_trace=base_lfp_trace,
            base_mean_lfp_trace=base_mean_lfp_trace,
            lfp_response=lfp_response,
            mean_lfp_response=mean_lfp_response,
            base_lfp_response=base_lfp_response,
            base_mean_lfp_response=base_mean_lfp_response,
            lfp_response_modulation=lfp_response_modulation,
            mean_lfp_response_modulation=mean_lfp_response_modulation,
            stimulus_amplitude_modulation=data_means["ampmod"][:-1],
            stimulus_waveform_modulation=data_means["wavmod"][:-1],
            lfp_sampling_rate=[lfp_sampling_rate] * len(data_means["marker"][:-1]),
            lfp_times=[lfp_means_time] * len(data_means["marker"][:-1]),
            experiment_date=[experiment_date] * len(data_means["marker"][:-1]),
            session_id=[session_id] * len(data_means["marker"][:-1]),
            zone=[zone] * len(data_means["marker"][:-1]),
        )
    )


def expand_data_to_single_trials(dfrow: pd.Series) -> pd.DataFrame:
    """Expand the DataFrame row containing the LFP data to single trials.

    Args:
        dfrow (pd.Series): DataFrame row containing the LFP data, as loaded by `load_lfp_data`.

    Returns:
        pd.DataFrame: DataFrame containing the single trial LFP data on each row, as an expansion of the input row.
    """
    num_bouts = dfrow["number_bouts"]

    new_df = pd.DataFrame()
    for i in range(num_bouts):
        lfp_trace = list(dfrow["lfp_trace"][i])
        mean_lfp_trace = [dfrow["mean_lfp_trace"].reshape(dfrow["mean_lfp_trace"].shape[0], -1).T[i]] * len(lfp_trace)
        base_mean_lfp_trace = [
            dfrow["base_mean_lfp_trace"].reshape(dfrow["base_mean_lfp_trace"].shape[0], -1).T[i]
        ] * len(lfp_trace)
        lfp_response = list(dfrow["lfp_response"][i])
        mean_lfp_response = [dfrow["mean_lfp_response"][i]] * len(lfp_trace)
        base_mean_lfp_response = [dfrow["base_mean_lfp_response"][i]] * len(lfp_trace)
        lfp_response_modulation = list(dfrow["lfp_response_modulation"][i])

        new_df = pd.concat(
            [
                new_df,
                pd.DataFrame(
                    dict(
                        lfp_trace=lfp_trace,
                        mean_lfp_trace=mean_lfp_trace,
                        base_mean_lfp_trace=base_mean_lfp_trace,
                        lfp_response=lfp_response,
                        mean_lfp_response=mean_lfp_response,
                        base_mean_lfp_response=base_mean_lfp_response,
                        lfp_response_modulation=lfp_response_modulation,
                    ),
                ),
            ],
            axis=0,
            ignore_index=True,
        )

    for col_name in [
        "stimulus_marker",
        "number_bouts",
        # "lfp_trace",
        # "mean_lfp_trace",
        # DELETED "base_lfp_trace",
        # "base_mean_lfp_trace",
        # "lfp_response",
        # "mean_lfp_response",
        # DELETED "base_lfp_response",
        # "base_mean_lfp_response",
        # "lfp_response_modulation",
        "mean_lfp_response_modulation",
        "stimulus_amplitude_modulation_x",
        "stimulus_waveform_modulation_x",
        "lfp_sampling_rate",
        "lfp_times",
        "fish_id",
        "experiment_date",
        "session_id",
        "zone",
        "paired_experiment",
        "stimulus_fname",
        "stimulus_sampling_rate",
        "stimulus_resistance",
        "stimulus_capacitance",
        "stimulus_amplitude_modulation_y",
        "stimulus_waveform_modulation_y",
        "stimulus_value_max",
        "stimulus_value_min",
        "waveform",
        "base_waveform",
    ]:
        new_df[col_name] = [dfrow[col_name]] * new_df.shape[0]
    return new_df
