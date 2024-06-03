import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from create_train_and_validation_datasets_for_regression import create_train_and_validation_datasets_for_regression


def my_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_fname", type=str, default="../data/lfp-abby/processed/single_trials.pkl")
    parser.add_argument("--percent_train", type=float, default=0.8)
    parser.add_argument("--number_repetitions", type=int, default=20)
    parser.add_argument("--save_fname", type=str, default="regression-models.pkl")
    return parser


def my_rmse(y: np.ndarray, y_hat: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute the root mean squared error between y and y_hat along the specified axis.

    Args:
        y (np.ndarray): True values.
        y_hat (np.ndarray): Predicted values.
        axis (int, optional): Axis for computing the mean. Defaults to 0.

    Returns:
        np.ndarray: RMSE values.
    """
    return np.sqrt(np.mean((y - y_hat) ** 2, axis=axis))


if __name__ == "__main__":
    input_noise_stds = [0, 0.05, 0.1, 0.2, 0.4, 0.8]

    parser = my_parser()
    args = parser.parse_args()

    data = pd.read_pickle(args.data_fname)

    experiments_df = data[["fish_id", "zone"]].groupby(["fish_id", "zone"], as_index=False).apply(lambda x: x.iloc[0])
    zones = experiments_df["zone"].unique()
    experiments_df = pd.concat(
        [experiments_df, pd.DataFrame(dict(fish_id=["fish"] * len(zones), zone=zones))], axis=0, ignore_index=True
    )

    regression_models = pd.DataFrame()
    for i in range(args.number_repetitions):
        print(f"Repetition {i+1}/{args.number_repetitions}. Input noise std: ", end="")
        for input_noise_std in input_noise_stds:
            print(f"{input_noise_std:.2f}, ", end="")
            for _, experiment in experiments_df.iterrows():
                fish_id = experiment["fish_id"]
                zone = experiment["zone"]

                train_data, valid_data = create_train_and_validation_datasets_for_regression(
                    dataframe=data, fish_id=fish_id, zone=zone, percent_train=args.percent_train
                )

                regression = LinearRegression().fit(
                    train_data[0] + np.random.randn(*train_data[0].shape) * input_noise_std, train_data[1]
                )

                train_error = my_rmse(train_data[1], regression.predict(train_data[0]))
                valid_error = my_rmse(valid_data[1], regression.predict(valid_data[0]))

                regression_models = pd.concat(
                    [
                        regression_models,
                        pd.DataFrame(
                            dict(
                                fish_id=[fish_id],
                                zone=[zone],
                                input_noise_std=[input_noise_std],
                                model_id=[i],
                                train_error=[train_error],
                                valid_error=[valid_error],
                                coefficients=[regression.coef_],
                                regression=[regression],
                            )
                        ),
                    ],
                    axis=0,
                    ignore_index=True,
                )
        print()
    regression_models.to_pickle(args.save_fname)
