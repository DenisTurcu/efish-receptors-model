import argparse
import pandas as pd
import lightning as L
from lightning.pytorch import loggers as pl_loggers
from torch.utils.data import DataLoader
from Convolutional_Mormyromast import ConvMormyromast
from Convolutional_Mormyromast_PL import ConvMormyromast_PL
from lfp_response_dataset import create_train_and_validation_datasets


def my_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_fname", type=str, default="../data/lfp-abby/processed/single_trials.pkl")
    parser.add_argument("--save_folder", type=str, default="lightning_logs")
    parser.add_argument("--percent_train", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--number_repetitions", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    return parser


if __name__ == "__main__":
    input_noise_stds = [0.05, 0.1, 0.2, 0.4, 0.8]

    parser = my_parser()
    args = parser.parse_args()

    data = pd.read_pickle(args.data_fname)

    experiments_df = (
        data[["fish_id", "zone", "session_id"]]
        .groupby(["fish_id", "zone", "session_id"], as_index=False)
        .apply(lambda x: x.iloc[0])
    )
    zones = experiments_df["zone"].unique()
    experiments_df = pd.concat(
        [experiments_df, pd.DataFrame(dict(fish_id=["fish"] * len(zones), zone=zones, session_id=""))],
        axis=0,
        ignore_index=True,
    )

    for i in range(args.number_repetitions):
        for input_noise_std in input_noise_stds:
            for _, experiment in experiments_df.iterrows():
                fish_id = experiment["fish_id"]
                zone = experiment["zone"]
                session_id = experiment["session_id"]

                train_dataset, valid_dataset = create_train_and_validation_datasets(
                    data, fish_id=fish_id, zone=zone, session_id=session_id, percent_train=args.percent_train
                )
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
                valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

                model = ConvMormyromast(
                    input_length=next(iter(train_loader))[0].shape[2],
                    input_channels=1,
                    conv_layer_fraction_widths=[1],
                    conv_output_channels=1,
                    conv_stride=25,
                    N_receptors=1,
                )

                model_PL = ConvMormyromast_PL(model, input_noise_std=input_noise_std, learning_rate=args.learning_rate)

                logger = pl_loggers.TensorBoardLogger(
                    save_dir=args.save_folder,
                    name=f"{fish_id}-{zone}-{session_id}-{str(input_noise_std).replace('.', 'p')}",
                )
                trainer = L.Trainer(max_steps=args.max_steps, logger=logger, devices=[args.device])
                trainer.fit(model=model_PL, train_dataloaders=train_loader, val_dataloaders=valid_loader)
