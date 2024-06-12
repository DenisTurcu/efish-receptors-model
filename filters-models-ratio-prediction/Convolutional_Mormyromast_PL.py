import numpy as np
import torch
import torch.nn as nn
import lightning as L


class ConvMormyromast_PL(L.LightningModule):
    def __init__(self, model: nn.Module, input_noise_std: float, learning_rate: float):
        super().__init__()
        self.model = model
        self.input_noise_std = input_noise_std
        self.learning_rate = learning_rate

    def training_step(self, batch):
        x, x0, y = batch
        x += torch.randn(*x.shape).to(x.device) * self.input_noise_std
        y_hat = self.model(x, x0)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        tensorboard = self.logger.experiment
        model_filter = np.tile(self.model.conv_list[0].weight.data.squeeze().cpu().numpy(), (200, 1))
        model_filter = model_filter - model_filter.min()
        model_filter = model_filter / model_filter.ptp()
        tensorboard.add_image(
            "filter",
            model_filter,
            global_step=self.global_step,
            dataformats="HW",
        )
        return loss

    def validation_step(self, batch):
        x, x0, y = batch
        y_hat = self.model(x, x0)
        val_loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", val_loss, on_epoch=True)
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
