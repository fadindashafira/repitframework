from fvmn import FVMNetwork
from repitframework.config import TrainingConfig
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

class FVMNLightning(pl.LightningModule):
    def __init__(self, training_config:TrainingConfig):
        super().__init__()
        self.training_config = training_config
        self.model = FVMNetwork(training_config)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss_mse = F.mse_loss(y_hat, y)
        loss_l1 = F.l1_loss(y_hat, y)
        self.log("val_loss_mse", loss_mse, on_epoch=True)
        self.log("val_loss_mae", loss_l1, on_epoch=True)
        return torch.max(torch.abs(y_hat - y))

    def test_step(self, batch, batch_idx):
        x,y = batch
        y_hat = self.model(x)
        loss_mse = F.mse_loss(y_hat, y)
        self.log("test_loss", loss_mse, on_epoch=True)
        return loss_mse

    def configure_optimizers(self):
        optimizer = self.training_config.optimizer(self.model.parameters(), lr=self.training_config.learning_rate)
        return optimizer

if __name__ == "__main__":
    training_config = TrainingConfig()
    dataset = torch.utils.data.TensorDataset(torch.randn(100,15), torch.randn(100,3))
    loader = torch.utils.data.DataLoader(dataset, batch_size=10)
    model = FVMNLightning(training_config)
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, loader)
    