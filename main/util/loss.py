import torch
from torch import nn
from torch.nn import functional as F


class PartialLoss(nn.Module):
    def __init__(self, confidence, confidence_dir, conf_ema_m=0.99):
        super().__init__()
        self.init_confidence = confidence
        self.confidence = confidence
        self.confidence_dir = confidence_dir
        self.conf_ema_m = conf_ema_m

    def set_conf_ema_m(self, epoch, start=0.95, end=0.8):
        epochs = 200
        self.conf_ema_m = 1.0 * epoch / epochs * (end - start) + start

    def forward(self, outputs, index):
        logsm_outputs = F.log_softmax(outputs, dim=1)
        final_outputs = logsm_outputs * self.confidence[index, :]
        average_loss = -((final_outputs).sum(dim=1)).mean()
        return average_loss

    def confidence_update(self, temp_un_conf, batch_index, batchY):
        with torch.no_grad():
            _, prot_pred = (temp_un_conf * batchY).max(dim=1)
            pseudo_label = F.one_hot(prot_pred, batchY.shape[1]).float().detach()
            self.confidence[batch_index, :] = (
                self.conf_ema_m * self.confidence[batch_index, :]
                + (1 - self.conf_ema_m) * pseudo_label
            )
        return None
