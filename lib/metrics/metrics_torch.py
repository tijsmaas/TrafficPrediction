import numpy as np
import torch


# Create a mask that is 1 for all values and 0 for all unknowns
def mask_nan_labels_torch(labels, null_val=0.):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    return mask


def calculate_metrics_torch(preds, labels, null_val=0.):
    mask = mask_nan_labels_torch(labels, null_val)
    mse = (preds - labels) ** 2
    mae = torch.abs(preds - labels)
    mape = mae / labels
    mae, mse, mape = [mask_and_fillna_torch(l, mask) for l in [mae, mse, mape]]
    rmse = torch.sqrt(mse)
    return mae, rmse, mape


def mask_and_fillna_torch(loss, mask):
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mse_torch(preds, labels, null_val=np.nan):
    mask = mask_nan_labels_torch(labels, null_val)
    mse = (preds - labels) ** 2
    return mask_and_fillna_torch(mse, mask)


def masked_rmse_torch(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse_torch(preds=preds, labels=labels, null_val=null_val))


def masked_mae_torch(preds, labels, null_val=np.nan):
    mask = mask_nan_labels_torch(labels, null_val)
    mae = torch.abs(preds - labels)
    return mask_and_fillna_torch(mae, mask)


def masked_mae_loss(scaler, null_val):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
        mae = masked_mae_tf(preds=preds, labels=labels, null_val=null_val)
        return mae

    return loss

def masked_mape_torch(preds, labels, null_val=np.nan):
    mask = mask_nan_labels_torch(labels, null_val)
    mape = torch.abs(preds - labels) / labels
    return mask_and_fillna_torch(mape, mask)


def metric(pred, real):
    mae = masked_mae_torch(pred, real, 0.0).item()
    mape = masked_mape_torch(pred, real, 0.0).item()
    rmse = masked_rmse_torch(pred, real, 0.0).item()
    return mae, rmse, mape
