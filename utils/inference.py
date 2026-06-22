import torch
from torch.utils.data import TensorDataset, DataLoader


@torch.no_grad()
def predict_logits_batched(model, psd, de, plv, wpli, batch_size=64, device=None):
    if device is None:
        device = next(model.parameters()).device
    ds = TensorDataset(psd, de, plv, wpli)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    outs = []
    model.eval()
    for xb_psd, xb_de, xb_plv, xb_wpli in dl:
        xb_psd = xb_psd.to(device)
        xb_de = xb_de.to(device)
        xb_plv = xb_plv.to(device)
        xb_wpli = xb_wpli.to(device)
        outs.append(model(xb_psd, xb_de, xb_plv, xb_wpli).cpu())
    return torch.cat(outs, dim=0)
