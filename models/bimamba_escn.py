import torch
import torch.nn as nn
from .gca import GCA
from .gass import GASS


class BiMambaESCN(nn.Module):

    def __init__(
        self,
        chans: int = 10,
        bands: int = 5,
        d_model: int = 64,
        d_state: int = 32,
        fusion: str = "mean",
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gca = GCA(feature_dim=bands, dropout=0.0)
        self.gass = GASS(chans=chans, bands=bands, d_model=d_model, d_state=d_state, reduction=4, fusion=fusion)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(chans * bands, num_classes),
        )

    def forward(self, psd: torch.Tensor, de: torch.Tensor, plv: torch.Tensor, wpli: torch.Tensor) -> torch.Tensor:
        b, t, c, f = psd.shape
        psd_e = psd.reshape(b * t, c, f)
        de_e = de.reshape(b * t, c, f)
        plv_e = plv.reshape(b * t, c, f)
        wpli_e = wpli.reshape(b * t, c, f)

        y_gca = self.gca(psd_e, de_e, plv_e, wpli_e).reshape(b, t, c, f)
        y_gass = self.gass(y_gca)
        z = y_gass.mean(dim=1)
        return self.head(z)
