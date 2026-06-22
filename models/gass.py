# -*- coding: utf-8 -*-
"""Gated Axial State-Space Module (GASS) for BiMamba-ESCN."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualGate(nn.Module):

    def __init__(self, chans: int = 10, bands: int = 5, reduction: int = 4):
        super().__init__()
        c_hidden = max(1, chans // reduction)
        f_hidden = max(1, bands // reduction)

        self.mlp_c = nn.Sequential(
            nn.Linear(chans, c_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(c_hidden, chans),
        )
        self.mlp_f = nn.Sequential(
            nn.Linear(bands, f_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(f_hidden, bands),
        )
        self.norm_f = nn.LayerNorm(bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B_eff, C, F]
        avg_c = x.mean(dim=2)
        max_c = x.max(dim=2).values
        s_c = torch.sigmoid(self.mlp_c(avg_c) + self.mlp_c(max_c)).unsqueeze(-1)

        y_tilde = x * s_c

        avg_f = y_tilde.mean(dim=1)
        max_f = y_tilde.max(dim=1).values
        s_f = torch.sigmoid(self.mlp_f(avg_f) + self.mlp_f(max_f)).unsqueeze(1)

        y_hat = y_tilde * s_f
        return self.norm_f(y_hat + x)


class BidirectionalSSMBlock(nn.Module):

    def __init__(self, bands: int = 5, d_model: int = 64, d_state: int = 32, fusion: str = "mean"):
        super().__init__()
        if fusion not in {"mean", "sum", "concat"}:
            raise ValueError("fusion must be one of {'mean', 'sum', 'concat'}")

        self.bands = bands
        self.d_model = d_model
        self.d_state = d_state
        self.fusion = fusion

        self.input_embed = nn.Linear(bands, d_model)
        self.local_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            groups=d_model,
            bias=True,
        )
        self.in_proj = nn.Linear(d_model, d_model * 2)

        self.x_proj = nn.Linear(d_model, d_model + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)

        A = torch.arange(1, d_state + 1).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A.float()))
        self.D = nn.Parameter(torch.ones(d_model))

        fused_dim = d_model if fusion != "concat" else d_model * 2
        self.gate_proj = nn.Linear(fused_dim, fused_dim, bias=True)
        self.res_proj = nn.Identity() if fusion != "concat" else nn.Linear(d_model, fused_dim)
        self.norm_f = nn.LayerNorm(fused_dim)
        self.out_proj = nn.Linear(fused_dim, bands)

    def _ssm_scan(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        batch, length, d_dim = x.shape
        d, n = self.A_log.shape
        if d != d_dim:
            raise RuntimeError(f"A_log feature dimension {d} does not match input {d_dim}")

        A = -torch.exp(self.A_log)

        x_dbl = self.x_proj(x)
        delta, b_term, c_term = x_dbl.split([d_dim, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        delta_a = torch.exp(delta.unsqueeze(-1) * A)
        delta_b = delta.unsqueeze(-1) * b_term.unsqueeze(2)
        bx = delta_b * x.unsqueeze(-1)

        h = torch.zeros(batch, d_dim, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(length):
            h = delta_a[:, t] * h + bx[:, t]
            ys.append((h @ c_term[:, t].unsqueeze(-1)).squeeze(-1))
        y = torch.stack(ys, dim=1)
        y = y + self.D * x
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B_axis, L, F]
        x_latent = self.input_embed(x)
        x_enh = self.local_conv(x_latent.transpose(1, 2)).transpose(1, 2)
        x_enh = F.silu(x_enh)

        x_and_res = self.in_proj(x_enh)
        x_core, x_res = x_and_res.chunk(2, dim=-1)

        y_fwd = self._ssm_scan(x_core)
        y_bwd = self._ssm_scan(torch.flip(x_core, dims=[1]))
        y_bwd = torch.flip(y_bwd, dims=[1])

        if self.fusion == "sum":
            z = y_fwd + y_bwd
        elif self.fusion == "concat":
            z = torch.cat([y_fwd, y_bwd], dim=-1)
        else:
            z = 0.5 * (y_fwd + y_bwd)

        gate = torch.sigmoid(self.gate_proj(z))
        res = self.res_proj(x_res)
        out_latent = self.norm_f(gate * z + F.silu(res))
        return self.out_proj(out_latent)


class GASS(nn.Module):

    def __init__(
        self,
        chans: int = 10,
        bands: int = 5,
        d_model: int = 64,
        d_state: int = 32,
        reduction: int = 4,
        fusion: str = "mean",
    ):
        super().__init__()
        self.chans = chans
        self.bands = bands
        self.d_model = d_model
        self.dualgate = DualGate(chans=chans, bands=bands, reduction=reduction)
        self.channel_ssm = BidirectionalSSMBlock(bands=bands, d_model=d_model, d_state=d_state, fusion=fusion)
        self.temporal_ssm = BidirectionalSSMBlock(bands=bands, d_model=d_model, d_state=d_state, fusion=fusion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C,F]
        b, t, c, f = x.shape
        if c != self.chans or f != self.bands:
            raise RuntimeError(f"Expected [B,T,{self.chans},{self.bands}], got {tuple(x.shape)}")

        x_eff = x.reshape(b * t, c, f)
        u0 = self.dualgate(x_eff)
        z_tilde = self.channel_ssm(u0)

        z_seq = z_tilde.reshape(b, t, c, f)
        x_temp = z_seq.permute(0, 2, 1, 3).reshape(b * c, t, f)
        y_temp = self.temporal_ssm(x_temp)
        return y_temp.reshape(b, c, t, f).permute(0, 2, 1, 3).contiguous()
