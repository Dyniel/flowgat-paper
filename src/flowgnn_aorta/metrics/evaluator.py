# -*- coding: utf-8 -*-
"""
Full evaluation pass: accumulates PP@δ on HE, AUC over tolerances, wall/core
split, clinical diagnostics (peak location, peak magnitude, ΔP, WSS), and
per-graph statistics (p50/p90 of PP and of various errors).

Returns a flat dict of scalars suitable for logging to W&B or CSV.
DDP aggregation is handled by the training engine, not here.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import torch
from torch import Tensor

from .clinical import (
    he_mask_from_speed,
    relative_vector_error,
    speed_relative_error,
    angular_error_rad,
    pp_at_tol,
    pp_auc,
    peak_velocity_localisation,
    peak_pressure_drop_error,
    wss_error_metrics,
)


# -----------------------------------------------------------------------------
#  A running accumulator that is cheap to update per graph and produces the
#  final dict at the end of the epoch.
# -----------------------------------------------------------------------------

class MetricAccumulator:
    def __init__(
        self,
        strict_tols: List[float],
        rel_thresholds: List[float],
        he_percentile: float,
        auc_range: Tuple[float, float],
        rel_tol_primary: float = 0.10,
        compute_wss: bool = True,
        mu_blood: float = 3.5e-3,      # Pa·s  (≈ 1060·3.3e-6)
    ):
        self.strict_tols = [float(t) for t in strict_tols]
        self.rel_thresholds = [float(t) for t in rel_thresholds]
        self.he_percentile = float(he_percentile)
        self.auc_range = (float(auc_range[0]), float(auc_range[1]))
        self.rel_tol_primary = float(rel_tol_primary)
        self.compute_wss = bool(compute_wss)
        self.mu_blood = float(mu_blood)

        # aggregate tensors (device-less, we return floats)
        self._sum_he = 0.0
        self._sum_ok_he = {t: 0.0 for t in self.strict_tols}
        self._sum_le_thr = {t: 0.0 for t in self.rel_thresholds}
        self._sum_err2_he = 0.0
        self._sum_rel_he = 0.0
        self._sum_sp_he = 0.0
        self._sum_ang_he = 0.0

        self._sum_he_wall = 0.0
        self._sum_he_core = 0.0
        self._sum_ok_he_wall = 0.0
        self._sum_ok_he_core = 0.0

        # per-graph lists (for quantiles)
        self._pp_frames: List[float] = []
        self._peak_loc_mm: List[float] = []
        self._peak_mag_rel_abs: List[float] = []
        self._dP_abs_err: List[float] = []

        # WSS (per wall node pooled)
        self._wss_mae: List[float] = []
        self._wss_r2: List[float] = []
        self._wss_bias: List[float] = []

    # ----------------------------------------------------------
    def update(
        self,
        pred: Tensor,
        y: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        wall_mask: Optional[Tensor] = None,
        core_mask: Optional[Tensor] = None,
        wall_normals: Optional[Tensor] = None,
    ) -> None:
        """Update with a single graph's tensors (detached, eval-mode)."""
        with torch.no_grad():
            he = he_mask_from_speed(y, self.he_percentile)          # [N] bool
            if he.sum() == 0:
                return

            rel = relative_vector_error(pred, y)                     # [N]
            sp  = speed_relative_error(pred, y)                      # [N]
            ang = angular_error_rad(pred, y)                         # [N]
            err2 = ((pred - y) ** 2).sum(dim=-1)                     # [N]

            n_he = float(he.sum().item())
            self._sum_he += n_he

            for t in self.strict_tols:
                self._sum_ok_he[t] += float(((rel <= t) & he).sum().item())
            for t in self.rel_thresholds:
                self._sum_le_thr[t] += float(((rel <= t) & he).sum().item())

            self._sum_err2_he += float((err2[he]).sum().item())
            self._sum_rel_he  += float((rel[he]).sum().item())
            self._sum_sp_he   += float((sp[he]).sum().item())
            self._sum_ang_he  += float((ang[he]).sum().item())

            # Wall / core split on HE
            if wall_mask is not None:
                if wall_mask.dtype != torch.bool:
                    wall_mask = wall_mask > 0.5
                he_wall = he & wall_mask
                he_core = he & (~wall_mask) if core_mask is None else he & core_mask.bool()
                self._sum_he_wall += float(he_wall.sum().item())
                self._sum_he_core += float(he_core.sum().item())
                if he_wall.any():
                    self._sum_ok_he_wall += float(((rel <= self.rel_tol_primary) & he_wall).sum().item())
                if he_core.any():
                    self._sum_ok_he_core += float(((rel <= self.rel_tol_primary) & he_core).sum().item())

            # Per-graph PP@primary for distribution stats
            pp_frame = float(((rel <= self.rel_tol_primary) & he).sum().item() / max(n_he, 1.0))
            self._pp_frames.append(pp_frame)

            # Clinical: peak location & ΔP
            peak = peak_velocity_localisation(pred, y, pos, topk=1)
            if not math.isnan(peak["peak_loc_mm"]):
                self._peak_loc_mm.append(peak["peak_loc_mm"])
                self._peak_mag_rel_abs.append(peak["peak_mag_rel_abs"])
            dP = peak_pressure_drop_error(pred, y)
            if not math.isnan(dP["dP_abs_err_mmHg"]):
                self._dP_abs_err.append(dP["dP_abs_err_mmHg"])

            # WSS (if we have wall mask + normals)
            if (
                self.compute_wss
                and wall_mask is not None
                and wall_normals is not None
                and wall_mask.sum() > 0
            ):
                wss = wss_error_metrics(
                    pred, y, pos, edge_index,
                    wall_mask, wall_normals, mu=self.mu_blood,
                )
                if not math.isnan(wss["wss_mae_Pa"]):
                    self._wss_mae.append(wss["wss_mae_Pa"])
                    self._wss_r2.append(wss["wss_r2"])
                    self._wss_bias.append(wss["wss_bias_Pa"])

    # ----------------------------------------------------------
    def finalize(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        denom = max(self._sum_he, 1.0)

        # PP@δ on HE — primary + strict
        for t in self.strict_tols:
            out[f"val/pp_at_{t:.3f}"] = (self._sum_ok_he[t] / denom) if self._sum_he > 0 else 0.0
        out["val/pp_at_rel_0.100"] = out[f"val/pp_at_{self.rel_tol_primary:.3f}"]  # legacy alias

        # AUC
        pp_by_tol: Dict[float, Tensor] = {
            t: torch.tensor(out[f"val/pp_at_{t:.3f}"]) for t in self.strict_tols
        }
        auc = pp_auc(pp_by_tol, self.auc_range[0], self.auc_range[1])
        out[f"val/pp_auc_{self.auc_range[0]:.3f}_{self.auc_range[1]:.3f}"] = float(auc.item())

        # Diagnostic fractions
        for thr in self.rel_thresholds:
            out[f"diag/rel_he_frac_le_{thr:.3f}"] = (
                (self._sum_le_thr[thr] / denom) if self._sum_he > 0 else 0.0
            )

        # HE means (vector error, speed err, angle err)
        out["val/ewrmse_he"] = (
            math.sqrt(max(self._sum_err2_he / denom, 0.0))
            if self._sum_he > 0 else float("nan")
        )
        out["val/rel_err_he_mean"] = (
            (self._sum_rel_he / denom) if self._sum_he > 0 else float("nan")
        )
        out["val/speed_err_he_mean"] = (
            (self._sum_sp_he / denom) if self._sum_he > 0 else float("nan")
        )
        out["val/angle_err_he_mean_deg"] = (
            (self._sum_ang_he / denom) * 180.0 / math.pi
            if self._sum_he > 0 else float("nan")
        )

        # Wall/core split
        out["val/pp_wall_0.100"] = (
            (self._sum_ok_he_wall / self._sum_he_wall)
            if self._sum_he_wall > 0 else float("nan")
        )
        out["val/pp_core_0.100"] = (
            (self._sum_ok_he_core / self._sum_he_core)
            if self._sum_he_core > 0 else float("nan")
        )

        # Per-graph distribution of PP
        if self._pp_frames:
            arr = np.asarray(self._pp_frames, dtype=np.float32)
            out["diag/pp_frame_mean"] = float(arr.mean())
            out["diag/pp_frame_p10"]  = float(np.quantile(arr, 0.10))
            out["diag/pp_frame_p50"]  = float(np.quantile(arr, 0.50))
            out["diag/pp_frame_p90"]  = float(np.quantile(arr, 0.90))

        # Clinical headline metrics
        if self._peak_loc_mm:
            arr = np.asarray(self._peak_loc_mm, dtype=np.float32)
            out["clin/peak_loc_mm_mean"]   = float(arr.mean())
            out["clin/peak_loc_mm_median"] = float(np.median(arr))
            out["clin/peak_loc_mm_p90"]    = float(np.quantile(arr, 0.90))
        if self._peak_mag_rel_abs:
            arr = np.asarray(self._peak_mag_rel_abs, dtype=np.float32)
            out["clin/peak_mag_rel_mean"]  = float(arr.mean())
            out["clin/peak_mag_rel_p90"]   = float(np.quantile(arr, 0.90))
        if self._dP_abs_err:
            arr = np.asarray(self._dP_abs_err, dtype=np.float32)
            out["clin/dP_mmHg_mae"]        = float(arr.mean())
            out["clin/dP_mmHg_p90"]        = float(np.quantile(arr, 0.90))

        # WSS
        if self._wss_mae:
            arr_mae = np.asarray(self._wss_mae, dtype=np.float32)
            arr_r2  = np.asarray(self._wss_r2,  dtype=np.float32)
            arr_b   = np.asarray(self._wss_bias, dtype=np.float32)
            out["clin/wss_mae_Pa_mean"] = float(arr_mae.mean())
            out["clin/wss_r2_mean"]     = float(arr_r2.mean())
            out["clin/wss_bias_Pa_mean"] = float(arr_b.mean())

        return out
