"""Shared drilling economics and sediment assumptions for heat and power."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# BLS CPI-U, US city average, all items, annual averages, not seasonally adjusted.
# https://www.bls.gov/cpi/tables/supplemental-files/historical-cpi-u-202402.pdf
# https://www.bls.gov/regions/southwest/data/xg-tables/ro6xg01.htm
# Source price years are model assumptions; see config/README.md.
CPI_US = {2017: 245.120, 2020: 258.811, 2021: 270.970, 2025: 321.943}
# Bundesbank/ECB 2020 annual average: 1 EUR = 1.1422 USD (not EUR per USD).
# https://www.bundesbank.de/resource/blob/895836/a5c692d5571fbc2729ebd2a735ebbce7/mL/2022-08-15-11-28-07-wechselkursstatistik-data.pdf
EUR_PER_USD_2020 = 1 / 1.1422


def _usd_to_eur2020_factor(currency_year):
    """Adjust source-year USD to 2020 USD, then convert to EUR2020."""
    return (CPI_US[2020] / CPI_US[int(currency_year)]) * EUR_PER_USD_2020


def _log_quadratic_cost_musd(depth_m, a, b, c):
    depth_m = np.asarray(depth_m, dtype=float)
    return (a * np.log(depth_m) + b * depth_m**2 + c) / 1e6


def _init_log_quadratic_guesses(depth_m, costs):
    p2 = np.polyfit(depth_m, costs, 2)
    b0 = max(p2[0], 1e-12)
    return 1e5, b0, 5e5


def _load_curve_from_csv(csv_path, currency_year):
    """Fit headerless depth (m), cost (million source-year USD per well) data."""
    csv_file = Path(csv_path)
    df = pd.read_csv(csv_file, header=None)

    depth_m = df.iloc[:, 0].to_numpy(dtype=float)
    costs = df.iloc[:, 1].to_numpy(dtype=float)

    p0 = _init_log_quadratic_guesses(depth_m, costs)
    coeffs, _ = curve_fit(_log_quadratic_cost_musd, depth_m, costs, p0=p0, maxfev=10000)
    return {
        "curve_type": "log_quadratic_usd",
        "a": float(coeffs[0]),
        "b": float(coeffs[1]),
        "c": float(coeffs[2]),
        "currency_year": int(currency_year),
    }


def resolve_drilling_curve_definition(curve_name, definitions):
    spec = dict(definitions[curve_name])
    if spec["curve_type"] == "csv_fit_usd":
        return _load_curve_from_csv(spec["csv_path"], spec["currency_year"])
    return spec


def drilling_cost_meur_per_well(depth_m, spec):
    curve_type = spec["curve_type"]

    if curve_type == "quadratic_eur2020":
        return (
            float(spec["constant"])
            + float(spec["linear"]) * depth_m
            + float(spec["quadratic"]) * depth_m**2
        ) / 1e6

    if curve_type == "log_quadratic_usd":
        cost_musd = _log_quadratic_cost_musd(
            depth_m, float(spec["a"]), float(spec["b"]), float(spec["c"])
        )
        return cost_musd * _usd_to_eur2020_factor(spec["currency_year"])


def apply_sed_nan_policy(sed_thickness, z_thr, policy):
    """Replace NaNs in sediment thickness according to policy.

    - depth_only: NaN -> z_thr (EGS only deeper than z_thr)
    - always_egs: NaN -> 0 (EGS at all depths).
    """
    if policy == "depth_only":
        fill = float(z_thr)
    elif policy == "always_egs":
        fill = 0.0
    return np.where(np.isfinite(sed_thickness), sed_thickness, fill).astype(np.float32)
