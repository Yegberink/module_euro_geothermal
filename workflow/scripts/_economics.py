"""Shared drilling economics for heat and power."""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# BLS CPI-U, US city average, all items, annual averages, not seasonally adjusted.
# https://www.bls.gov/cpi/tables/supplemental-files/historical-cpi-u-202402.pdf
# https://www.bls.gov/regions/southwest/data/xg-tables/ro6xg01.htm
# Source price years are model assumptions; see INTERFACE.yaml.
CPI_US = {2017: 245.120, 2020: 258.811, 2021: 270.970, 2025: 321.943}
# Bundesbank/ECB 2020 annual average: 1 EUR = 1.1422 USD (not EUR per USD).
# https://www.bundesbank.de/resource/blob/895836/a5c692d5571fbc2729ebd2a735ebbce7/mL/2022-08-15-11-28-07-wechselkursstatistik-data.pdf
EUR_PER_USD_2020 = 1 / 1.1422

DRILLING_CURVES = {
    # ThermoGIS coefficients adopted as EUR2020 in the manuscript.
    # https://www.thermogis.nl/economisch-model
    "thermogis": {
        "curve_type": "quadratic_eur2020",
        "constant": 250000.0,
        "linear": 700.0,
        "quadratic": 0.2,
    },
    "shamoushaki_2021": {
        "curve_type": "log_quadratic_usd",
        "a": 554800.0,
        "b": 0.2171,
        "c": 813500.0,
        "currency_year": 2021,
    },
    "getem_2017": {"curve_type": "csv_fit_usd", "currency_year": 2017},
    "getem_2025": {"curve_type": "csv_fit_usd", "currency_year": 2025},
}


def _log_quadratic_cost_musd(depth_m, a, b, c):
    depth_m = np.asarray(depth_m, dtype=float)
    return (a * np.log(depth_m) + b * depth_m**2 + c) / 1e6


def resolve_drilling_curve_definition(curve_name, csv_path=None):
    """Resolve coefficients, fitting depth/cost data for CSV-based USD curves."""
    spec = dict(DRILLING_CURVES[curve_name])
    if spec["curve_type"] != "csv_fit_usd":
        return spec
    # Headerless depth (m), cost (million source-year USD per well) data.
    df = pd.read_csv(csv_path, header=None)

    depth_m = df.iloc[:, 0].to_numpy(dtype=float)
    costs = df.iloc[:, 1].to_numpy(dtype=float)

    p0 = (1e5, max(np.polyfit(depth_m, costs, 2)[0], 1e-12), 5e5)
    coeffs, _ = curve_fit(_log_quadratic_cost_musd, depth_m, costs, p0=p0, maxfev=10000)
    return {
        "curve_type": "log_quadratic_usd",
        "a": float(coeffs[0]),
        "b": float(coeffs[1]),
        "c": float(coeffs[2]),
        "currency_year": int(spec["currency_year"]),
    }


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
        # Adjust source-year USD to 2020 USD, then convert to EUR2020.
        return cost_musd * (
            (CPI_US[2020] / CPI_US[int(spec["currency_year"])]) * EUR_PER_USD_2020
        )
