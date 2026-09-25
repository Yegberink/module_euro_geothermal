"""Calculate heat at the shallowest required temperature, once per grid cell.

The drilling-cost functions and sediment policy below are intentionally copied
from calculate_LCOE.py, with their annotations retained. Heat economics remain
independent of power-cycle efficiency and power-plant costs.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from _heat import write_raster
from scipy.optimize import curve_fit

CPI_US = {2017: 245.120, 2020: 258.811, 2021: 270.970}
CPI_2025_PROXY = 324.368
EUR_PER_USD_2020 = 1 / 1.1422


def _usd_to_eur2020_factor(currency_year):
    if int(currency_year) == 2025:
        return (CPI_US[2020] / CPI_2025_PROXY) * EUR_PER_USD_2020
    return (CPI_US[2020] / CPI_US[int(currency_year)]) * EUR_PER_USD_2020


def _log_quadratic_cost_meur(depth_m, a, b, c):
    depth_m = np.asarray(depth_m, dtype=float)
    return (a * np.log(depth_m) + b * depth_m**2 + c) / 1e6


def _init_log_quadratic_guesses(depth_m, costs):
    p2 = np.polyfit(depth_m, costs, 2)
    b0 = max(p2[0], 1e-12)
    return 1e5, b0, 5e5


def _load_curve_from_csv(csv_path, currency_year):
    csv_file = Path(csv_path)
    df = pd.read_csv(csv_file, header=None)

    depth_m = df.iloc[:, 0].to_numpy(dtype=float)
    costs = df.iloc[:, 1].to_numpy(dtype=float)

    p0 = _init_log_quadratic_guesses(depth_m, costs)
    coeffs, _ = curve_fit(_log_quadratic_cost_meur, depth_m, costs, p0=p0, maxfev=10000)
    return {
        "curve_type": "log_quadratic_usd",
        "a": float(coeffs[0]),
        "b": float(coeffs[1]),
        "c": float(coeffs[2]),
        "currency_year": int(currency_year),
    }


def _resolve_drilling_curve_definition(curve_name, definitions):
    builtins = {
        "thermogis": {
            "curve_type": "quadratic_eur2020",
            "constant": 250000.0,
            "linear": 700.0,
            "quadratic": 0.2,
        },
        "shamoushaki_2021": {
            "curve_type": "log_quadratic_usd",
            "a": 5.548e5,
            "b": 0.2171,
            "c": 8.135e5,
            "currency_year": 2021,
        },
    }

    spec = builtins.get(curve_name)
    if curve_name in definitions:
        custom_spec = dict(definitions[curve_name])
        if custom_spec.get("curve_type") == "csv_fit_usd":
            spec = _load_curve_from_csv(
                csv_path=custom_spec["csv_path"],
                currency_year=custom_spec["currency_year"],
            )
        else:
            spec = custom_spec

    return spec


DRILLING_COST_CURVE_SPEC = None


def _drilling_cost_meur_per_well(depth_m):
    spec = DRILLING_COST_CURVE_SPEC
    curve_type = spec["curve_type"]

    if curve_type == "quadratic_eur2020":
        return (
            float(spec["constant"])
            + float(spec["linear"]) * depth_m
            + float(spec["quadratic"]) * depth_m**2
        ) / 1e6

    if curve_type == "log_quadratic_usd":
        cost_meur = _log_quadratic_cost_meur(
            depth_m, float(spec["a"]), float(spec["b"]), float(spec["c"])
        )
        return cost_meur * _usd_to_eur2020_factor(spec["currency_year"])


def _apply_sed_nan_policy(sed_thickness, z_thr=3000.0, policy="always_egs"):
    """Replace NaNs in sediment thickness according to policy.

    - depth_only: NaN -> z_thr (EGS only deeper than z_thr)
    - always_egs: NaN -> 0 (EGS at all depths).
    """
    if policy == "depth_only":
        fill = float(z_thr)
    elif policy == "always_egs":
        fill = 0.0
    return np.where(np.isfinite(sed_thickness), sed_thickness, fill).astype(np.float32)


def shallowest_depth(temperature, depths, target, max_depth):
    """Find exact matches or adjacent upward crossings without extrapolation.

    Temperature has shape (depth, y, x). A bracket may end beyond max_depth,
    provided its interpolated crossing does not. Missing layers never form a
    bracket, and non-monotonic profiles use their shallowest valid crossing.
    """
    depths = np.asarray(depths, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    if not np.isfinite(depths).all() or np.any(np.diff(depths) <= 0):
        raise ValueError("Temperature depths must be finite and strictly increasing.")
    result = np.full(temperature.shape[1:], np.nan)
    for index, depth in enumerate(depths):
        layer = temperature[index]
        exact = np.isfinite(layer) & (layer == target) & np.isnan(result)
        if 0 < depth <= max_depth:
            result[exact] = depth
        if index == 0:
            continue
        previous = temperature[index - 1]
        crossing = (
            np.isnan(result)
            & np.isfinite(previous)
            & np.isfinite(layer)
            & (previous < target)
            & (layer > target)
        )
        candidate = np.full_like(result, np.nan)
        np.divide(target - previous, layer - previous, out=candidate, where=crossing)
        candidate = depths[index - 1] + candidate * (depth - depths[index - 1])
        valid = crossing & (candidate > 0) & (candidate <= max_depth)
        result[valid] = candidate[valid]
        if depth >= max_depth:
            break
    return result


def calculate_heat(temperature, depths, sediment, model):
    """Return depth, thermal capacity, annual heat and additive LCOH components.

    Like the electricity model, one source cell represents 100 km² and permits
    fractional doublets. Partial development therefore scales output and costs
    proportionally. No electricity-conversion efficiency or parasitic fraction
    is applied to useful thermal output.
    """
    global DRILLING_COST_CURVE_SPEC
    DRILLING_COST_CURVE_SPEC = _resolve_drilling_curve_definition(
        model["drilling_cost_curve"], model["drilling_cost_curve_definitions"]
    )
    depth = shallowest_depth(
        temperature, depths, model["heat_production_temperature"], model["max_depth"]
    )
    sed_ok = _apply_sed_nan_policy(
        sediment, z_thr=model["EGS_cutoff"], policy=model["SED_NAN_POLICY"]
    )
    cutoff = np.minimum(model["EGS_cutoff"], sed_ok)
    is_egs = depth > cutoff
    valid = np.isfinite(depth) & (model["include_egs"] | ~is_egs)

    # `well_spacing` and `area_available` are both expressed in km / km^2 here.
    doublet_occupation = 2 * model["well_spacing"] ** 2  # km^2 per doublet
    area_available = 100
    n_pairs = area_available / doublet_occupation
    nr_wells = n_pairs * 2
    flowrate_wells = np.where(
        is_egs,
        model["flowrate_egs"] * n_pairs,
        model["flowrate_conventional"] * n_pairs,
    )
    Cp_water = 4250.0
    dT = model["heat_production_temperature"] - model["heat_reinjection_temperature"]
    therm_out = flowrate_wells * Cp_water * dT / 1e6
    load_hours = model["heat_capacity_factor"] * 24.0 * 365.0
    annual_heat = therm_out * load_hours

    well_cost_no_stim = _drilling_cost_meur_per_well(depth) * nr_wells
    stimulation_cost = model["stimulation_cost_per_doublet"] * n_pairs
    stimulation_investment = np.where(is_egs, stimulation_cost, 0.0)
    pump_investment = model["pump_cost"] * n_pairs
    thermalplant_investment = (
        therm_out * 1000 * model["thermal_plant_cost_per_kw"] / 1e6
    )
    capex_total = (
        well_cost_no_stim
        + stimulation_investment
        + pump_investment
        + thermalplant_investment
    )
    fixed_OM_costs = capex_total * model["fixed_OM_rate"]

    discount_rate = 0.075
    TT = float(model["lifetime"])
    annuity_factor = (discount_rate * (1 + discount_rate) ** TT) / (
        (1 + discount_rate) ** TT - 1
    )
    meur_to_eur_per_mwh = 1e6 / annual_heat
    components = {
        "lcoh_drilling": well_cost_no_stim * annuity_factor * meur_to_eur_per_mwh,
        "lcoh_stimulation": stimulation_investment
        * annuity_factor
        * meur_to_eur_per_mwh,
        "lcoh_pump": pump_investment * annuity_factor * meur_to_eur_per_mwh,
        "lcoh_thermal_plant": thermalplant_investment
        * annuity_factor
        * meur_to_eur_per_mwh,
        "lcoh_fixed_om": fixed_OM_costs * meur_to_eur_per_mwh,
    }
    lcoh = sum(components.values())
    valid &= np.isfinite(lcoh) & np.isfinite(annual_heat) & (annual_heat > 0)
    values = {
        "depth": depth,
        "capacity": therm_out,
        "potential": annual_heat,
        "lcoh": lcoh,
        **components,
    }
    return {name: np.where(valid, value, np.nan) for name, value in values.items()}


def main():
    """Read the prepared source window and write the three declared heat maps."""
    window = json.loads(Path(snakemake.input.extent).read_text())
    with xr.open_dataset(snakemake.input.nc) as source:
        crs = source.attrs.get("crs", "EPSG:3857")
        temperature = (
            source.temperature.isel(x=slice(*window["x"]), y=slice(*window["y"]))
            .transpose("depth_m", "y", "x")
            .load()
        )
        # Preserve source resolution even for a one-cell crop.
        dx = float(source.x[1] - source.x[0])
        dy = float(source.y[1] - source.y[0])
    with xr.open_dataset(snakemake.input.sed) as source:
        if source.attrs.get("crs", crs) != crs:
            raise ValueError("Sediment-grid CRS does not match temperature-grid CRS.")
        sediment = source.sediment_thickness_m.sel(
            x=temperature.x, y=temperature.y
        ).values
    model = dict(snakemake.params.model)
    definitions = {
        name: dict(spec)
        for name, spec in model["drilling_cost_curve_definitions"].items()
    }
    # Resolve module-relative baseline curves to Snakemake's declared inputs.
    for year in [2017, 2025]:
        if f"getem_{year}" in definitions:
            definitions[f"getem_{year}"]["csv_path"] = snakemake.input[
                f"drilling_costs_{year}"
            ]
    model["drilling_cost_curve_definitions"] = definitions
    values = calculate_heat(
        temperature.values, temperature.depth_m.values, sediment, model
    )
    from affine import Affine

    transform = Affine(
        dx, 0, float(temperature.x[0]) - dx / 2, 0, dy, float(temperature.y[0]) - dy / 2
    )
    for name, unit in [
        ("lcoh", "EUR2020/MWh_th"),
        ("potential", "MWh_th/year"),
        ("depth", "m"),
    ]:
        data = (
            xr.DataArray(
                values[name],
                dims=("y", "x"),
                coords={"x": temperature.x, "y": temperature.y},
            )
            .rio.write_crs(crs)
            .rio.write_transform(transform)
        )
        if window["empty"]:
            data[:] = np.nan
        write_raster(data, snakemake.output[name], unit)


if __name__ == "__main__":
    sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)
    main()
