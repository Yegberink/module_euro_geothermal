"""Calculate heat at the shallowest required temperature, once per grid cell."""

import json
import sys
from pathlib import Path

import _schemas
import numpy as np
import xarray as xr
from _economics import drilling_cost_meur_per_well, resolve_drilling_curve_definition
from _heat import write_raster


def shallowest_depth(temperature, depths, target, max_depth):
    """Find exact matches or adjacent upward crossings without extrapolation.

    Temperature has shape (depth, y, x). A bracket may end beyond max_depth,
    provided its interpolated crossing does not. Missing layers never form a
    bracket, and non-monotonic profiles use their shallowest valid crossing.
    """
    depths = np.asarray(depths, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
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


def calculate_heat(
    temperature, depths, sediment, model, eligibility_fraction=1.0, drilling_csv=None
):
    """Return depth, thermal capacity, annual heat and additive LCOH components.

    Like the electricity model, one source cell represents 100 km² and permits
    fractional doublets. Partial development therefore scales output and costs
    proportionally. No electricity-conversion efficiency or parasitic fraction
    is applied to useful thermal output.
    """
    curve_spec = resolve_drilling_curve_definition(
        model["drilling_cost_curve"], drilling_csv
    )
    depth = shallowest_depth(
        temperature, depths, model["heat_production_temperature"], model["max_depth"]
    )
    # Missing sediment implies EGS at every positive depth.
    sed_ok = np.where(np.isfinite(sediment), sediment, 0.0).astype(np.float32)
    cutoff = np.minimum(model["EGS_cutoff"], sed_ok)
    is_egs = depth > cutoff
    valid = np.isfinite(depth) & (model["include_egs"] | ~is_egs)

    # `well_spacing` and `area_available` are both expressed in km / km^2 here.
    doublet_occupation = 2 * model["well_spacing"] ** 2  # km^2 per doublet
    fraction = np.asarray(eligibility_fraction, dtype=float)
    valid &= fraction > 0
    area_available = 100 * fraction
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

    well_cost_no_stim = drilling_cost_meur_per_well(depth, curve_spec) * nr_wells
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

    discount_rate = float(model["discount_rate"])
    TT = float(model["lifetime"])
    annuity_factor = (
        1 / TT
        if discount_rate == 0
        else (discount_rate * (1 + discount_rate) ** TT)
        / ((1 + discount_rate) ** TT - 1)
    )
    meur_to_eur_per_mwh = np.divide(
        1e6, annual_heat, out=np.full_like(annual_heat, np.nan), where=annual_heat > 0
    )
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
        _schemas.TemperatureDepthSchema.validate(source.depth_m.values)
        crs = source.attrs["crs"]
        temperature = (
            source.temperature.isel(x=slice(*window["x"]), y=slice(*window["y"]))
            .transpose("depth_m", "y", "x")
            .load()
        )
        # Preserve source resolution even for a one-cell crop.
        dx = float(source.x[1] - source.x[0])
        dy = float(source.y[1] - source.y[0])
    with xr.open_dataset(snakemake.input.sed) as source:
        if source.attrs["crs"] != crs:
            raise ValueError("Sediment-grid CRS does not match temperature-grid CRS.")
        sediment = source.sediment_thickness_m.sel(
            x=temperature.x, y=temperature.y
        ).values
    model = dict(snakemake.params.items())
    from _eligibility import load_fraction

    fraction = load_fraction(
        snakemake.input.eligibility, temperature.x, temperature.y, crs
    )
    values = calculate_heat(
        temperature.values,
        temperature.depth_m.values,
        sediment,
        model,
        fraction.values,
        drilling_csv=snakemake.input.drilling_csv[0]
        if snakemake.input.drilling_csv
        else None,
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
