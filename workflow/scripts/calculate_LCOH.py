"""Calculate heat at the shallowest required temperature, once per grid cell."""

import json
import sys
from pathlib import Path

import _schemas
import numpy as np
import xarray as xr
from _economics import drilling_cost_meur_per_well, resolve_drilling_curve_definition
from _eligibility import load_fraction
from _heat import write_raster
from affine import Affine

LCOH_COMPONENT_NAMES = ("drilling", "stimulation", "pump", "thermal_plant", "fixed_om")


def shallowest_depth(temperature, depths, target, max_depth):
    """Find exact matches or adjacent upward crossings without extrapolation.

    The final temperature axis is depth. A bracket may end beyond max_depth,
    provided its interpolated crossing does not. Missing layers never form a
    bracket, and non-monotonic profiles use their shallowest valid crossing.
    """
    depths = np.asarray(depths, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    result = np.full(temperature.shape[:-1], np.nan)
    for index, depth in enumerate(depths):
        layer = temperature[..., index]
        exact = np.isfinite(layer) & (layer == target) & np.isnan(result)
        if 0 < depth <= max_depth:
            result[exact] = depth
        if index == 0:
            continue
        previous = temperature[..., index - 1]
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
    scenario = model
    include_egs = bool(scenario["include_egs"])
    curve_spec = resolve_drilling_curve_definition(
        scenario["drilling_cost_curve"], drilling_csv
    )
    # Keep grid labels throughout the calculation, including NumPy callers.
    T = xr.DataArray(temperature, dims=("depth_m", "y", "x")).assign_coords(
        depth_m=depths
    )
    sed2d = xr.DataArray(sediment, dims=("y", "x"), coords={"y": T.y, "x": T.x})
    use_dask = "parallelized" if hasattr(T.data, "chunks") else False
    depth = xr.apply_ufunc(
        shallowest_depth,
        T,
        T.depth_m,
        input_core_dims=[["depth_m"], ["depth_m"]],
        output_core_dims=[[]],
        kwargs={
            "target": scenario["heat_production_temperature"],
            "max_depth": scenario["max_depth"],
        },
        dask=use_dask,
        output_dtypes=[float],
    )

    # Missing sediment implies EGS at every positive depth.
    sed2d_ok = xr.where(np.isfinite(sed2d), sed2d, 0.0).astype(np.float32)
    cutoff2d = np.minimum(scenario["EGS_cutoff"], sed2d_ok)
    is_egs = depth > cutoff2d
    fraction = xr.ones_like(depth) * eligibility_fraction
    area = 100 * fraction
    n_pairs = area.astype(float) / (2 * scenario["well_spacing"] ** 2)
    flowrate = xr.where(
        is_egs,
        scenario["flowrate_egs"] * n_pairs,
        scenario["flowrate_conventional"] * n_pairs,
    )
    dT = (
        scenario["heat_production_temperature"]
        - scenario["heat_reinjection_temperature"]
    )
    power = flowrate * 4250.0 * dT / 1e6
    annual_heat = power * (scenario["heat_capacity_factor"] * 24.0 * 365.0)
    valid_power = np.isfinite(depth) & (fraction > 0)
    if not include_egs:
        valid_power = valid_power & ~is_egs

    well_cost = xr.apply_ufunc(
        drilling_cost_meur_per_well,
        depth.astype(float),
        kwargs={"spec": curve_spec},
        dask=use_dask,
        output_dtypes=[float],
    ) * (n_pairs * 2)
    investments = (
        well_cost,
        xr.where(is_egs, scenario["stimulation_cost_per_doublet"] * n_pairs, 0.0),
        scenario["pump_cost"] * n_pairs,
        (power * 1000.0 * scenario["thermal_plant_cost_per_kw"]) / 1e6,
    )
    capex_total = sum(investments)
    lifetime = float(scenario["lifetime"])
    rate = float(scenario["discount_rate"])
    annuity = (
        1 / lifetime
        if rate == 0
        else rate * (1 + rate) ** lifetime / ((1 + rate) ** lifetime - 1)
    )
    conversion = 1e6 / annual_heat.where(annual_heat > 0.0)
    component_arrays = [cost * annuity * conversion for cost in investments]
    component_arrays.append(capex_total * scenario["fixed_OM_rate"] * conversion)
    lcoh = sum(component_arrays)
    valid_lcoh = np.isfinite(annual_heat) & (annual_heat > 0.0) & np.isfinite(lcoh)
    specific_capex = capex_total * 1e3 / power.where(power > 0.0)
    lcoh_components = dict(zip(LCOH_COMPONENT_NAMES, component_arrays))
    values = {
        "depth": depth,
        "capacity": power,
        "potential": annual_heat,
        "lcoh": lcoh,
        "specific_capex": specific_capex,
        **{f"lcoh_{name}": values for name, values in lcoh_components.items()},
    }
    # Heat maps share one validity mask, including depth and installed capacity.
    return {
        name: value.where(valid_power & valid_lcoh).transpose("y", "x")
        for name, value in values.items()
    }


def main():
    """Read the prepared source window and write the declared heat maps."""
    window = json.loads(Path(snakemake.input.extent).read_text())
    with xr.open_dataset(snakemake.input.nc) as source:
        _schemas.validate_temperature(source)
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
        sediment = (
            source.sediment_thickness_m.sel(x=temperature.x, y=temperature.y)
            .transpose("y", "x")
            .load()
        )
    model = dict(snakemake.params.items())
    model.update(model.pop("scenario"))
    fraction = load_fraction(snakemake.input.eligibility, temperature.x, temperature.y)
    values = calculate_heat(
        temperature,
        temperature.depth_m,
        sediment,
        model,
        fraction,
        drilling_csv=snakemake.input.drilling_csv[0]
        if snakemake.input.drilling_csv
        else None,
    )
    transform = Affine(
        dx, 0, float(temperature.x[0]) - dx / 2, 0, dy, float(temperature.y[0]) - dy / 2
    )
    for name, unit in [
        ("lcoh", "EUR2020/MWh_th"),
        ("specific_capex", "EUR2020/kW_th"),
        ("potential", "MWh_th/year"),
        ("depth", "m"),
    ]:
        data = values[name].rename(None)
        data.attrs = {}
        data = data.rio.write_crs(crs).rio.write_transform(transform)
        if window["empty"]:
            data[:] = np.nan
        write_raster(data, snakemake.output[name], unit)


if __name__ == "__main__":
    sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)
    main()
