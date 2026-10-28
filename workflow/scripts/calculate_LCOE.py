"""Calculate depth-resolved geothermal potential and electricity costs."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from _economics import drilling_cost_meur_per_well, resolve_drilling_curve_definition
from _eligibility import load_fraction

if "snakemake" in dir():
    sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)

# Set some global constants from Snakemake parameters to avoid repeated lookups.
include_egs = bool(snakemake.params.include_egs)
DRILLING_COST_CURVE_SPEC = resolve_drilling_curve_definition(
    snakemake.params.drilling_cost_curve,
    snakemake.input.drilling_csv[0] if snakemake.input.drilling_csv else None,
)


def load_efficiency_curve_csv(csv_path, cycle):
    """Load the efficiency columns for a validated lowercase cycle name."""
    df = pd.read_csv(csv_path)

    t_col = f"{cycle}_temperature"
    e_col = f"{cycle}_efficiency"

    sub = df[[t_col, e_col]].dropna()

    temps = sub[t_col].astype(float).to_numpy()

    # Handle percentages like "7.0%"
    eff_raw = sub[e_col].astype(str).str.rstrip("%").astype(float)
    effs = eff_raw.to_numpy() / 100.0

    return temps, effs


LCOE_COMPONENT_NAMES = ("drilling", "stimulation", "pump", "power_plant", "fixed_om")


def main():
    """Calculate and write the LCOE cube requested by Snakemake."""
    scenario = snakemake.params.scenario
    cycle = snakemake.params.cycle
    cycle_key = {"orc": "ORC", "flash": "Flash", "kalina": "Kalina"}[cycle]

    # Explicit outputs from Snakemake rule (one tech per job)
    cube_nc = Path(snakemake.output.cube)

    ds = xr.open_dataset(snakemake.input.nc)
    crs_str = ds.attrs["crs"]

    T = ds["temperature"]  # (depth_m, y, x)
    fraction = load_fraction(snakemake.input.eligibility, T.x, T.y)
    area = 100 * fraction

    max_depth = float(scenario["max_depth"])

    # Determine effective depth limit
    effective_max = max_depth
    if not include_egs:
        effective_max = min(effective_max, scenario["EGS_cutoff"])

    # Slice depth dimension
    T = T.sel(depth_m=slice(None, effective_max))

    # Re-broadcast depth coordinate to match sliced T
    depth3d = T["depth_m"].broadcast_like(T)

    # Select the exact subset of the full, pre-aligned GST-1 grid.
    sediment_ds = xr.open_dataset(snakemake.input.sed)
    sed2d = sediment_ds.sediment_thickness_m.sel(x=T.x, y=T.y).transpose("y", "x")

    # Missing sediment implies EGS at every positive depth.
    sed2d_ok = xr.where(np.isfinite(sed2d), sed2d, 0.0).astype(np.float32)
    cutoff2d = np.minimum(scenario["EGS_cutoff"], sed2d_ok).astype(np.float32)
    cutoff2d.name = "egs_cutoff_depth_m"
    cutoff2d.attrs.update({"crs": crs_str, "units": "m"})
    cutoff3d = cutoff2d.broadcast_like(T)

    temps_curve, effs_curve = load_efficiency_curve_csv(snakemake.input.eff, cycle)

    use_dask = "parallelized" if hasattr(T.data, "chunks") else False

    eff = xr.apply_ufunc(
        np.interp,
        T,
        input_core_dims=[[]],
        kwargs={"xp": temps_curve, "fp": effs_curve, "left": np.nan, "right": np.nan},
        dask=use_dask,
        output_dtypes=[np.float32],
    ).astype(np.float32)

    t_out = float(scenario["temp_turbine_out"][cycle_key])
    dT = T.astype(float) - t_out
    is_egs = depth3d > cutoff3d
    n_pairs = area.astype(float) / (2 * scenario["well_spacing"] ** 2)
    flowrate = xr.where(
        is_egs,
        scenario["flowrate_egs"] * n_pairs,
        scenario["flowrate_conventional"] * n_pairs,
    )
    power = flowrate * 4250.0 * dT / 1e6 * eff.astype(float)
    valid_power = np.isfinite(eff) & np.isfinite(dT) & (dT >= scenario["min_dt"])
    valid_power &= np.isfinite(power) & (power > 0.0)
    if not include_egs:
        valid_power &= ~is_egs
    power = power.where(valid_power)

    net_fraction = 1.0 - scenario["parasitic_fraction"]
    cap = power * net_fraction
    cap = cap.where(np.isfinite(cap) & (cap > 0.0))
    net_elec = power * (scenario["capacity_factor"] * 24.0 * 365.0) * net_fraction

    # Drilling depends only on depth; broadcast its costs over the doublets.
    well_cost = xr.apply_ufunc(
        drilling_cost_meur_per_well,
        T.depth_m.astype(float),
        kwargs={"spec": DRILLING_COST_CURVE_SPEC},
        dask=use_dask,
        output_dtypes=[float],
    ) * (n_pairs * 2)
    investments = (
        well_cost,
        xr.where(is_egs, scenario["stimulation_cost_per_doublet"] * n_pairs, 0.0),
        scenario["pump_cost"] * n_pairs,
        (power * 1000.0 * scenario[f"{cycle}_money_per_kw"]) / 1e6,
    )
    capex_total = sum(investments)
    lifetime = float(scenario["lifetime"])
    rate = snakemake.params.discount_rate
    annuity = (
        1 / lifetime
        if rate == 0
        else rate * (1 + rate) ** lifetime / ((1 + rate) ** lifetime - 1)
    )
    conversion = 1e6 / net_elec
    component_arrays = [cost * annuity * conversion for cost in investments]
    component_arrays.append(capex_total * scenario["fixed_OM_rate"] * conversion)
    lcoe = sum(component_arrays)
    valid_lcoe = np.isfinite(net_elec) & (net_elec > 0.0) & np.isfinite(lcoe)
    specific_capex = capex_total * 1e3 / (power * net_fraction)
    lcoe, *component_arrays, specific_capex = [
        output.where(valid_lcoe)
        .transpose("depth_m", "y", "x")
        .assign_attrs(eff.attrs, units="EUR2020/MWh_e", tech=cycle_key, crs=crs_str)
        for output in (lcoe, *component_arrays, specific_capex)
    ]
    lcoe_components = dict(zip(LCOE_COMPONENT_NAMES, component_arrays))
    cap = cap.transpose("depth_m", "y", "x").assign_attrs(
        eff.attrs, units="MW", tech=cycle_key, crs=crs_str
    )

    ds_out = xr.Dataset(
        data_vars={
            "temperature": T.astype(np.float32),
            "lcoe": lcoe.astype(np.float32),
            **{
                f"lcoe_{name}": values.astype(np.float32)
                for name, values in lcoe_components.items()
            },
            "specific_capex": specific_capex.astype(np.float32),
            "potential": cap.astype(np.float32),
            "egs_cutoff_depth_m": cutoff3d.astype(np.float32),
        },
        coords={"depth_m": lcoe["depth_m"], "y": lcoe["y"], "x": lcoe["x"]},
        attrs={
            "crs": crs_str,
            "tech": cycle_key,
            "carrier_out": "power",
            "currency": "EUR",
            "price_year": 2020,
            "include_egs": int(include_egs),
            "EGS_cutoff_rule": "cutoff(x,y)=min(EGS_cutoff, sed_thickness_m(x,y)); missing sediment thickness is 0 m",
        },
    )

    ds_out["temperature"].attrs.update(
        {"units": "degC", "long_name": "Reservoir temperature"}
    )
    ds_out["lcoe"].attrs.update(
        {"units": "EUR2020/MWh_e", "long_name": f"LCOE ({cycle_key})"}
    )
    ds_out["specific_capex"].attrs.update(
        units="EUR2020/kW_e", long_name="CAPEX per net electrical capacity"
    )
    for name in LCOE_COMPONENT_NAMES:
        label = name.replace("_", " ").title()
        ds_out[f"lcoe_{name}"].attrs.update(
            {
                "units": "EUR2020/MWh_e",
                "long_name": f"{label} contribution to LCOE ({cycle_key})",
            }
        )
    ds_out["potential"].attrs.update(
        {"units": "MW", "long_name": f"Net electric capacity ({cycle_key})"}
    )
    ds_out["egs_cutoff_depth_m"].attrs.update(
        {"units": "m", "long_name": "EGS cutoff depth"}
    )

    ds_out.to_netcdf(cube_nc)


if __name__ == "__main__":
    main()
