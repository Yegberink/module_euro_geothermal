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


VOXEL_NC = snakemake.input.nc
VAR = "temperature"

# GST-1 sediment thickness pre-aligned to the full temperature grid.
SEDIMENT_NC = snakemake.input.sed

# efficiency curve CSV
CSV_PATH = snakemake.input.eff

# EGS cutoff rule:
Z_THR_M = float(snakemake.params.EGS_cutoff)

# other configs
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


def gross_power_and_doublets(
    efficiency,
    production_temp,
    depth,
    temp_turb_out,
    sed_thickness,
    area_available,
    min_dt,
    well_spacing,
    flowrate_egs,
    flowrate_conventional,
):
    """Return gross MW, doublet count, and EGS status for one voxel."""
    # Missing sediment implies EGS at every positive depth.
    sed_ok = np.where(np.isfinite(sed_thickness), sed_thickness, 0.0).astype(np.float32)
    cutoff = np.minimum(Z_THR_M, sed_ok)
    is_egs = depth > cutoff if include_egs else np.full_like(depth, False, dtype=bool)

    # Spacing is in km and available area in km²; each doublet occupies two squares.
    n_pairs = area_available / (2 * well_spacing**2)
    flowrate_wells = np.where(
        is_egs, flowrate_egs * n_pairs, flowrate_conventional * n_pairs
    )
    dT = production_temp - temp_turb_out
    if not np.isfinite(efficiency) or not np.isfinite(dT) or dT < min_dt:
        return np.nan, n_pairs, is_egs

    Cp_water = 4250.0
    therm_out = flowrate_wells * Cp_water * dT / 1e6
    power = therm_out * efficiency
    if not np.isfinite(power) or power <= 0.0:
        return np.nan, n_pairs, is_egs
    return power, n_pairs, is_egs


def calculate_LCOE_components_vec(
    efficiency,
    production_temp,
    depth,
    temp_turb_out,
    sed_thickness,
    area_available,
    lifetime,
    discount_rate,
    cycle_type,
    stimulation_cost_per_doublet,
    pump_cost,
    fixed_OM_rate,
    parasitic_fraction,
    capacity_factor,
    well_spacing,
    flowrate_egs,
    flowrate_conventional,
    orc_money_per_kw,
    flash_money_per_kw,
    kalina_money_per_kw,
    min_dt,
):
    """Return total LCOE followed by its five additive cost contributions."""
    nan_result = tuple(np.float32(np.nan) for _ in range(1 + len(LCOE_COMPONENT_NAMES)))
    W, n_pairs, is_egs = gross_power_and_doublets(
        efficiency,
        production_temp,
        depth,
        temp_turb_out,
        sed_thickness,
        area_available,
        min_dt,
        well_spacing,
        flowrate_egs,
        flowrate_conventional,
    )
    if not np.isfinite(W):
        return nan_result
    nr_wells = n_pairs * 2

    if cycle_type == "flash":
        eur_per_kw = flash_money_per_kw
    elif cycle_type == "orc":
        eur_per_kw = orc_money_per_kw
    elif cycle_type == "kalina":
        eur_per_kw = kalina_money_per_kw
    else:
        return nan_result

    powerplant_investment = (W * 1000.0 * eur_per_kw) / 1e6
    stimulation_cost = stimulation_cost_per_doublet * n_pairs

    well_cost_no_stim = (
        drilling_cost_meur_per_well(depth, DRILLING_COST_CURVE_SPEC) * nr_wells
    )
    stimulation_investment = stimulation_cost if is_egs else 0.0
    pump_investment = pump_cost * n_pairs
    capex_total = (
        well_cost_no_stim
        + stimulation_investment
        + pump_investment
        + powerplant_investment
    )

    load_hours = capacity_factor * 24.0 * 365.0
    elec_out = W * load_hours
    net_elec = elec_out * (1.0 - parasitic_fraction)

    if (not np.isfinite(net_elec)) or (net_elec <= 0.0):
        return nan_result

    fixed_OM_costs = capex_total * fixed_OM_rate
    opex_total = fixed_OM_costs

    TT = float(lifetime)
    annuity_factor = (
        1 / TT
        if discount_rate == 0
        else (discount_rate * (1 + discount_rate) ** TT)
        / ((1 + discount_rate) ** TT - 1)
    )

    meur_to_eur_per_mwh = 1e6 / net_elec
    component_lcoe = (
        well_cost_no_stim * annuity_factor * meur_to_eur_per_mwh,
        stimulation_investment * annuity_factor * meur_to_eur_per_mwh,
        pump_investment * annuity_factor * meur_to_eur_per_mwh,
        powerplant_investment * annuity_factor * meur_to_eur_per_mwh,
        opex_total * meur_to_eur_per_mwh,
    )
    lcoe = sum(component_lcoe)

    if not np.isfinite(lcoe):
        return nan_result

    return (np.float32(lcoe), *(np.float32(value) for value in component_lcoe))


def calculate_capacity_vec(
    efficiency,
    production_temp,
    depth,
    temp_turb_out,
    sed_thickness,
    area_available,
    min_dt,
    parasitic_fraction,
    well_spacing,
    flowrate_egs,
    flowrate_conventional,
):
    """Calculate net installed capacity for one voxel."""
    power, _, _ = gross_power_and_doublets(
        efficiency,
        production_temp,
        depth,
        temp_turb_out,
        sed_thickness,
        area_available,
        min_dt,
        well_spacing,
        flowrate_egs,
        flowrate_conventional,
    )
    potential = power * (1.0 - parasitic_fraction)

    if (not np.isfinite(potential)) or (potential <= 0.0):
        return np.float32(np.nan)

    return np.float32(potential)


def main():
    """Calculate and write the LCOE cube requested by Snakemake."""
    cycle = snakemake.params.cycle
    cycle_key = {"orc": "ORC", "flash": "Flash", "kalina": "Kalina"}[cycle]

    # Explicit outputs from Snakemake rule (one tech per job)
    cube_nc = Path(snakemake.output.cube)

    ds = xr.open_dataset(VOXEL_NC)
    crs_str = ds.attrs["crs"]

    T = ds[VAR]  # (depth_m, y, x)
    fraction = load_fraction(snakemake.input.eligibility, T.x, T.y, crs_str)
    area = 100 * fraction

    max_depth = float(snakemake.params.max_depth)

    # Determine effective depth limit
    effective_max = max_depth
    if not include_egs:
        effective_max = min(effective_max, Z_THR_M)

    # Slice depth dimension
    T = T.sel(depth_m=slice(None, effective_max))

    # Re-broadcast depth coordinate to match sliced T
    depth3d = T["depth_m"].broadcast_like(T)

    # Select the exact subset of the full, pre-aligned GST-1 grid.
    sediment_ds = xr.open_dataset(SEDIMENT_NC)
    sediment_crs = sediment_ds.attrs["crs"]
    if sediment_crs != crs_str:
        raise ValueError(
            f"Sediment-grid CRS {sediment_crs!r} does not match temperature-grid CRS {crs_str!r}"
        )
    sed2d = sediment_ds["sediment_thickness_m"].sel(x=T["x"], y=T["y"])
    sed3d = sed2d.broadcast_like(T)

    # Missing sediment implies EGS at every positive depth.
    sed2d_ok = xr.where(np.isfinite(sed2d), sed2d, 0.0).astype(np.float32)
    cutoff2d = xr.apply_ufunc(np.minimum, Z_THR_M, sed2d_ok).astype(np.float32)
    cutoff2d.name = "egs_cutoff_depth_m"
    cutoff2d.attrs.update({"crs": crs_str, "units": "m"})
    cutoff3d = cutoff2d.broadcast_like(T)

    temps_curve, effs_curve = load_efficiency_curve_csv(CSV_PATH, cycle)

    use_dask = "parallelized" if hasattr(T.data, "chunks") else False

    eff = xr.apply_ufunc(
        np.interp,
        T,
        input_core_dims=[[]],
        kwargs={"xp": temps_curve, "fp": effs_curve, "left": np.nan, "right": np.nan},
        vectorize=True,
        dask=use_dask,
        output_dtypes=[np.float32],
    ).astype(np.float32)

    t_out = float(snakemake.params.temp_turbine_out[cycle_key])

    # Total LCOE and its exact additive cost-component contributions.
    lcoe_outputs = xr.apply_ufunc(
        calculate_LCOE_components_vec,
        eff,
        T,
        depth3d,
        t_out,
        sed3d,
        area,
        input_core_dims=[[], [], [], [], [], []],
        output_core_dims=[[] for _ in range(1 + len(LCOE_COMPONENT_NAMES))],
        vectorize=True,
        dask=use_dask,
        output_dtypes=[np.float32 for _ in range(1 + len(LCOE_COMPONENT_NAMES))],
        keep_attrs=True,
        kwargs=dict(
            cycle_type=cycle,
            stimulation_cost_per_doublet=snakemake.params.stimulation_cost_per_doublet,
            pump_cost=snakemake.params.pump_cost,
            fixed_OM_rate=snakemake.params.fixed_OM_rate,
            parasitic_fraction=snakemake.params.parasitic_fraction,
            capacity_factor=snakemake.params.capacity_factor,
            orc_money_per_kw=snakemake.params.orc_money_per_kw,
            flash_money_per_kw=snakemake.params.flash_money_per_kw,
            kalina_money_per_kw=snakemake.params.kalina_money_per_kw,
            lifetime=snakemake.params.lifetime,
            discount_rate=snakemake.params.discount_rate,
            min_dt=snakemake.params.min_dt,
            well_spacing=snakemake.params.well_spacing,
            flowrate_egs=snakemake.params.flowrate_egs,
            flowrate_conventional=snakemake.params.flowrate_conventional,
        ),
    )
    lcoe, *component_arrays = [
        output.transpose("depth_m", "y", "x").assign_attrs(
            units="EUR2020/MWh_e", tech=cycle_key, crs=crs_str
        )
        for output in lcoe_outputs
    ]
    lcoe_components = dict(zip(LCOE_COMPONENT_NAMES, component_arrays))

    # Installed capacity cube (net MW)
    cap = (
        xr.apply_ufunc(
            calculate_capacity_vec,
            eff,
            T,
            depth3d,
            t_out,
            sed3d,
            area,
            input_core_dims=[[], [], [], [], [], []],
            vectorize=True,
            dask=use_dask,
            output_dtypes=[np.float32],
            keep_attrs=True,
            kwargs=dict(
                min_dt=snakemake.params.min_dt,
                parasitic_fraction=snakemake.params.parasitic_fraction,
                well_spacing=snakemake.params.well_spacing,
                flowrate_egs=snakemake.params.flowrate_egs,
                flowrate_conventional=snakemake.params.flowrate_conventional,
            ),
        )
        .transpose("depth_m", "y", "x")
        .assign_attrs(units="MW", tech=cycle_key, crs=crs_str)
    )

    ds_out = xr.Dataset(
        data_vars={
            "temperature": T.astype(np.float32),
            "lcoe": lcoe.astype(np.float32),
            **{
                f"lcoe_{name}": values.astype(np.float32)
                for name, values in lcoe_components.items()
            },
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
