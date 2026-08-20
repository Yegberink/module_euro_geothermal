from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy.optimize import curve_fit
import _schemas

def _parse_scalar(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, int, float)):
        return value

    text = str(value).strip()
    if text.lower() in ("true", "yes", "y", "1"):
        return True
    if text.lower() in ("false", "no", "n", "0"):
        return False
    try:
        return float(text)
    except ValueError:
        return text


def _scenario_row():
    rows = pd.read_csv(snakemake.input.scenario_table, sep="\t")
    row = rows.loc[rows["scenario"].astype(str) == str(snakemake.wildcards.scenario)]
    return {key: _parse_scalar(value) for key, value in row.iloc[0].items()}


SCENARIO = _scenario_row()


def cfg(key, default=None):
    value = SCENARIO.get(key)
    if value is not None:
        return value

    if key.startswith("temp_turbine_out_"):
        cycle = key.removeprefix("temp_turbine_out_")
        return snakemake.config["temp_turbine_out"][cycle]

    if default is not None:
        return snakemake.config.get(key, default)
    return snakemake.config[key]


def drilling_cost_curve_definitions():
    definitions = dict(snakemake.config["drilling_cost_curve_definitions"])
    if "getem_2017" in definitions:
        definitions["getem_2017"] = dict(definitions["getem_2017"])
        definitions["getem_2017"]["csv_path"] = snakemake.input.drilling_costs_2017
    if "getem_2025" in definitions:
        definitions["getem_2025"] = dict(definitions["getem_2025"])
        definitions["getem_2025"]["csv_path"] = snakemake.input.drilling_costs_2025
    return definitions


VOXEL_NC = snakemake.input.nc
VAR = "temperature"

# Sediment thickness (Laske/Masters 3-layer) NetCDF
SED_NC = snakemake.input.sed  # has thk1/thk2/thk3 on latitude/longitude grid (km)

# efficiency curve CSV
CSV_PATH = snakemake.input.eff

# EGS cutoff rule:
Z_THR_M = float(cfg("EGS_cutoff"))

# How to handle NaNs in sed thickness after resampling:
SED_NAN_POLICY = cfg("SED_NAN_POLICY")

# Turbine outlet temps (°C), dict like {"ORC": 70.0, "Flash": 150.0, "Kalina": 80.0}
temp_turbine_out = {
    cycle: float(cfg(f"temp_turbine_out_{cycle}"))
    for cycle in ("ORC", "Flash", "Kalina")
}

# other configs
carrier_out = cfg("carrier_out")
include_egs = bool(cfg("include_egs"))
DRILLING_COST_CURVE = str(cfg("drilling_cost_curve", "thermogis"))
DRILLING_COST_CURVE_DEFINITIONS = drilling_cost_curve_definitions()

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


DRILLING_COST_CURVE_SPEC = _resolve_drilling_curve_definition(
    DRILLING_COST_CURVE,
    DRILLING_COST_CURVE_DEFINITIONS,
)


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
            depth_m,
            float(spec["a"]),
            float(spec["b"]),
            float(spec["c"]),
        )
        return cost_meur * _usd_to_eur2020_factor(spec["currency_year"])


def load_efficiency_curve_csv(csv_path, cycle_key):
    """
    cycle_key is expected to be one of: ORC, Flash, Kalina (matching your config keys).
    This function lowercases it to match CSV columns like orc_temperature/orc_efficiency.
    """
    cycle = str(cycle_key).lower()

    df = pd.read_csv(csv_path)

    t_col = f"{cycle}_temperature"
    e_col = f"{cycle}_efficiency"

    sub = df[[t_col, e_col]].dropna()

    temps = sub[t_col].astype(float).to_numpy()

    # Handle percentages like "7.0%"
    eff_raw = sub[e_col].astype(str).str.rstrip("%").astype(float)
    effs = eff_raw.to_numpy() / 100.0

    return temps, effs


def interp_eff(temps_grid, temps_curve, effs_curve):
    eff = np.interp(temps_grid, temps_curve, effs_curve, left=np.nan, right=np.nan)
    return eff.astype(np.float32, copy=False)


def sed_from_laske_nc_to_grid(
    nc_path,
    x_coords,
    y_coords,
    crs_target,
    lat_name="latitude",
    lon_name="longitude",
    thk_vars=("thk1", "thk2", "thk3"),
    resampling=Resampling.nearest,  # nearest is usually better for threshold logic
):
    """
    Read Laske/Masters 3-layer sediment thickness (km) from NetCDF and resample/reproject
    onto voxel grid (x,y in EPSG:3857). Returns sed thickness in meters as (y,x).
    """
    ds = xr.open_dataset(nc_path)

    thk_total_km = sum(ds[v] for v in thk_vars)
    thk_total_m = (thk_total_km * 1000.0).astype(np.float32)

    lat = thk_total_m[lat_name].values
    lon = thk_total_m[lon_name].values

    da = thk_total_m.transpose(lat_name, lon_name)
    src = da.values.astype(np.float32)

    fill = da.attrs.get("_FillValue", None) or da.attrs.get("missing_value", None)
    if fill is not None:
        src = np.where(src == fill, np.nan, src)

    dlon = float(lon[1] - lon[0])
    dlat = float(lat[1] - lat[0])

    lat_is_ascending = np.all(np.diff(lat) > 0)
    if lat_is_ascending:
        src = src[::-1, :]
        lat_top = float(lat.max())
    else:
        lat_top = float(lat[0])

    src_transform = from_origin(
        float(lon.min() - 0.5 * dlon),
        float(lat_top + 0.5 * abs(dlat)),
        abs(dlon),
        abs(dlat),
    )

    dx = float(x_coords[1] - x_coords[0])
    dy = float(y_coords[1] - y_coords[0])
    x_min = float(x_coords.min() - 0.5 * dx)
    y_max = float(y_coords.max() + 0.5 * abs(dy))
    dst_transform = from_origin(x_min, y_max, abs(dx), abs(dy))

    dst = np.full((len(y_coords), len(x_coords)), np.nan, dtype=np.float32)

    reproject(
        source=src,
        destination=dst,
        src_transform=src_transform,
        src_crs=CRS.from_epsg(4326),
        dst_transform=dst_transform,
        dst_crs=CRS.from_user_input(crs_target),
        resampling=resampling,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    # y_coords in dataset may be ascending; align to it
    if np.all(np.diff(y_coords) > 0):
        dst = dst[::-1, :]

    sed2d = xr.DataArray(dst, dims=("y", "x"), coords={"y": y_coords, "x": x_coords}, name="sed_thickness_m")
    sed2d.attrs.update({"crs": crs_target, "units": "m"})
    return sed2d


def _apply_sed_nan_policy(sed_thickness, z_thr=Z_THR_M, policy=SED_NAN_POLICY):
    """
    Replace NaNs in sediment thickness according to policy.
    - depth_only: NaN -> z_thr (EGS only deeper than z_thr)
    - always_egs: NaN -> 0 (EGS at all depths)
    """
    if policy == "depth_only":
        fill = float(z_thr)
    elif policy == "always_egs":
        fill = 0.0
    return np.where(np.isfinite(sed_thickness), sed_thickness, fill).astype(np.float32)


LCOE_COMPONENT_NAMES = (
    "drilling",
    "stimulation",
    "pump",
    "power_plant",
    "fixed_om",
)


def calculate_LCOE_components_vec(
    efficiency,
    production_temp,
    depth,
    temp_turb_out,
    sed_thickness,
    lifetime,
    cycle_type,
    stimulation_cost_per_doublet,
    pump_cost,
    fixed_OM_rate,
    parasitic_fraction,
    capacity_factor,
    well_spacing,
    flowrate_egs,
    flowrate_conventional,
    area_available,
    orc_money_per_kw,
    flash_money_per_kw,
    kalina_money_per_kw,
    min_dt
):
    """Return total LCOE followed by its five additive cost contributions."""
    nan_result = tuple(np.float32(np.nan) for _ in range(1 + len(LCOE_COMPONENT_NAMES)))
    Cp_water = 4250.0
    MIN_DT = min_dt

    sed_ok = _apply_sed_nan_policy(sed_thickness, z_thr=Z_THR_M, policy=SED_NAN_POLICY)
    cutoff = np.minimum(Z_THR_M, sed_ok)

    if include_egs:
        is_egs = depth > cutoff
    else:
        is_egs = np.full_like(depth, False, dtype=bool)

    # `well_spacing` and `area_available` are both expressed in km / km^2 here.
    doublet_occupation = 2 * well_spacing**2  # km^2 per doublet
    n_pairs = area_available / doublet_occupation

    nr_wells = n_pairs * 2
    flowrate_wells = np.where(is_egs, flowrate_egs * n_pairs, flowrate_conventional * n_pairs)

    dT = production_temp - temp_turb_out

    # Early rejection
    if (not np.isfinite(efficiency)) or (not np.isfinite(dT)) or (dT < MIN_DT):
        return nan_result

    therm_out = flowrate_wells * Cp_water * dT / 1e6
    power_out = therm_out * efficiency

    if (not np.isfinite(power_out)) or (power_out <= 0.0):
        return nan_result

    W = power_out

    ct = str(cycle_type).lower()
    if ct == "flash":
        eur_per_kw = flash_money_per_kw
    elif ct == "orc":
        eur_per_kw = orc_money_per_kw
    elif ct == "kalina":
        eur_per_kw = kalina_money_per_kw
    else:
        return nan_result

    powerplant_investment = (W * 1000.0 * eur_per_kw) / 1e6
    stimulation_cost = stimulation_cost_per_doublet * n_pairs

    well_cost_no_stim = _drilling_cost_meur_per_well(depth) * nr_wells
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

    discount_rate = 0.075
    TT = float(lifetime)
    annuity_factor = (discount_rate * (1 + discount_rate) ** TT) / ((1 + discount_rate) ** TT - 1)

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
    min_dt,
    parasitic_fraction,
    well_spacing,
    flowrate_egs,
    flowrate_conventional,
    area_available,
):
    Cp_water = 4250.0
    MIN_DT = min_dt

    sed_ok = _apply_sed_nan_policy(sed_thickness, z_thr=Z_THR_M, policy=SED_NAN_POLICY)
    cutoff = np.minimum(Z_THR_M, sed_ok)

    if include_egs:
        is_egs = depth > cutoff
    else:
        is_egs = np.full_like(depth, False, dtype=bool)

    # `well_spacing` and `area_available` are both expressed in km / km^2 here,
    # so `n_pairs` is the number of geothermal doublets that fit in one grid cell.
    doublet_occupation = 2 * well_spacing**2  # km^2 per doublet

    n_pairs = area_available / doublet_occupation

    flowrate_wells = np.where(is_egs, flowrate_egs * n_pairs, flowrate_conventional * n_pairs)

    dT = production_temp - temp_turb_out

    # Early rejection
    if (not np.isfinite(efficiency)) or (not np.isfinite(dT)) or (dT < MIN_DT):
        return np.float32(np.nan)

    therm_out = flowrate_wells * Cp_water * dT / 1e6
    power_out_gross = therm_out * efficiency

    if (not np.isfinite(power_out_gross)) or (power_out_gross <= 0.0):
        return np.float32(np.nan)

    W_net = power_out_gross * (1.0 - parasitic_fraction)
    potential = W_net

    if (not np.isfinite(potential)) or (potential <= 0.0):
        return np.float32(np.nan)

    return np.float32(potential)


def _normalize_cycle_param(x: str) -> str:
    x = str(x).strip().lower()
    return x

def main():
    cycle = _normalize_cycle_param(snakemake.params.get("cycle"))
    cycle_key = {"orc": "ORC", "flash": "Flash", "kalina": "Kalina"}[cycle]

    # Explicit outputs from Snakemake rule (one tech per job)
    cube_nc = Path(snakemake.output.cube)
    qc_lcoe = Path(snakemake.output.qc_lcoe) if hasattr(snakemake.output, "qc_lcoe") else None
    qc_pot = Path(snakemake.output.qc_potential) if hasattr(snakemake.output, "qc_potential") else None

    ds = xr.open_dataset(VOXEL_NC)
    crs_str = str(ds.attrs.get("crs", "")).strip() or "EPSG:3857"

    T = ds[VAR]  # (depth_m, y, x)

    max_depth = float(cfg("max_depth", np.inf))
    include_egs = bool(cfg("include_egs", True))
    egs_cutoff = float(cfg("EGS_cutoff", np.inf))

    # Determine effective depth limit
    effective_max = max_depth
    if not include_egs:
        effective_max = min(effective_max, egs_cutoff)

    # Slice depth dimension
    T = T.sel(depth_m=slice(None, effective_max))

    # Re-broadcast depth coordinate to match sliced T
    depth3d = T["depth_m"].broadcast_like(T)

    x_coords = T["x"].values
    y_coords = T["y"].values

    # --- load and align sediment thickness to voxel grid ---
    sed2d = sed_from_laske_nc_to_grid(SED_NC, x_coords, y_coords, crs_str)
    sed3d = sed2d.broadcast_like(T)

    # --- cutoff depth map (2D) consistent with NaN policy ---
    sed2d_ok = xr.apply_ufunc(_apply_sed_nan_policy, sed2d)
    cutoff2d = xr.apply_ufunc(np.minimum, Z_THR_M, sed2d_ok).astype(np.float32)
    cutoff2d.name = "egs_cutoff_depth_m"
    cutoff2d.attrs.update({"crs": crs_str, "units": "m"})
    cutoff3d = cutoff2d.broadcast_like(T)

    temps_curve, effs_curve = load_efficiency_curve_csv(CSV_PATH, cycle_key)

    use_dask = "parallelized" if hasattr(T.data, "chunks") else False

    eff = xr.apply_ufunc(
        interp_eff,
        T,
        input_core_dims=[[]],
        kwargs={"temps_curve": temps_curve, "effs_curve": effs_curve},
        vectorize=True,
        dask=use_dask,
        output_dtypes=[np.float32],
    )

    t_out = float(temp_turbine_out[cycle_key])

    # Total LCOE and its exact additive cost-component contributions.
    lcoe_outputs = xr.apply_ufunc(
        calculate_LCOE_components_vec,
        eff,
        T,
        depth3d,
        t_out,
        sed3d,
        input_core_dims=[[], [], [], [], []],
        output_core_dims=[[] for _ in range(1 + len(LCOE_COMPONENT_NAMES))],
        vectorize=True,
        dask=use_dask,
        output_dtypes=[np.float32 for _ in range(1 + len(LCOE_COMPONENT_NAMES))],
        keep_attrs=True,
        kwargs=dict(
            cycle_type=cycle,
            stimulation_cost_per_doublet=cfg("stimulation_cost_per_doublet"),
            pump_cost=cfg("pump_cost"),
            fixed_OM_rate=cfg("fixed_OM_rate"),
            parasitic_fraction=cfg("parasitic_fraction"),
            capacity_factor=cfg("capacity_factor"),
            orc_money_per_kw=cfg("orc_money_per_kw"),
            flash_money_per_kw=cfg("flash_money_per_kw"),
            kalina_money_per_kw=cfg("kalina_money_per_kw"),
            lifetime=cfg("lifetime"),
            min_dt=cfg("min_dt"),
            well_spacing=cfg("well_spacing"),
            flowrate_egs=cfg("flowrate_egs"),
            flowrate_conventional=cfg("flowrate_conventional"),
            area_available=100,
        ),
    )
    lcoe, *component_arrays = [
        output.transpose("depth_m", "y", "x").assign_attrs(
            units="€/MWh_e",
            tech=str(cycle_key),
            crs=crs_str,
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
            input_core_dims=[[], [], [], [], []],
            vectorize=True,
            dask=use_dask,
            output_dtypes=[np.float32],
            keep_attrs=True,
            kwargs=dict(
                min_dt=cfg("min_dt"),
                parasitic_fraction=cfg("parasitic_fraction"),
                well_spacing=cfg("well_spacing"),
                flowrate_egs=cfg("flowrate_egs"),
                flowrate_conventional=cfg("flowrate_conventional"),
                area_available=100,
            ),
        )
        .transpose("depth_m", "y", "x")
        .assign_attrs(units="MW", tech=str(cycle_key), crs=crs_str)
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
        coords={
            "depth_m": lcoe["depth_m"],
            "y": lcoe["y"],
            "x": lcoe["x"],
        },
        attrs={
            "crs": crs_str,
            "tech": str(cycle_key),
            "carrier_out": str(carrier_out),
            "include_egs": bool(include_egs),
            "SED_NAN_POLICY": str(SED_NAN_POLICY),
            "EGS_cutoff_rule": "cutoff(x,y)=min(EGS_cutoff, sed_thickness_m(x,y)) with NaN policy applied",
        },
    )

    ds_out["temperature"].attrs.update({"units": "degC", "long_name": "Reservoir temperature"})
    ds_out["lcoe"].attrs.update({"units": "€/MWh_e", "long_name": f"LCOE ({cycle_key})"})
    for name in LCOE_COMPONENT_NAMES:
        label = name.replace("_", " ").title()
        ds_out[f"lcoe_{name}"].attrs.update(
            {
                "units": "€/MWh_e",
                "long_name": f"{label} contribution to LCOE ({cycle_key})",
            }
        )
    ds_out["potential"].attrs.update({"units": "MW", "long_name": f"Net electric capacity ({cycle_key})"})
    ds_out["egs_cutoff_depth_m"].attrs.update({"units": "m", "long_name": "EGS cutoff depth"})

    ds_out.to_netcdf(cube_nc)

if __name__ == "__main__":
    main()
