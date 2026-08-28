from pathlib import Path
import re
import sys
import pandas as pd
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import _schemas

if "snakemake" in dir():
    sys.stdout = sys.stderr = open(snakemake.log[0], "w", buffering=1)

# set some global style parameters for consistent, publication-quality figures
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman"]
mpl.rcParams["mathtext.fontset"] = "stix"   # makes math look Times-like
mpl.rcParams["axes.unicode_minus"] = False  # proper minus sign rendering


# -----------------------------
# Helpers: parsing + IO
# -----------------------------
ALL_CYCLES = ["orc", "flash", "kalina"]
DEFAULT_TECH_LABEL = {"orc": "ORC", "flash": "Flash", "kalina": "Kalina"}


def make_cycle_re(cycles):
    # matches ".../<cycle>_2d_lcoe.tif" or ".../<cycle>_2d_potential.tif"
    # restricted to the *included* cycles so we don't accidentally accept others
    pat = (
        r"/(?P<cycle>"
        + "|".join(map(re.escape, cycles))
        + r")_2d_(?P<kind>lcoe|potential)\.tif$"
    )
    return re.compile(pat, re.IGNORECASE)


def list_to_cycle_map(paths, kind_expected, cycles, cycle_re):
    """Turn list of paths like .../orc_2d_lcoe.tif into {"orc": path, ...}."""
    out = {}
    for p in paths:
        m = cycle_re.search(str(p).replace("\\", "/"))
        cyc = m.group("cycle").lower()
        kind = m.group("kind").lower()
        out[cyc] = str(p)
    return out


def ensure_north_up(arr, transform):
    """
    Ensure array is north-up and east-right for plotting.

    For a "north-up" raster, the affine transform typically has:
      a > 0 (pixel width), e < 0 (pixel height negative)
    If e > 0 -> y increases upward in array (south-up visually), so flip vertically.
    If a < 0 -> x decreases to the right, so flip horizontally.
    """
    out = arr
    if transform.e > 0:  # south-up
        out = np.flipud(out)
    if transform.a < 0:  # west-left
        out = np.fliplr(out)
    return out


def read_tif(path):
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float32)
        nodata = src.nodata
        profile = src.profile
        transform = src.transform

    if nodata is not None:
        a = np.where(a == nodata, np.nan, a)
    a = np.where(np.isfinite(a), a, np.nan).astype(np.float32)
    a = ensure_north_up(a, transform)
    return a, profile, transform


def stack_by_tech(paths_by_tech, cycles):
    arrs = []
    profile = None
    transform = None
    for t in cycles:
        a, p, tr = read_tif(paths_by_tech[t])
        arrs.append(a)
        profile = p if profile is None else profile
        transform = tr if transform is None else transform
    return np.stack(arrs, axis=0), profile, transform  # (tech, y, x)

def save_pixel_table(
    out_path,
    scenario,
    baseline,
    cycles,
    base_lcoe_stack,
    scen_lcoe_stack,
    base_pot_stack,
    scen_pot_stack=None,
):
    """
    Save per-pixel long-format table for boxplots/statistics.

    Output columns:
      scenario, baseline, row, col, tech,
      lcoe_base, lcoe_scen, potential_base, potential_scen, pct_change_lcoe
    """
    out_path = Path(out_path)
    rows = []

    ntech, ny, nx = base_lcoe_stack.shape
    for i, tech in enumerate(cycles):
        lcoe_base = base_lcoe_stack[i]
        lcoe_scen = scen_lcoe_stack[i]
        pot_base = base_pot_stack[i]

        if scen_pot_stack is not None:
            pot_scen = scen_pot_stack[i]
        else:
            pot_scen = np.full_like(pot_base, np.nan, dtype=np.float32)

        pct_lcoe = pct_change_map(lcoe_base, lcoe_scen)

        valid = (
            np.isfinite(lcoe_base)
            | np.isfinite(lcoe_scen)
            | np.isfinite(pot_base)
            | np.isfinite(pot_scen)
        )

        rr, cc = np.where(valid)

        df_i = pd.DataFrame({
            "scenario": scenario,
            "baseline": baseline,
            "row": rr.astype(np.int32),
            "col": cc.astype(np.int32),
            "tech": tech,
            "lcoe_base": lcoe_base[rr, cc].astype(np.float32),
            "lcoe_scen": lcoe_scen[rr, cc].astype(np.float32),
            "potential_base": pot_base[rr, cc].astype(np.float32),
            "potential_scen": pot_scen[rr, cc].astype(np.float32),
            "pct_change_lcoe": pct_lcoe[rr, cc].astype(np.float32),
        })

        rows.append(df_i)

    df = pd.concat(rows, ignore_index=True) if rows else _schemas.empty_boxplot_table()
    _schemas.write_boxplot_table(df, out_path)


# -----------------------------
# Winner/min + metrics helpers
# -----------------------------
def argmin_with_nan(stack):
    """Return winner (-1 if all NaN), best, margin (2nd-best - best)."""
    all_nan = np.all(~np.isfinite(stack), axis=0)
    stack2 = np.where(np.isfinite(stack), stack, np.inf)

    winner = np.argmin(stack2, axis=0).astype(np.int16)
    winner = np.where(all_nan, -1, winner)

    best = np.min(stack2, axis=0).astype(np.float32)
    best = np.where(all_nan, np.nan, best).astype(np.float32)

    if stack2.shape[0] < 2:
        margin = np.full(best.shape, np.nan, dtype=np.float32)
        return winner, best, margin

    sorted2 = np.sort(stack2, axis=0)
    runner = sorted2[1, :, :].astype(np.float32)
    runner = np.where(all_nan, np.nan, runner).astype(np.float32)

    margin = (runner - best).astype(np.float32)
    return winner, best, margin


def shares(winner, cycles, tech_code):
    mask = winner >= 0
    n = int(mask.sum())
    if n == 0:
        return {t: np.nan for t in cycles}
    return {t: float((winner == tech_code[t]).sum() / n) for t in cycles}


def transition_matrix(w0, w1, cycles):
    mask = (w0 >= 0) & (w1 >= 0)
    mat = np.zeros((len(cycles), len(cycles)), dtype=np.int64)
    for i in range(len(cycles)):
        for j in range(len(cycles)):
            mat[i, j] = int(((w0 == i) & (w1 == j) & mask).sum())
    return mat, int(mask.sum())


def pct_change_map(base_arr, scen_arr, eps=1e-6):
    """
    Percent change (scenario vs baseline) per-cell, robust to NaNs and tiny denominators.
    Returns pct_change with NaNs where invalid.
    """
    pct = np.full_like(base_arr, np.nan, dtype=np.float32)
    ok = np.isfinite(base_arr) & np.isfinite(scen_arr) & (np.abs(base_arr) > eps)
    pct[ok] = (100.0 * (scen_arr[ok] - base_arr[ok]) / base_arr[ok]).astype(np.float32)
    return pct


def finite_nanmin(arr):
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.nan
    return float(np.nanmin(arr[finite]))


def finite_nanmax(arr):
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.nan
    return float(np.nanmax(arr[finite]))


def finite_nanmean(arr):
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.nan
    return float(np.nanmean(arr[finite]))


# -----------------------------
# Plot helpers
# -----------------------------
def make_tech_cmap(cycles):
    tech_colors = {
        "orc":    "#0072B2",  # blue
        "flash":  "#E69F00",  # orange
        "kalina": "#009E73",  # green
    }

    colors = [tech_colors[c] for c in cycles]

    cmap = ListedColormap(colors, name="tech_cmap")
    norm = BoundaryNorm(
        np.arange(-0.5, len(cycles) + 0.5, 1.0),
        len(colors)
    )

    return colors, cmap, norm



def save_4panel_figure(
    out_fig,
    scenario,
    baseline,
    cycles,
    tech_label,
    tech_code,
    w_base,
    w_scen,
    switch,
    pct_change_sens,
):
    out_fig = Path(out_fig)
    base_img = np.where(w_base >= 0, w_base.astype(np.float32), np.nan)
    scen_img = np.where(w_scen >= 0, w_scen.astype(np.float32), np.nan)

    tech_colors, tech_cmap, tech_norm = make_tech_cmap(cycles)

    switch_colors = [
        "#BDBDBD",  # grey
        "#1F78B4",  # blue highlight
        ]
    switch_cmap = ListedColormap(switch_colors, name="switch_cmap")
    switch_norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5], ncolors=switch_cmap.N)

    fig = plt.figure(figsize=(8.2, 9.0), constrained_layout=False)

    gs = GridSpec(
        2, 3, figure=fig,
        width_ratios=[1, 1, 0.05],   # colorbar column
        wspace=0.03,                 # horizontal gap
        hspace=0.03                  # vertical gap (match wspace)
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[1, 2])  # cbar spans both rows

    # Manually control outer margins (these matter a lot)
    fig.subplots_adjust(left=0.03, right=0.93, bottom=0.05, top=0.90)


    for ax in (ax1, ax2, ax3, ax4):
        ax.set_aspect("auto")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#f2f2f2")
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.8)


    ax1.imshow(base_img, cmap=tech_cmap, norm=tech_norm, aspect="auto")
    ax2.imshow(scen_img, cmap=tech_cmap, norm=tech_norm, aspect="auto")
    ax3.imshow(switch, cmap=switch_cmap, norm=switch_norm, aspect="auto")


    vals = pct_change_sens[np.isfinite(pct_change_sens)]

    if vals.size == 0:
        # fallback
        im4 = ax4.imshow(pct_change_sens, cmap="viridis", vmin=0, vmax=1, aspect="auto")

    else:
        vmin_data = float(np.min(vals))
        vmax_data = float(np.max(vals))

        has_neg = vmin_data < 0
        has_pos = vmax_data > 0

        if has_neg and has_pos:
            # Diverging: centered at 0
            norm = TwoSlopeNorm(vmin=vmin_data, vcenter=0.0, vmax=vmax_data)
            im4 = ax4.imshow(pct_change_sens, cmap="BrBG", norm=norm, aspect="auto")

        elif has_pos:
            # Sequential positive
            im4 = ax4.imshow(pct_change_sens, cmap="cividis", vmin=0.0, vmax=vmax_data, aspect="auto")
        else:
            # Sequential negative (all <= 0): show magnitude nicely
            im4 = ax4.imshow(pct_change_sens, cmap="cividis_r", vmin=vmin_data, vmax=0.0, aspect="auto")

    cbar = fig.colorbar(im4, cax=cax)
    cbar.set_label("%", rotation=0, labelpad=10)


    panel_labels = {"a)": ax1, "b)": ax2, "c)": ax3, "d)": ax4}
    for label, ax in panel_labels.items():
        ax.text(
            0.02, 0.98, label,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=13, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
        )

    tech_patches = [
        Patch(facecolor=tech_colors[tech_code[t]], edgecolor="none", label=tech_label[t])
        for t in cycles
    ]
    switch_patches = [
        Patch(facecolor=switch_colors[0], edgecolor="none", label="Not switched"),
        Patch(facecolor=switch_colors[1], edgecolor="none", label="Switched"),
    ]

    # Legends: create them as usual...
    leg1 = ax1.legend(handles=tech_patches, loc="lower left", frameon=True, fontsize=9)
    leg2 = ax2.legend(handles=tech_patches, loc="lower left", frameon=True, fontsize=9)
    leg3 = ax3.legend(handles=switch_patches, loc="lower left", frameon=True, fontsize=9)

    # Key line: don’t let legends participate in layout sizing
    for leg in (leg1, leg2, leg3):
        leg.set_in_layout(False)

    fig.savefig(out_fig, dpi=240)
    plt.close(fig)


def save_1panel_figure(out_fig, scenario, baseline, sens_tech, pct_change_sens):
    out_fig = Path(out_fig)
    fig = plt.figure(figsize=(6.8, 6.0), constrained_layout=False)

    L, R = 0.08, 0.88
    B, T = 0.10, 0.88
    cbar_w = 0.03
    cbar_gap = 0.02

    ax = fig.add_axes([L, B, (R - L) - cbar_w - cbar_gap, (T - B)])
    cax = fig.add_axes([R - cbar_w, B, cbar_w, (T - B)])

    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#f2f2f2")
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)

    im = ax.imshow(pct_change_sens, aspect="auto")
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("%", rotation=0, labelpad=10)

    ax.text(
        0.02, 0.98, "a)",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=13, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
    )

    fig.savefig(out_fig, dpi=240)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    # cycles included (from config.yaml which_cycles passed into params)
    cycles = [c.strip().lower() for c in snakemake.params["cycles"]]
    baseline = str(snakemake.params["baseline"])
    scenario = str(snakemake.wildcards.scenario)

    # sensitivity technology for %Δ LCOE map
    sens_tech = str(snakemake.params.get("sens_tech", "orc")).strip().lower()

    tech_label = {t: DEFAULT_TECH_LABEL.get(t, t) for t in cycles}
    tech_code = {t: i for i, t in enumerate(cycles)}
    cycle_re = make_cycle_re(cycles)

    # Inputs are LISTS
    base_lcoe_paths = list_to_cycle_map(snakemake.input.base_lcoe, "lcoe", cycles, cycle_re)
    scen_lcoe_paths = list_to_cycle_map(snakemake.input.scen_lcoe, "lcoe", cycles, cycle_re)
    base_pot_paths = list_to_cycle_map(snakemake.input.base_pot, "potential", cycles, cycle_re)
    # scen_pot not used here (kept for symmetry / future); you can drop if not in Snakemake rule
    scen_pot_paths = list_to_cycle_map(snakemake.input.scen_pot, "potential", cycles, cycle_re)

    # Read/stack rasters (already north-up corrected)
    base_stack, _, _ = stack_by_tech(base_lcoe_paths, cycles)
    scen_stack, _, _ = stack_by_tech(scen_lcoe_paths, cycles)
    base_pot_stack, _, _ = stack_by_tech(base_pot_paths, cycles)
    scen_pot_stack, _, _ = stack_by_tech(scen_pot_paths, cycles)

    save_pixel_table(
        snakemake.output.boxplot_data,
        scenario=scenario,
        baseline=baseline,
        cycles=cycles,
        base_lcoe_stack=base_stack,
        scen_lcoe_stack=scen_stack,
        base_pot_stack=base_pot_stack,
        scen_pot_stack=scen_pot_stack,
    )

    # Always compute %Δ for *sens_tech* only (not “best tech”)
    sens_idx = tech_code[sens_tech]

    lcoe_base_sens = base_stack[sens_idx].astype(np.float32)
    lcoe_scen_sens = scen_stack[sens_idx].astype(np.float32)
    pct_change_sens = pct_change_map(lcoe_base_sens, lcoe_scen_sens)

    # Summary stats for %Δ map
    mask_pct = np.isfinite(pct_change_sens)
    if mask_pct.any():
        mean_abs_pct_change_sens = float(np.nanmean(np.abs(pct_change_sens[mask_pct])))
        max_abs_pct_change_sens = float(np.nanmax(np.abs(pct_change_sens[mask_pct])))
    else:
        mean_abs_pct_change_sens = np.nan
        max_abs_pct_change_sens = np.nan

    # If single-tech mode: only plot 1 panel, and write minimal metrics (no switch rates)
    if len(cycles) == 1:
        min_lcoe_base = finite_nanmin(base_stack[0])
        min_lcoe_scen = finite_nanmin(scen_stack[0])
        mean_lcoe_base = finite_nanmean(base_stack[0])
        mean_lcoe_scen = finite_nanmean(scen_stack[0])
        max_lcoe_base = finite_nanmax(base_stack[0])
        max_lcoe_scen = finite_nanmax(scen_stack[0])

        save_1panel_figure(
            snakemake.output.fig,
            scenario=scenario,
            baseline=baseline,
            sens_tech=sens_tech,
            pct_change_sens=pct_change_sens,
        )

        out_metrics = Path(snakemake.output.metrics)
        with out_metrics.open("w") as f:
            f.write("metric\tvalue\n")
            f.write(f"scenario\t{scenario}\n")
            f.write(f"baseline\t{baseline}\n")
            f.write(f"cycles_included\t{','.join(cycles)}\n")
            f.write(f"sens_tech\t{sens_tech}\n")

            # Defined metrics for single-tech mode
            f.write(f"mean_abs_pct_change_sens\t{mean_abs_pct_change_sens}\n")
            f.write(f"max_abs_pct_change_sens\t{max_abs_pct_change_sens}\n")
            f.write(f"min_lcoe_baseline\t{min_lcoe_base}\n")
            f.write(f"min_lcoe_scenario\t{min_lcoe_scen}\n")
            f.write(f"mean_lcoe_baseline\t{mean_lcoe_base}\n")
            f.write(f"mean_lcoe_scenario\t{mean_lcoe_scen}\n")
            f.write(f"max_lcoe_baseline\t{max_lcoe_base}\n")
            f.write(f"max_lcoe_scenario\t{max_lcoe_scen}\n")
            f.write(f"min_lcoe_baseline_{cycles[0]}\t{min_lcoe_base}\n")
            f.write(f"min_lcoe_scenario_{cycles[0]}\t{min_lcoe_scen}\n")
            f.write(f"mean_lcoe_baseline_{cycles[0]}\t{mean_lcoe_base}\n")
            f.write(f"mean_lcoe_scenario_{cycles[0]}\t{mean_lcoe_scen}\n")
            f.write(f"max_lcoe_baseline_{cycles[0]}\t{max_lcoe_base}\n")
            f.write(f"max_lcoe_scenario_{cycles[0]}\t{max_lcoe_scen}\n")

            # Keep these keys present (as nan) so downstream code that expects them won’t crash
            f.write("switch_rate_pixels\tnan\n")
            f.write("switch_rate_potential_weighted\tnan\n")
            f.write("mean_abs_pct_change_best\tnan\n")
            f.write("mean_abs_pct_change_best_margin_weighted\tnan\n")
            f.write("mean_abs_pct_change_best_simple\tnan\n")
            f.write("max_abs_pct_change_best\tnan\n")
            f.write("transition_total_pixels\tnan\n")

            # Shares are not that meaningful with one tech, but harmless and well-defined
            f.write(f"share_baseline_{cycles[0]}\t1.0\n")
            f.write(f"share_scenario_{cycles[0]}\t1.0\n")

        _schemas.SensitivityMetricsSchema.validate(pd.read_csv(out_metrics, sep="\t", dtype=str))
        return

    # -----------------------------
    # Multi-tech mode (>=2)
    # -----------------------------
    w_base, best_base, margin_base = argmin_with_nan(base_stack)
    w_scen, best_scen, _margin_scen = argmin_with_nan(scen_stack)

    valid_both = (w_base >= 0) & (w_scen >= 0)
    switch = np.where(valid_both, (w_base != w_scen).astype(np.float32), np.nan)

    # Baseline best-tech potential as weights for weighted switch rate
    base_pot_stack, _, _ = stack_by_tech(base_pot_paths, cycles)
    pot_base_best = np.full_like(best_base, np.nan, dtype=np.float32)
    for i, t in enumerate(cycles):
        pot_base_best = np.where((w_base == i), base_pot_stack[i], pot_base_best)

    # Shares / transitions
    share_base = shares(w_base, cycles, tech_code)
    share_scen = shares(w_scen, cycles, tech_code)
    mat, n_both = transition_matrix(w_base, w_scen, cycles)

    switch_rate = float(np.nanmean(switch)) if np.isfinite(switch).any() else np.nan

    mask_w = np.isfinite(switch) & np.isfinite(pot_base_best) & (pot_base_best > 0)
    if mask_w.any():
        switch_rate_w = float(
            np.nansum(switch[mask_w] * pot_base_best[mask_w]) / np.nansum(pot_base_best[mask_w])
        )
    else:
        switch_rate_w = np.nan

    # Keep your old “best-tech” change metrics too (still useful generally),
    # but NOTE: these are based on best tech, not sens tech.
    pct_change_best = pct_change_map(best_base, best_scen)
    min_lcoe_base = finite_nanmin(best_base)
    min_lcoe_scen = finite_nanmin(best_scen)
    mean_lcoe_base = finite_nanmean(best_base)
    mean_lcoe_scen = finite_nanmean(best_scen)
    max_lcoe_base = finite_nanmax(best_base)
    max_lcoe_scen = finite_nanmax(best_scen)
    mask_best = np.isfinite(pct_change_best)
    if mask_best.any():
        mean_abs_pct_change_best_simple = float(np.nanmean(np.abs(pct_change_best[mask_best])))
        max_abs_pct_change_best = float(np.nanmax(np.abs(pct_change_best[mask_best])))
    else:
        mean_abs_pct_change_best_simple = np.nan
        max_abs_pct_change_best = np.nan

    eps = 1e-6
    w_margin = 1.0 / (margin_base + eps)
    mask_m = np.isfinite(pct_change_best) & np.isfinite(w_margin) & (w_base >= 0)
    if mask_m.any():
        mean_abs_pct_best = float(np.nanmean(np.abs(pct_change_best[mask_m])))
        mean_abs_pct_best_margin = float(
            np.nansum(np.abs(pct_change_best[mask_m]) * w_margin[mask_m]) / np.nansum(w_margin[mask_m])
        )
    else:
        mean_abs_pct_best = np.nan
        mean_abs_pct_best_margin = np.nan

    # Figure (4 panels) — panel d is now %Δ for sens_tech only
    save_4panel_figure(
        snakemake.output.fig,
        scenario=scenario,
        baseline=baseline,
        cycles=cycles,
        tech_label=tech_label,
        tech_code=tech_code,
        w_base=w_base,
        w_scen=w_scen,
        switch=switch,
        pct_change_sens=pct_change_sens,
    )

    # Metrics TSV
    out_metrics = Path(snakemake.output.metrics)

    with out_metrics.open("w") as f:
        f.write("metric\tvalue\n")
        f.write(f"scenario\t{scenario}\n")
        f.write(f"baseline\t{baseline}\n")
        f.write(f"cycles_included\t{','.join(cycles)}\n")
        f.write(f"sens_tech\t{sens_tech}\n")

        # switch metrics (only meaningful in >=2 mode)
        f.write(f"switch_rate_pixels\t{switch_rate}\n")
        f.write(f"switch_rate_potential_weighted\t{switch_rate_w}\n")

        # “best tech” metrics (kept for continuity)
        f.write(f"mean_abs_pct_change_best\t{mean_abs_pct_best}\n")
        f.write(f"mean_abs_pct_change_best_margin_weighted\t{mean_abs_pct_best_margin}\n")
        f.write(f"mean_abs_pct_change_best_simple\t{mean_abs_pct_change_best_simple}\n")
        f.write(f"max_abs_pct_change_best\t{max_abs_pct_change_best}\n")

        # NEW/explicit: sens-tech metrics (what panel d shows)
        f.write(f"mean_abs_pct_change_sens\t{mean_abs_pct_change_sens}\n")
        f.write(f"max_abs_pct_change_sens\t{max_abs_pct_change_sens}\n")
        f.write(f"min_lcoe_baseline\t{min_lcoe_base}\n")
        f.write(f"min_lcoe_scenario\t{min_lcoe_scen}\n")
        f.write(f"mean_lcoe_baseline\t{mean_lcoe_base}\n")
        f.write(f"mean_lcoe_scenario\t{mean_lcoe_scen}\n")
        f.write(f"max_lcoe_baseline\t{max_lcoe_base}\n")
        f.write(f"max_lcoe_scenario\t{max_lcoe_scen}\n")

        # shares
        for t in cycles:
            f.write(f"min_lcoe_baseline_{t}\t{finite_nanmin(base_stack[tech_code[t]])}\n")
            f.write(f"min_lcoe_scenario_{t}\t{finite_nanmin(scen_stack[tech_code[t]])}\n")
            f.write(f"mean_lcoe_baseline_{t}\t{finite_nanmean(base_stack[tech_code[t]])}\n")
            f.write(f"mean_lcoe_scenario_{t}\t{finite_nanmean(scen_stack[tech_code[t]])}\n")
            f.write(f"max_lcoe_baseline_{t}\t{finite_nanmax(base_stack[tech_code[t]])}\n")
            f.write(f"max_lcoe_scenario_{t}\t{finite_nanmax(scen_stack[tech_code[t]])}\n")
            f.write(f"share_baseline_{t}\t{share_base[t]}\n")
            f.write(f"share_scenario_{t}\t{share_scen[t]}\n")

        # transitions
        f.write(f"transition_total_pixels\t{n_both}\n")
        for i, t_from in enumerate(cycles):
            for j, t_to in enumerate(cycles):
                f.write(f"transition_{t_from}_to_{t_to}\t{mat[i,j]}\n")

    _schemas.SensitivityMetricsSchema.validate(pd.read_csv(out_metrics, sep="\t", dtype=str))

if __name__ == "__main__":
    main()
