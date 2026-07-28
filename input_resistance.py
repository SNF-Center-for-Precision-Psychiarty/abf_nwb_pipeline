# This file calculates the input resistance from the hyperpolarizing I-V curve
"""
Input resistance from the hyperpolarizing branch of the I-V curve.

Method:
    1. Take every sweep whose injected current is hyperpolarizing (< 0 pA).
    2. For each, measure the STEADY-STATE voltage and current, i.e. the mean
       over the last 80ms of the stimulus (1ms buffer before the step ends).
       By then the capacitive charging transient and the HCN sag have settled,
       so the deflection reflects the ohmic membrane resistance.
    3. Fit V = R * I + V0 across those sweeps by least squares.
       The slope R is in mV/pA (= GOhm) and is reported in MOhm.

Why the whole hyperpolarizing branch: a single step, or a mean taken across
the full stimulus window, mixes the peak deflection with the sag relaxation
and turns cell-to-cell differences in HCN current into apparent differences in
input resistance. Fitting the branch also gives an R^2 that flags non-ohmic or
unstable recordings.
"""
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for speed
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path
from analysis_config import (
    HYPERPOLARIZING_THRESHOLD_PA,
    STEADY_STATE_WINDOW_S,
    STEADY_STATE_BUFFER_S,
    MIN_IV_FIT_POINTS,
    MIN_IV_CURRENT_RANGE_PA,
    PA_MV_SLOPE_TO_MOHM,
)
from sag_current import find_negative_sweeps


def _steady_state_mean(sweep_df, stim_start_s, stim_end_s):
    """
    Mean of `value` over the last STEADY_STATE_WINDOW_S of the stimulus,
    ending STEADY_STATE_BUFFER_S before the step turns off.

    Returns np.nan if the sweep has no samples in that window.
    """
    if len(sweep_df) == 0:
        return np.nan

    t = sweep_df["t_s"].values
    v = sweep_df["value"].values

    finite = ~np.isnan(v)
    t, v = t[finite], v[finite]
    if len(t) == 0:
        return np.nan

    win_end = stim_end_s - STEADY_STATE_BUFFER_S
    win_start = max(win_end - STEADY_STATE_WINDOW_S, stim_start_s)

    in_window = (t >= win_start) & (t <= win_end)
    if in_window.sum() == 0:
        return np.nan

    return float(np.mean(v[in_window]))


def get_input_resistance(df, df_pA, bundle_path, sweep_config=None):

    # Calculate input resistance
    print("CALCULATING INPUT RESISTANCE")
    print("NOTE: Rin is the slope of steady-state voltage vs injected current")
    print("      fitted across ALL hyperpolarizing sweeps (< 0 pA)")
    bundle_path = Path(bundle_path)

    if sweep_config is None:
        raise ValueError("sweep_config is required for input resistance calculation")

    # Load manifest to detect protocol type
    man = json.loads((bundle_path / "manifest.json").read_text())
    is_mixed = "stimulus" in man.get("tables", {}) and "response" in man.get("tables", {})
    print(f"Protocol type: {'MIXED' if is_mixed else 'SINGLE'}")

    # Reference stimulus window — only a fallback for sweeps whose own window is
    # missing from sweep_config. Times in sweep_config are absolute for both
    # single and mixed protocols, so no per-protocol conversion is needed.
    try:
        # Prefer a valid sweep, but fall back to any sweep with stimulus timing
        # so input resistance can still run on bundles where 0 sweeps are valid.
        valid_windows = None
        fallback_sweep = None
        fallback_windows = None

        for sweep_id, sweep_data in sweep_config.get("sweeps", {}).items():
            windows = sweep_data.get("windows") or {}
            if windows.get("stimulus_start_s") is None or windows.get("stimulus_end_s") is None:
                continue
            if sweep_data.get("valid", False):
                valid_windows = windows
                break
            if fallback_sweep is None:
                fallback_sweep = sweep_id
                fallback_windows = windows

        if valid_windows is not None:
            ref_windows = valid_windows
        elif fallback_windows is not None:
            print(f"WARNING: No sweep marked valid in sweep_config — "
                  f"deriving stimulus window from sweep {fallback_sweep} (invalid).")
            ref_windows = fallback_windows
        else:
            raise ValueError("Could not find stimulus window in sweep_config")

        t_stim_min = ref_windows["stimulus_start_s"]
        t_stim_max = ref_windows["stimulus_end_s"]
        print(f"Reference stimulus window: [{t_stim_min:.6f}, {t_stim_max:.6f}] s")
    except (KeyError, TypeError) as e:
        raise ValueError(f"Failed to extract stimulus window from sweep_config: {e}")

    df_mv_all = df
    df_pA_all = df_pA

    # If mV data has multiple channels (can happen after hardware malfunction fix),
    # filter to keep only one channel (the one we selected as correct)
    if "channel_index" in df_mv_all.columns:
        channels = df_mv_all["channel_index"].unique()
        if len(channels) > 1:
            # Use only the first channel (or the most common one)
            primary_channel = df_mv_all["channel_index"].value_counts().idxmax()
            print(f"  Note: Multiple mV channels detected. Using channel {primary_channel}")
            df_mv_all = df_mv_all[df_mv_all["channel_index"] == primary_channel]

    # Select the hyperpolarizing branch from the per-sweep analysis table, the
    # same source sag_current.py uses, rather than re-detecting peaks here.
    df_analysis = pd.read_parquet(bundle_path / "analysis.parquet")
    hyper_sweeps = find_negative_sweeps(df_analysis, threshold_pA=HYPERPOLARIZING_THRESHOLD_PA)

    # A hyperpolarizing step should never fire; if one did, the sweep is either
    # mislabelled or has rebound spikes bleeding into the window — drop it.
    spiking = set(
        df_analysis[
            df_analysis["spike_frequency_Hz"].fillna(0) > 0
        ]["sweep"].astype(int).tolist()
    )
    dropped_spiking = sorted(set(hyper_sweeps) & spiking)
    hyper_sweeps = sorted(set(hyper_sweeps) - spiking)

    print(f"  Hyperpolarizing sweeps (< {HYPERPOLARIZING_THRESHOLD_PA:g} pA): {len(hyper_sweeps)}")
    if dropped_spiking:
        print(f"  Excluded {len(dropped_spiking)} hyperpolarizing sweep(s) with detected spikes: {dropped_spiking}")
    print(f"  Steady-state window: last {STEADY_STATE_WINDOW_S * 1000:.0f}ms of stimulus "
          f"({STEADY_STATE_BUFFER_S * 1000:.0f}ms buffer before end)")

    current_vals = []
    voltage_vals = []
    sweeps_used = []

    for sweep_number in hyper_sweeps:
        # Per-sweep stimulus window; sweeps can differ in timing within a bundle
        sweep_str = str(int(sweep_number))
        sweep_windows = sweep_config.get("sweeps", {}).get(sweep_str, {}).get("windows", {})
        sweep_t_stim_min = sweep_windows.get("stimulus_start_s", t_stim_min)
        sweep_t_stim_max = sweep_windows.get("stimulus_end_s", t_stim_max)

        mv_sweep = df_mv_all[df_mv_all["sweep"] == sweep_number]
        pa_sweep = df_pA_all[df_pA_all["sweep"] == sweep_number]

        if len(mv_sweep) == 0 or len(pa_sweep) == 0:
            print(f"  WARNING: Sweep {sweep_number} missing "
                  f"{'mV' if len(mv_sweep) == 0 else 'pA'} data — skipped")
            continue

        voltage_ss = _steady_state_mean(mv_sweep, sweep_t_stim_min, sweep_t_stim_max)
        current_ss = _steady_state_mean(pa_sweep, sweep_t_stim_min, sweep_t_stim_max)

        if np.isnan(voltage_ss) or np.isnan(current_ss):
            print(f"  WARNING: Sweep {sweep_number} has no samples in the steady-state window — skipped")
            continue

        current_vals.append(current_ss)
        voltage_vals.append(voltage_ss)
        sweeps_used.append(int(sweep_number))
        print(f"  Sweep {sweep_number}: I={current_ss:.2f} pA, V={voltage_ss:.4f} mV")

    print(f"  Using {len(sweeps_used)} sweeps for the I-V fit")

    rin_mohm = np.nan
    intercept = np.nan
    r_squared = np.nan

    current_range = float(np.ptp(current_vals)) if current_vals else 0.0

    if len(current_vals) < MIN_IV_FIT_POINTS:
        print(f"ERROR: Only {len(current_vals)} hyperpolarizing sweep(s) available — "
              f"need at least {MIN_IV_FIT_POINTS} to fit the I-V curve")
    elif current_range < MIN_IV_CURRENT_RANGE_PA:
        print(f"ERROR: Hyperpolarizing sweeps span only {current_range:.2f} pA "
              f"(< {MIN_IV_CURRENT_RANGE_PA:g} pA) — cannot fit I-V curve")
    else:
        if len(current_vals) < 3:
            print(f"WARNING: Only {len(current_vals)} sweeps — the fit is exactly "
                  f"determined, so R^2 says nothing about fit quality")

        current_vals = np.array(current_vals)
        voltage_vals = np.array(voltage_vals)
        slope, intercept, r_value, p_value, std_err = linregress(current_vals, voltage_vals)
        # slope is mV/pA = 1e-3 V / 1e-12 A = 1e9 Ohm = GOhm, so x1000 for MOhm
        rin_mohm = slope * PA_MV_SLOPE_TO_MOHM
        r_squared = r_value ** 2
        print(f"Rin = {rin_mohm:.2f} MOhm (R^2 = {r_squared:.3f}, n = {len(sweeps_used)} sweeps)")

        # Plot I-V curve with best fit
        plot_dir = bundle_path / "Input_Resistance"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.scatter(current_vals, voltage_vals, s=25, alpha=0.8, label="Steady-state (per sweep)")

        # Best-fit line across the full current range
        i_line = np.linspace(current_vals.min(), current_vals.max(), 100)
        plt.plot(i_line, intercept + slope * i_line, 'r',
                 label=f'Fit: V = {slope:.4f}*I + {intercept:.2f}')

        plt.xlabel("Current (pA)")
        plt.ylabel("Voltage (mV)")
        plt.title(f"I-V curve: {len(sweeps_used)} hyperpolarizing sweeps\n"
                  f"Rin = {rin_mohm:.1f} MOhm, R² = {r_squared:.3f}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / 'InputResistance.jpeg')
        plt.close()

    # Update manifest
    manifest_path = bundle_path / 'manifest.json'

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    analysis_section = manifest.setdefault("analysis", {})
    analysis_section["input_resistance"] = float(rin_mohm)
    analysis_section["input_resistance_intercept_mV"] = float(intercept)
    analysis_section["input_resistance_r2"] = float(r_squared)
    analysis_section["input_resistance_n_sweeps"] = len(sweeps_used)
    analysis_section["input_resistance_sweeps"] = sweeps_used

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
