"""
Calculate sag current from hyperpolarizing current sweeps.

Sag is the voltage response during hyperpolarizing current injection,
caused by HCN (hyperpolarization-activated cyclic nucleotide-gated) channels.

Theory:
    When negative current is injected:
    1. Voltage initially hyperpolarizes (becomes more negative)
    2. Over time, HCN channels open, allowing positive current to flow back in
    3. Voltage "sags" or relaxes back toward less negative values
    4. The amount of sag indicates HCN channel activity

Measurements:
    - Sag voltage (mV): V_steady_state - V_min
    - Sag ratio (dimensionless): sag_voltage / (V_baseline - V_min)
      * 0 = no sag (no HCN channels)
      * 1 = complete relaxation back to baseline
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def find_most_hyperpolarizing_sweep(analysis_parquet: pd.DataFrame):
    """
    Find the sweep with the most negative injected current.
    
    Returns:
        sweep number of the most hyperpolarizing current injection
    """

    # Find row with minimum current
    idx = analysis_parquet['avg_injected_current_pA'].idxmin()

    sweep = analysis_parquet.loc[idx, 'sweep']

    return int(sweep)


def measure_voltage_response(
    mv_data: pd.DataFrame,
    sweep: int,
    sweep_config: dict = None,
    sampling_rate: float = 200000  # Hz
) -> dict:
    """
    Measure key voltage points during a hyperpolarizing sweep.

    Updated definitions:
    - v_baseline = most negative voltage in first 80 ms of stimulus
    - v_min      = most negative voltage during entire stimulus
    - v_steady   = mean voltage in last 80 ms of stimulus
                   with 1 ms buffer before stimulus end
    """

    sweep_data = mv_data[mv_data['sweep'] == sweep]

    if len(sweep_data) == 0:
        return None

    times = sweep_data['t_s'].values
    voltages = sweep_data['value'].values

    # Get stimulus window from config
    if sweep_config is None:
        sweep_config = {}

    sweep_str = str(int(sweep))

    if sweep_str in sweep_config:
        windows = sweep_config[sweep_str].get('windows', {})
        stimulus_start = windows.get('stimulus_start_s', 0.01)
        stimulus_end = windows.get('stimulus_end_s', times[-1])
    else:
        stimulus_start = 0.01
        stimulus_end = times[-1]

    # Extract stimulus portion
    stim_mask = (times >= stimulus_start) & (times <= stimulus_end)

    if stim_mask.sum() == 0:
        return None

    stim_times = times[stim_mask]
    stim_voltages = voltages[stim_mask]

    #v_baseline
    baseline_window_end = stimulus_start + 0.080

    baseline_mask = (times >= stimulus_start) & (times <= baseline_window_end)

    baseline_voltages = voltages[baseline_mask]

    if len(baseline_voltages) > 0:
        v_baseline = np.min(baseline_voltages)
    else:
        v_baseline = np.nan

    #v_min
    v_min = np.min(stim_voltages)
    t_v_min = stim_times[np.argmin(stim_voltages)]

    #v_steady
    # Last 80 ms of stimulus with 1 ms buffer before end
    steady_start = stimulus_end - 0.080 - 0.001  # 81ms before end = 80ms window + 1ms buffer
    steady_end = stimulus_end - 0.001            # 1ms before end (buffer)

    steady_mask = (times >= steady_start) & (times <= steady_end)

    steady_voltages = voltages[steady_mask]

    if len(steady_voltages) > 0:
        v_steady = np.mean(steady_voltages)
    else:
        v_steady = np.nan

    return {
        'v_baseline': v_baseline,
        'v_min': v_min,
        'v_steady': v_steady,
        't_v_min': t_v_min,
    }


def calculate_sag(voltage_response: dict) -> dict:
    """
    Calculate sag metrics from voltage response measurements.
    
    Args:
        voltage_response: Dict from measure_voltage_response()
    
    Returns:
        Dictionary with:
        - 'sag_voltage_mV': Absolute sag (V_steady - V_min, in mV)
        - 'sag_ratio': Normalized sag (0-1)
                      0 = no recovery, 1 = complete recovery
        - 'sag_percent': Sag as percentage of total hyperpolarization
    """
    if voltage_response is None:
        return None
    
    v_baseline = voltage_response['v_baseline']
    v_min = voltage_response['v_min']
    v_steady = voltage_response['v_steady']
    
    # Total hyperpolarization (how far voltage dropped)
    total_hyperpol = v_baseline - v_min
    
    # Sag voltage (how much it recovered)
    sag_voltage = v_steady - v_min
    
    # Sag ratio (what fraction of the drop did it recover from?)
    if total_hyperpol != 0:
        sag_ratio = sag_voltage / total_hyperpol
    else:
        sag_ratio = 0
    
    # Sag as percentage
    sag_percent = sag_ratio * 100
    
    return {
        'sag_voltage_mV': sag_voltage,
        'sag_ratio': sag_ratio,
        'sag_percent': sag_percent,
        'total_hyperpol_mV': total_hyperpol,
        'v_baseline_mV': v_baseline,
        'v_min_mV': v_min,
        'v_steady_mV': v_steady,
    }


def calculate_sag_for_bundle(
    bundle_dir: str,
    verbose: bool = True
) -> dict:
    """
    Calculate sag for the most hyperpolarizing sweep in a bundle.
    
    Args:
        bundle_dir: Path to bundle directory
        verbose: Print progress information
    
    Returns:
        Dictionary with results:
        - 'hyper_sweeps': List containing the most hyperpolarizing sweep
        - 'sag_results': Dict mapping sweep_num → sag measurements
        - 'summary': Summary statistics
    """

    bundle_path = Path(bundle_dir)

    # -----------------------------
    # Locate required files
    # -----------------------------
    mv_files = list(bundle_path.rglob("mV_*.parquet"))
    analysis_files = list(bundle_path.rglob("analysis.parquet"))
    sweep_config_files = list(bundle_path.rglob("sweep_config.json"))

    if not mv_files or not analysis_files:
        if verbose:
            print(f"⚠ Missing parquet files in {bundle_dir}")
        return None

    # -----------------------------
    # Load data
    # -----------------------------
    mv_data = pd.read_parquet(mv_files[0])
    analysis_data = pd.read_parquet(analysis_files[0])

    # -----------------------------
    # Load sweep_config if available
    # -----------------------------
    sweep_config = {}

    if sweep_config_files:
        import json
        with open(sweep_config_files[0], 'r') as f:
            config_data = json.load(f)
            if 'sweeps' in config_data:
                sweep_config = config_data['sweeps']

    # -----------------------------
    # Identify most hyperpolarizing sweep
    # -----------------------------
    most_hyper_sweep = find_most_hyperpolarizing_sweep(analysis_data)
    hyper_sweeps = [most_hyper_sweep]

    if verbose:
        current = analysis_data.loc[
            analysis_data['sweep'] == most_hyper_sweep,
            'avg_injected_current_pA'
        ].iloc[0]

        print(f"\n[Sag Current Analysis]")
        print(f"  Using most hyperpolarizing sweep:")
        print(f"  Sweep {most_hyper_sweep} ({current:.0f} pA)")

    # -----------------------------
    # Measure sag
    # -----------------------------
    sag_results = {}
    sag_ratios = []

    for sweep in hyper_sweeps:

        voltage_response = measure_voltage_response(
            mv_data,
            sweep,
            sweep_config=sweep_config
        )

        if voltage_response is None:
            continue

        sag_measurements = calculate_sag(voltage_response)

        sag_results[sweep] = sag_measurements
        sag_ratios.append(sag_measurements['sag_ratio'])

        if verbose:
            current = analysis_data[
                analysis_data['sweep'] == sweep
            ]['avg_injected_current_pA'].iloc[0]

            print(f"\n  Sweep {sweep} ({current:.0f} pA):")
            print(f"    V_baseline: {sag_measurements['v_baseline_mV']:.2f} mV")
            print(f"    V_min:      {sag_measurements['v_min_mV']:.2f} mV")
            print(f"    V_steady:   {sag_measurements['v_steady_mV']:.2f} mV")
            print(f"    Sag voltage: {sag_measurements['sag_voltage_mV']:.2f} mV")
            print(f"    Sag ratio:   {sag_measurements['sag_ratio']:.3f} ({sag_measurements['sag_percent']:.1f}%)")

        # -----------------------------
        # Generate diagnostic plot
        # -----------------------------
        if verbose:

            sweep_data = mv_data[mv_data['sweep'] == sweep]

            times = sweep_data['t_s'].values
            voltages = sweep_data['value'].values

            sweep_str = str(int(sweep))

            if sweep_str in sweep_config:
                windows = sweep_config[sweep_str]['windows']
                stimulus_start = windows['stimulus_start_s']
                stimulus_end = windows['stimulus_end_s']
            else:
                stimulus_start = 0.01
                stimulus_end = times[-1]

            plot_sag_diagnostics(
                bundle_dir,
                times,
                voltages,
                stimulus_start,
                stimulus_end,
                sag_measurements['v_baseline_mV'],
                sag_measurements['v_min_mV'],
                sag_measurements['v_steady_mV']
            )

    # -----------------------------
    # Summary statistics
    # -----------------------------
    if sag_ratios:

        mean_sag = np.mean(sag_ratios)
        std_sag = np.std(sag_ratios)

        summary = {
            'n_sweeps': len(sag_results),
            'mean_sag_ratio': mean_sag,
            'std_sag_ratio': std_sag,
            'min_sag_ratio': np.min(sag_ratios),
            'max_sag_ratio': np.max(sag_ratios),
        }

    else:
        summary = None

    if verbose and summary:

        print(f"\n  ─── SUMMARY ───")
        print(f"  Mean sag ratio: {summary['mean_sag_ratio']:.3f} ± {summary['std_sag_ratio']:.3f}")
        print(f"  Range: {summary['min_sag_ratio']:.3f} - {summary['max_sag_ratio']:.3f}")

    return {
        'hyper_sweeps': hyper_sweeps,
        'sag_results': sag_results,
        'summary': summary,
    }



def plot_sag_diagnostics(bundle_path, times, voltages, stimulus_start, stimulus_end,
                         v_baseline, v_min, v_steady):

    baseline_end = stimulus_start + 0.080
    steady_start = stimulus_end - 0.081
    steady_end = stimulus_end - 0.001

    plt.figure(figsize=(8,4))

    plt.plot(times, voltages, color="black", linewidth=1)

    # stimulus region
    plt.axvspan(stimulus_start, stimulus_end,
                color="blue", alpha=0.1, label="v_min window")

    # baseline window
    plt.axvspan(stimulus_start, baseline_end,
                color="green", alpha=0.2, label="v_baseline window")

    # steady window
    plt.axvspan(steady_start, steady_end,
                color="orange", alpha=0.2, label="v_steady window")

    # key points
    plt.scatter([],[], label=f"v_baseline={v_baseline:.2f} mV", color="green")
    plt.scatter([],[], label=f"v_min={v_min:.2f} mV", color="red")
    plt.scatter([],[], label=f"v_steady={v_steady:.2f} mV", color="orange")

    plt.axhline(v_baseline, color="green", linestyle="--")
    plt.axhline(v_min, color="red", linestyle="--")
    plt.axhline(v_steady, color="orange", linestyle="--")

    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.title("Sag Diagnostic Plot")

    plt.legend()
    plt.tight_layout()
    plot_dir = Path(bundle_path)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / "SagCurrent.jpeg", dpi=300)
    plt.close()

# FOR TESTING
# if __name__ == "__main__":
#     # Test on test_bundle2
#     bundle_dir = "test_bundle2/sub-131113"
#     results = calculate_sag_for_bundle(bundle_dir, verbose=True)
    
#     if results:
#         print("\n" + "="*70)
#         print("Results can be integrated into analysis pipeline")
#         print("="*70)
