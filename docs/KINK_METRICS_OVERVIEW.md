## 🔬 Kink Detection Metrics (dV/dt Analysis)## 🔬 Kink Detection Metrics (dV/dt Analysis)



This module detects and characterizes **"kinks"** in the spike upstroke using the derivative of voltage (**dV/dt**).This module detects and characterizes **“kinks”** in the spike upstroke using the derivative of voltage (**dV/dt**).



A **kink** is defined as a **secondary peak in dV/dt that occurs before the main upstroke peak**.A **kink** is defined as a **secondary peak in dV/dt that occurs before the main upstroke peak**.



------



## 🧠 How Kink Detection Works (Per Spike)## 🧠 How Kink Detection Works (Per Spike)



For each detected spike, kink detection follows a **multi-stage filtering approach**:For each detected spike:



### Step 1: Compute dV/dt### 1. Compute dV/dt

- Calculate the derivative of voltage over the upstroke window.- Calculate the derivative of voltage over the upstroke window.

- Window: From stimulus threshold to main upstroke peak

---

---

### 2. Find all peaks in dV/dt

### Step 2: Identify the main upstroke- Use peak detection to identify **all local maxima** in the dV/dt signal.

- Find the **maximum dV/dt value** in the window — this is the true upstroke peak- These include:

- All subsequent analysis uses this as the reference point  - The **main upstroke peak** (largest peak)

  - Any **smaller bumps** (potential kinks or noise)

---

---

### Step 3a: Find candidate peaks using find_peaks()

- Use scipy's `find_peaks()` with prominence threshold:### 3. Identify the main upstroke

  - **Prominence threshold:** 10% of max dV/dt- The **maximum dV/dt value** is treated as the true upstroke.

  - **Min distance between peaks:** 5 samples

  - Result: Major, well-separated dV/dt peaks---



---### 4. Look only at peaks before the upstroke

- Kinks must occur **before** the main upstroke.

### Step 3b: Find ALL local maxima- All peaks after the upstroke are ignored.

- Additionally, identify **every local maximum** in the upstroke window

- Catches peaks that have high absolute height but low prominence---

- Calculate prominence for each local maximum

### 5. Select ONE candidate kink

---- Among all pre-upstroke peaks, we select:

  

### Step 3c: Apply multi-stage filtering to all candidates  👉 **the largest (strongest) peak**



#### Filter 1: Prominence filter (5% of max dV/dt)- Only this peak is tested as a potential kink.

- Exclude peaks with prominence below 5% of main upstroke

- Removes noise-induced tiny wiggles and fluctuations> ⚠️ Important:  

- Keeps only meaningful secondary peaks> We do **NOT** test all peaks — only the strongest pre-upstroke peak.



#### Filter 2: Distance filter (within 10 samples of upstroke)---

- **NEW in current version:** Exclude peaks within last 10 samples before main upstroke

- Prevents artifact peaks right at the upstroke edge### 6. Apply kink validation criteria

- Biological kinks should be **separated in time from the main peak**

- Avoids confusing late upstroke phase with a secondary peakThe selected peak must pass two checks:



#### Filter 3: Window filter (threshold to upstroke)#### ✅ Size (ratio threshold)

- Only consider peaks between stimulus threshold and main upstroke\[

- Excludes pre-stimulus activity\text{kink ratio} = \frac{\text{kink peak height}}{\text{main upstroke height}}

\]

---

- Must exceed a minimum threshold (e.g., 0.2)

### Step 4: Select the strongest candidate- Filters out small noise fluctuations

- Among all peaks that survived the three filters:

  ---

  👉 **Select the peak with highest dV/dt value**

#### ✅ Timing constraint

- This is the strongest (most prominent) biological feature remaining- Kink must occur within a small window before the upstroke (e.g., ≤ 1 ms)

- Prevents unrelated earlier bumps from being counted

---

---

### Step 5: Apply ratio threshold validation

### 7. Final classification

The selected peak must meet this criterion:

If both checks pass:

#### ✅ Size (ratio threshold)

$$\text{kink ratio} = \frac{\text{kink peak height}}{\text{main upstroke height}}$$```text

has_kink = True
- Must exceed minimum threshold: **≥ 0.2** (20%)
- Filters out small noise fluctuations that survived prominence filtering
- Ensures kink is meaningful relative to main upstroke
- If ratio < 0.2, no kink is detected

---

### Step 6: Compute timing metrics

If the peak passes the ratio threshold, calculate:

- **Kink interval:** Time from kink peak to main upstroke peak (in milliseconds)
- **Kink height:** Absolute dV/dt value at kink peak (in mV/ms)
- **Kink ratio:** Relative size compared to main upstroke (0.2 - 1.0)

---

### Step 7: Final classification

If all validation checks pass:

```text
has_kink = True
kink_interval_ms = [timing value]
kink_ratio = [ratio value ≥ 0.2]
kink_height_dvdt = [peak dV/dt value]
num_kinks = 1
```

If any validation check fails:

```text
has_kink = False
num_kinks = 0
kink_interval_ms = NaN
kink_ratio = NaN
kink_height_dvdt = NaN
```

---

## 📊 Configuration Parameters

All thresholds are defined in `kink_detection.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `KINK_DETECTION_PROMINENCE_PERCENT` | 0.1 | 10% of max dV/dt for initial peak detection |
| `KINK_DETECTION_MIN_DISTANCE_SAMPLES` | 5 | Minimum samples between detected peaks |
| `KINK_DETECTION_MIN_PROMINENCE_FOR_LOCAL_MAXIMA` | 0.05 | 5% of max dV/dt for local maxima filtering |
| `KINK_MAX_DISTANCE_FROM_UPSTROKE_SAMPLES` | 10 | Maximum distance from main upstroke (prevents edge artifacts) |
| `KINK_RATIO_THRESHOLD` | 0.2 | Minimum kink-to-main ratio for detection (20%) |

---

## 🔬 Biological Interpretation

A detected kink may represent:

1. **Secondary ion channel opening** during the upstroke phase
2. **Transition between different membrane conductances** (e.g., Na+ → K+ channel dynamics)
3. **Compartmental effects** in the axon or soma
4. **Different phases of action potential initiation**

**Higher kink ratio** (closer to 1.0) suggests a more dramatic secondary phase.

**Earlier timing** (>2 ms before main peak) suggests involvement in early threshold dynamics.

---

## 🎯 Key Improvements in Current Version

The current implementation includes:

✅ **Two-stage peak detection** (find_peaks + local maxima) — catches all potential kinks  
✅ **Distance filter** — prevents confusion with upstroke edge artifacts  
✅ **Prominence filtering** — removes pure noise  
✅ **Ratio validation** — ensures biological significance  
✅ **Timing metrics** — quantifies kink position relative to upstroke  

### Recent Validation (Sweep 39, Peak 1 Test)
- **Filtered data (5 kHz lowpass):** Correctly detected kink with ratio 0.251 (25.1%)
- **Unfiltered data (raw):** No detection due to noise masking (ratio 0%)
- **Conclusion:** Filter improves detection by revealing masked biological features

---

## 📝 Output Fields

For each spike with a detected kink:

- `num_kinks`: Always 0 or 1 (single kink detection)
- `kink_interval_ms`: Time from kink to main upstroke (milliseconds)
- `kink_ratio`: Relative height (kink height / main upstroke height)
- `kink_height_dvdt`: Absolute dV/dt at kink peak (mV/ms)
- `kink_idx`: Array index of kink within upstroke window

---

## 📚 References

- `kink_detection.py`: Full implementation with debug output
- `spike_detection_new.py`: Integration with main spike analysis pipeline
- `test_kink_sweep39_peak1.py`: Diagnostic test comparing filtered vs unfiltered data
- `KINK_FILTER_ANALYSIS.md`: Analysis of filter effects on kink detection
