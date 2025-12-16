# Sleep-Apnea-Dataset
ECG Dataset
#%% md

#%%
""""""""

"""HR & HRV FEATURE EXTRACTION FROM ECG (.dat + .hea)
====================================================

Author  : Md. Sohel Miah
Purpose : Extract Heart Rate (HR) and Heart Rate Variability (HRV)
          features for biomedical / ML research.
Input   : PhysioNet ECG files (.dat + .hea)
Output  : CSV file with HR & HRV features
====================================================
"""

# ==================================================
# 1. IMPORT REQUIRED LIBRARIES
# ==================================================

import os  # File & directory operations
import glob  # Locate multiple files
import numpy as np  # Numerical computation
import pandas as pd  # CSV and table handling
import warnings  # Ignore warnings

import wfdb  # Read .dat + .hea files
from scipy.signal import welch, find_peaks

warnings.filterwarnings("ignore")


# ==================================================
# 2. RR INTERVAL EXTRACTION FROM .dat + .hea
# ==================================================

def extract_rr_from_dat(record_path):
    """
    Extract RR intervals (in seconds) from ECG (.dat + .hea)
    """

    # Read ECG record (WFDB automatically reads .hea + .dat)
    record = wfdb.rdrecord(record_path)

    fs = record.fs  # Sampling frequency
    ecg = record.p_signal[:, 0]  # First ECG channel

    # Simple R-peak detection
    peaks, _ = find_peaks(
        ecg,
        distance=0.6 * fs,  # minimum RR ~ 600 ms
        height=np.mean(ecg)
    )

    # RR intervals in seconds
    rr_intervals = np.diff(peaks) / fs

    return rr_intervals


# ==================================================
# 3. HEART RATE CALCULATION FUNCTION
# ==================================================

def compute_heart_rate(rr_intervals):
    """
    Calculate heart rate features from RR intervals.
    RR intervals must be in seconds.
    """

    rr = np.array(rr_intervals)
    hr = 60 / rr

    return {
        "mean_hr": np.mean(hr),
        "min_hr": np.min(hr),
        "max_hr": np.max(hr)
    }


# ==================================================
# 4. TIME-DOMAIN HRV FEATURES
# ==================================================

def time_domain_hrv(rr_intervals):
    """
    Extract time-domain HRV features.
    """

    rr = np.array(rr_intervals)

    mean_rr = np.mean(rr)
    sdnn = np.std(rr, ddof=1)

    diff_rr = np.diff(rr)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))

    nn50 = np.sum(np.abs(diff_rr) > 0.05)
    pnn50 = (nn50 / len(diff_rr)) * 100

    return {
        "mean_rr": mean_rr,
        "sdnn": sdnn,
        "rmssd": rmssd,
        "nn50": nn50,
        "pnn50": pnn50
    }


# ==================================================
# 5. FREQUENCY-DOMAIN HRV FEATURES
# ==================================================

def frequency_domain_hrv(rr_intervals, fs=4.0):
    """
    Extract frequency-domain HRV features using Welch PSD.
    """

    rr = np.array(rr_intervals)

    time_axis = np.cumsum(rr)
    interp_time = np.arange(0, time_axis[-1], 1 / fs)
    interp_rr = np.interp(interp_time, time_axis, rr)

    freqs, psd = welch(interp_rr, fs=fs, nperseg=256)

    vlf_band = (freqs >= 0.003) & (freqs < 0.04)
    lf_band = (freqs >= 0.04) & (freqs < 0.15)
    hf_band = (freqs >= 0.15) & (freqs < 0.4)

    vlf = np.trapz(psd[vlf_band], freqs[vlf_band])
    lf = np.trapz(psd[lf_band], freqs[lf_band])
    hf = np.trapz(psd[hf_band], freqs[hf_band])

    lf_hf_ratio = lf / hf if hf != 0 else np.nan

    return {
        "vlf": vlf,
        "lf": lf,
        "hf": hf,
        "lf_hf_ratio": lf_hf_ratio
    }


# ==================================================
# 6. MAIN PROGRAM
# ==================================================

def main():
    # Input directory containing .dat + .hea files
    input_dir = r"D:\Sleep Apnea ecg Dataset\apnea-ecg-database-1.0.0"

    # Output files
    output_file = "hr_hrv_summary.csv"
    error_file = "hr_hrv_missing.csv"

    results = []
    errors = []

    # Load all .dat files
    files = glob.glob(os.path.join(input_dir, "*.dat"))

    for file in files:
        try:
            record_path = os.path.splitext(file)[0]  # remove .dat

            rr_intervals = extract_rr_from_dat(record_path)

            if len(rr_intervals) < 10:
                raise ValueError("Insufficient RR intervals")

            hr_features = compute_heart_rate(rr_intervals)
            td_features = time_domain_hrv(rr_intervals)
            fd_features = frequency_domain_hrv(rr_intervals)

            row = {
                "file_name": os.path.basename(file),
                **hr_features,
                **td_features,
                **fd_features
            }

            results.append(row)

        except Exception as e:
            errors.append({
                "file_name": os.path.basename(file),
                "error": str(e)
            })

    # Save outputs
    pd.DataFrame(results).to_csv(output_file, index=False)
    pd.DataFrame(errors).to_csv(error_file, index=False)

    print("HR and HRV feature extraction completed successfully.")


# ==================================================
# 7. PROGRAM ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()
#%%
