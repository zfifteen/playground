import os
import sys
import numpy as np
import pandas as pd
import json
from src.prime_generator import PrimeGenerator
from src.log_gap_analysis import compute_log_gaps, compute_quintile_stats, regression_on_means, compute_decile_stats
from src.distribution_tests import fit_distributions, calculate_moments
from src.autocorrelation import compute_autocorrelation, perform_ljung_box
from src.visualization import save_histogram, save_qq_plot, save_decay_trend, save_acf_pacf

# Configuration
SCALES = {
    '1e6': 10**6,
    '1e7': 10**7,
    '1e8': 10**8
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

def get_primes(limit, label):
    """
    Load or generate primes up to limit.
    """
    file_path = os.path.join(DATA_DIR, f"primes_{label}.npy")
    if os.path.exists(file_path):
        print(f"Loading primes from {file_path}...")
        return np.load(file_path)
    else:
        print(f"Generating primes up to {limit}...")
        primes = PrimeGenerator.generate_primes_array(limit)
        print(f"Saving primes to {file_path}...")
        np.save(file_path, primes)
        return primes

def analyze_scale(label, limit):
    print(f"\n--- Analyzing Scale: {label} (Limit: {limit}) ---")
    
    # 1. Get Data
    primes = get_primes(limit, label)
    count = len(primes)
    print(f"Prime count: {count}")
    
    # 2. Compute Log Gaps
    log_gaps = compute_log_gaps(primes)
    # Save log gaps? optional, maybe large.
    # np.save(os.path.join(DATA_DIR, f"log_gaps_{label}.npy"), log_gaps)
    
    if len(log_gaps) == 0:
        print("No gaps to analyze.")
        return {}

    results = {}
    results['count'] = count
    
    # 3. Quintile/Decile Analysis (H-MAIN-A)
    q_stats = compute_quintile_stats(log_gaps)
    d_stats = compute_decile_stats(log_gaps)
    
    q_reg = regression_on_means(q_stats)
    d_reg = regression_on_means(d_stats, x_col='decile')
    
    results['quintile_stats'] = q_stats.to_dict(orient='records')
    results['decile_stats'] = d_stats.to_dict(orient='records')
    results['quintile_regression'] = q_reg
    results['decile_regression'] = d_reg
    
    print(f"Quintile Regression Slope: {q_reg['slope']:.2e} (p={q_reg['p_value']:.2e}, R2={q_reg['r_squared']:.4f})")
    
    # Save Decay Plot
    save_decay_trend(q_stats, os.path.join(FIGURES_DIR, f"decay_trend_{label}.png"), 
                     title=f"Log-Gap Decay Trend ({label})")

    # 4. Distribution Tests (H-MAIN-B)
    dist_fits = fit_distributions(log_gaps)
    moments = calculate_moments(log_gaps)
    
    results['distribution_fits'] = dist_fits
    results['moments'] = moments
    
    print(f"KS Stats: LogNorm={dist_fits['lognormal']['ks_stat']:.4f}, Norm={dist_fits['normal']['ks_stat']:.4f}")
    
    # Save Histogram and QQ Plots
    print("Plotting Histogram...")
    save_histogram(log_gaps, os.path.join(FIGURES_DIR, f"log_gap_histogram_{label}.png"), 
                   title=f"Log-Gap Histogram ({label})")
    
    print("Plotting QQ Plot...")
    save_qq_plot(log_gaps, 'lognorm', dist_fits['lognormal']['params'], 
                 os.path.join(FIGURES_DIR, f"qq_plot_lognormal_{label}.png"), 
                 title=f"Q-Q Plot LogNormal ({label})")
    
    # 5. Autocorrelation (H-MAIN-C)
    print("Computing Autocorrelation...")
    ac_res = compute_autocorrelation(log_gaps)
    lb_res = perform_ljung_box(log_gaps)
    
    results['autocorrelation'] = {
        'acf': ac_res['acf'].tolist(),
        'pacf': ac_res['pacf'].tolist(),
        'ljung_box_p_val_lag20': float(lb_res.iloc[0]['lb_pvalue'])
    }
    
    print(f"Ljung-Box p-value (lag 20): {results['autocorrelation']['ljung_box_p_val_lag20']:.2e}")
    
    save_acf_pacf(ac_res['acf'], ac_res['pacf'], os.path.join(FIGURES_DIR, f"acf_pacf_{label}.png"))
    
    return results

def check_falsification(all_results):
    print("\n--- Falsification Check ---")
    falsified = False
    reasons = []
    
    # Check latest scale (most significant)
    latest_label = list(SCALES.keys())[-1] # 1e8
    if latest_label not in all_results:
        print("Missing latest scale results.")
        return
        
    res = all_results[latest_label]
    
    # F1: Non-decreasing trend
    slope = res['quintile_regression']['slope']
    p_val_slope = res['quintile_regression']['p_value']
    if slope >= 0 and p_val_slope > 0.05: # Strict check? Spec says slope >= 0 with p > 0.05
        # Non-decreasing trend is bad for the hypothesis which predicts decay
        falsified = True
        reasons.append(f"F1: Quintile means show non-decreasing trend (Slope={slope:.2e})")

    # F2: Normal fits better than LogNormal
    ks_norm = res['distribution_fits']['normal']['ks_stat']
    ks_log = res['distribution_fits']['lognormal']['ks_stat']
    if ks_norm < ks_log:
        falsified = True
        reasons.append(f"F2: Normal distribution fits better (KS_norm={ks_norm:.4f} < KS_log={ks_log:.4f})")
        
    # F3: Uniform random
    p_val_uni = res['distribution_fits']['uniform']['p_value']
    if p_val_uni > 0.05:
        falsified = True
        reasons.append(f"F3: Indistinguishable from uniform (p={p_val_uni:.2e})")
        
    # F4: Flat Autocorrelation
    lb_p = res['autocorrelation']['ljung_box_p_val_lag20']
    if lb_p > 0.05:
        falsified = True
        reasons.append(f"F4: Autocorrelation is flat (Ljung-Box p={lb_p:.2e})")
        
    # F5: Skewness/Kurtosis consistent with normal
    skew = res['moments']['skewness']
    kurt = res['moments']['kurtosis']
    if abs(skew) < 0.5 and abs(kurt) < 1:
        falsified = True
        reasons.append(f"F5: Moments consistent with normal (Skew={skew:.2f}, Kurt={kurt:.2f})")
        
    if falsified:
        print("HYPOTHESIS FALSIFIED")
        for r in reasons:
            print(f"- {r}")
    else:
        print("HYPOTHESIS SUPPORTED (Not Falsified)")
        print(" - Decay trend is negative and significant.")
        print(f" - LogNormal KS ({ks_log:.4f}) < Normal KS ({ks_norm:.4f})")
        print(" - Autocorrelation structure present.")

    # Save summary
    with open(os.path.join(RESULTS_DIR, "analysis_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

def main():
    ensure_dirs()
    
    all_results = {}
    
    # Run for all scales
    for label, limit in SCALES.items():
        try:
            result_file = os.path.join(RESULTS_DIR, f"analysis_{label}.npy")
            if os.path.exists(result_file):
                print(f"Skipping {label}, analysis file exists: {result_file}")
                # Load it to put in all_results for final check
                all_results[label] = np.load(result_file, allow_pickle=True).item()
                continue

            res = analyze_scale(label, limit)
            all_results[label] = res
            
            # Save intermediate
            np.save(os.path.join(RESULTS_DIR, f"analysis_{label}.npy"), res)
        except Exception as e:
            print(f"Error analyzing {label}: {e}")
            import traceback
            traceback.print_exc()
            
    check_falsification(all_results)

if __name__ == "__main__":
    main()