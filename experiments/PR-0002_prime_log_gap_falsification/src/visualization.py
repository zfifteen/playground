import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import os

# Use non-interactive backend
plt.switch_backend('Agg')

def save_histogram(data, output_path, title="Log-Gap Histogram"):
    plt.figure(figsize=(10, 6))
    
    if len(data) > 100000:
        plot_data = np.random.choice(data, 100000, replace=False)
        title += " (Downsampled to 100k)"
    else:
        plot_data = data
        
    plt.hist(plot_data, bins=100, density=True, alpha=0.7, color='blue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Log Gap')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()

def save_qq_plot(data, dist_name, dist_params, output_path, title="Q-Q Plot"):
    plt.figure(figsize=(10, 6))
    
    # Downsample if too large to avoid massive plot generation time/size
    if len(data) > 10000:
        # Use random sampling or systematic sampling
        # Systematic is better for preserving distribution shape in tails?
        # Actually random is fine for QQ usually if N is large enough.
        # Let's take 10000 random points.
        plot_data = np.random.choice(data, 10000, replace=False)
        title += " (Downsampled to 10k)"
    else:
        plot_data = data
    
    if dist_name == 'lognorm':
        stats.probplot(plot_data, dist=stats.lognorm, sparams=dist_params, plot=plt)
    elif dist_name == 'norm':
        stats.probplot(plot_data, dist=stats.norm, sparams=dist_params, plot=plt)
    else:
        stats.probplot(plot_data, dist=dist_name, sparams=dist_params, plot=plt)
        
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()

def save_decay_trend(quintile_df, output_path, title="Log-Gap Decay Trend"):
    plt.figure(figsize=(10, 6))
    x = quintile_df['quintile']
    y = quintile_df['mean']
    y_err = quintile_df['std'] / np.sqrt(quintile_df['count'])
    
    plt.errorbar(x, y, yerr=y_err, fmt='o-', capsize=5, label='Quintile Mean')
    
    # Add regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    plt.plot(x, slope*x + intercept, 'r--', label=f'Fit (R²={r_value**2:.3f})')
    
    plt.title(title)
    plt.xlabel('Quintile')
    plt.ylabel('Mean Log Gap')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()

def save_acf_pacf(acf_vals, pacf_vals, output_path):
    plt.figure(figsize=(12, 5))
    
    lags = np.arange(len(acf_vals))
    
    plt.subplot(1, 2, 1)
    plt.stem(lags, acf_vals)
    plt.title('Autocorrelation Function (ACF)')
    plt.xlabel('Lag')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.stem(lags, pacf_vals)
    plt.title('Partial Autocorrelation Function (PACF)')
    plt.xlabel('Lag')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()