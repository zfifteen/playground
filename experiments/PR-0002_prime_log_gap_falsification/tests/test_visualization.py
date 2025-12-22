import unittest
import numpy as np
import pandas as pd
import os
import tempfile
import shutil
from src.visualization import (
    save_histogram,
    save_qq_plot,
    save_decay_trend,
    save_acf_pacf
)


class TestVisualization(unittest.TestCase):
    
    def setUp(self):
        """Create temporary directory for test outputs"""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.test_dir)
    
    def test_save_histogram_simple(self):
        """Test histogram saving with simple data"""
        np.random.seed(42)
        data = np.random.random(1000)
        output_path = os.path.join(self.test_dir, 'histogram.png')
        
        save_histogram(data, output_path, title="Test Histogram")
        
        # Check that file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Check that file has non-zero size
        self.assertGreater(os.path.getsize(output_path), 0)
    
    def test_save_histogram_large_data(self):
        """Test histogram with large dataset - should downsample"""
        np.random.seed(42)
        data = np.random.random(200000)
        output_path = os.path.join(self.test_dir, 'histogram_large.png')
        
        save_histogram(data, output_path, title="Large Dataset")
        
        # Should still create the file
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)
    
    def test_save_histogram_custom_title(self):
        """Test histogram with custom title"""
        np.random.seed(42)
        data = np.random.random(500)
        output_path = os.path.join(self.test_dir, 'histogram_custom.png')
        
        save_histogram(data, output_path, title="Custom Title")
        
        self.assertTrue(os.path.exists(output_path))
    
    def test_save_qq_plot_normal(self):
        """Test Q-Q plot for normal distribution"""
        np.random.seed(42)
        data = np.random.normal(loc=5, scale=2, size=1000)
        output_path = os.path.join(self.test_dir, 'qq_normal.png')
        
        # Normal distribution params (mu, std)
        params = (5, 2)
        
        save_qq_plot(data, 'norm', params, output_path, title="Normal Q-Q")
        
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)
    
    def test_save_qq_plot_lognormal(self):
        """Test Q-Q plot for lognormal distribution"""
        np.random.seed(42)
        data = np.random.lognormal(mean=0, sigma=0.5, size=1000)
        output_path = os.path.join(self.test_dir, 'qq_lognormal.png')
        
        # Lognormal params (shape, loc, scale)
        from scipy import stats
        shape, loc, scale = stats.lognorm.fit(data)
        params = (shape, loc, scale)
        
        save_qq_plot(data, 'lognorm', params, output_path, title="Lognormal Q-Q")
        
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)
    
    def test_save_qq_plot_large_data(self):
        """Test Q-Q plot with large dataset - should downsample"""
        np.random.seed(42)
        data = np.random.normal(0, 1, size=50000)
        output_path = os.path.join(self.test_dir, 'qq_large.png')
        
        params = (0, 1)
        
        save_qq_plot(data, 'norm', params, output_path, title="Large Q-Q")
        
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)
    
    def test_save_decay_trend_simple(self):
        """Test decay trend plot with quintile data"""
        quintile_df = pd.DataFrame({
            'quintile': [1, 2, 3, 4, 5],
            'mean': [5.0, 4.5, 4.0, 3.5, 3.0],
            'std': [0.5, 0.5, 0.5, 0.5, 0.5],
            'count': [100, 100, 100, 100, 100]
        })
        output_path = os.path.join(self.test_dir, 'decay_trend.png')
        
        save_decay_trend(quintile_df, output_path, title="Decay Trend")
        
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)
    
    def test_save_decay_trend_increasing(self):
        """Test decay trend plot with increasing means"""
        quintile_df = pd.DataFrame({
            'quintile': [1, 2, 3, 4, 5],
            'mean': [1.0, 2.0, 3.0, 4.0, 5.0],
            'std': [0.2, 0.2, 0.2, 0.2, 0.2],
            'count': [50, 50, 50, 50, 50]
        })
        output_path = os.path.join(self.test_dir, 'decay_trend_inc.png')
        
        save_decay_trend(quintile_df, output_path, title="Increasing Trend")
        
        self.assertTrue(os.path.exists(output_path))
    
    def test_save_acf_pacf_simple(self):
        """Test ACF/PACF plot saving"""
        np.random.seed(42)
        # Create simple ACF/PACF values
        acf_vals = np.array([1.0, 0.5, 0.25, 0.125, 0.06, 0.03])
        pacf_vals = np.array([1.0, 0.5, 0.1, 0.05, 0.02, 0.01])
        
        output_path = os.path.join(self.test_dir, 'acf_pacf.png')
        
        save_acf_pacf(acf_vals, pacf_vals, output_path)
        
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)
    
    def test_save_acf_pacf_many_lags(self):
        """Test ACF/PACF plot with many lags"""
        np.random.seed(42)
        nlags = 50
        acf_vals = np.exp(-np.arange(nlags+1) * 0.1)  # Exponential decay
        pacf_vals = np.exp(-np.arange(nlags+1) * 0.2)
        
        output_path = os.path.join(self.test_dir, 'acf_pacf_many.png')
        
        save_acf_pacf(acf_vals, pacf_vals, output_path)
        
        self.assertTrue(os.path.exists(output_path))
    
    def test_save_acf_pacf_negative_values(self):
        """Test ACF/PACF plot with negative correlations"""
        acf_vals = np.array([1.0, -0.3, -0.5, 0.2, 0.1, -0.1])
        pacf_vals = np.array([1.0, -0.3, 0.1, -0.05, 0.02, 0.01])
        
        output_path = os.path.join(self.test_dir, 'acf_pacf_neg.png')
        
        save_acf_pacf(acf_vals, pacf_vals, output_path)
        
        self.assertTrue(os.path.exists(output_path))
    
    def test_multiple_plots_in_same_directory(self):
        """Test creating multiple plots in same directory"""
        np.random.seed(42)
        data = np.random.random(1000)
        
        # Create multiple plots
        save_histogram(data, os.path.join(self.test_dir, 'plot1.png'))
        save_histogram(data, os.path.join(self.test_dir, 'plot2.png'))
        save_histogram(data, os.path.join(self.test_dir, 'plot3.png'))
        
        # All should exist
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'plot1.png')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'plot2.png')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'plot3.png')))


if __name__ == '__main__':
    unittest.main()
