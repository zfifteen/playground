import unittest
import numpy as np
import pandas as pd
from src.log_gap_analysis import (
    compute_log_gaps,
    compute_quintile_stats,
    compute_decile_stats,
    regression_on_means
)


class TestLogGapAnalysis(unittest.TestCase):
    
    def test_compute_log_gaps_simple(self):
        """Test log gap computation with simple primes"""
        primes = np.array([2, 3, 5, 7, 11])
        log_gaps = compute_log_gaps(primes)
        
        # Manually compute expected values
        expected = np.log(primes[1:]) - np.log(primes[:-1])
        
        self.assertEqual(len(log_gaps), 4)
        np.testing.assert_array_almost_equal(log_gaps, expected)
    
    def test_compute_log_gaps_empty(self):
        """Test with empty array"""
        primes = np.array([])
        log_gaps = compute_log_gaps(primes)
        self.assertEqual(len(log_gaps), 0)
    
    def test_compute_log_gaps_single(self):
        """Test with single prime"""
        primes = np.array([2])
        log_gaps = compute_log_gaps(primes)
        self.assertEqual(len(log_gaps), 0)
    
    def test_compute_quintile_stats_simple(self):
        """Test quintile statistics computation"""
        # Create 100 log gaps
        log_gaps = np.random.random(100)
        
        df = compute_quintile_stats(log_gaps)
        
        # Should have 5 quintiles
        self.assertEqual(len(df), 5)
        
        # Check column names
        self.assertIn('quintile', df.columns)
        self.assertIn('mean', df.columns)
        self.assertIn('std', df.columns)
        self.assertIn('count', df.columns)
        
        # Quintile numbers should be 1-5
        self.assertEqual(list(df['quintile']), [1, 2, 3, 4, 5])
        
        # Each quintile should have ~20 elements
        for count in df['count'][:-1]:
            self.assertEqual(count, 20)
    
    def test_compute_quintile_stats_known_values(self):
        """Test with known values"""
        # Create array with known structure: 5 groups of 10 identical values
        log_gaps = np.concatenate([
            np.ones(10) * 1.0,
            np.ones(10) * 2.0,
            np.ones(10) * 3.0,
            np.ones(10) * 4.0,
            np.ones(10) * 5.0
        ])
        
        df = compute_quintile_stats(log_gaps)
        
        # Check means
        np.testing.assert_array_almost_equal(df['mean'].values, [1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Standard deviations should be 0 since all values in each quintile are identical
        np.testing.assert_array_almost_equal(df['std'].values, [0.0, 0.0, 0.0, 0.0, 0.0])
    
    def test_compute_decile_stats_simple(self):
        """Test decile statistics computation"""
        # Create 100 log gaps
        log_gaps = np.random.random(100)
        
        df = compute_decile_stats(log_gaps)
        
        # Should have 10 deciles
        self.assertEqual(len(df), 10)
        
        # Check column names
        self.assertIn('decile', df.columns)
        self.assertIn('mean', df.columns)
        self.assertIn('std', df.columns)
        self.assertIn('count', df.columns)
        
        # Decile numbers should be 1-10
        self.assertEqual(list(df['decile']), list(range(1, 11)))
    
    def test_regression_on_means_simple(self):
        """Test regression computation"""
        # Create simple linear data
        stats_df = pd.DataFrame({
            'quintile': [1, 2, 3, 4, 5],
            'mean': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        result = regression_on_means(stats_df)
        
        # Check keys in result
        self.assertIn('slope', result)
        self.assertIn('intercept', result)
        self.assertIn('r_squared', result)
        self.assertIn('p_value', result)
        self.assertIn('std_err', result)
        
        # For perfect linear relationship, R² should be 1
        self.assertAlmostEqual(result['r_squared'], 1.0, places=10)
        self.assertAlmostEqual(result['slope'], 1.0, places=10)
        self.assertAlmostEqual(result['intercept'], 0.0, places=10)
    
    def test_regression_on_means_custom_columns(self):
        """Test regression with custom column names"""
        stats_df = pd.DataFrame({
            'decile': [1, 2, 3, 4, 5],
            'mean': [2.0, 4.0, 6.0, 8.0, 10.0]
        })
        
        result = regression_on_means(stats_df, x_col='decile', y_col='mean')
        
        # For y = 2x, slope should be 2
        self.assertAlmostEqual(result['slope'], 2.0, places=10)
        self.assertAlmostEqual(result['intercept'], 0.0, places=10)
        self.assertAlmostEqual(result['r_squared'], 1.0, places=10)


if __name__ == '__main__':
    unittest.main()
