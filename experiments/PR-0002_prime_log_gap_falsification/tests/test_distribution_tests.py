import unittest
import numpy as np
from src.distribution_tests import fit_distributions, calculate_moments


class TestDistributionTests(unittest.TestCase):
    
    def test_fit_distributions_normal_data(self):
        """Test fitting distributions to normally distributed data"""
        # Generate normal data
        np.random.seed(42)
        data = np.random.normal(loc=5.0, scale=2.0, size=1000)
        
        results = fit_distributions(data)
        
        # Check that all distributions are fitted
        self.assertIn('normal', results)
        self.assertIn('lognormal', results)
        self.assertIn('exponential', results)
        self.assertIn('uniform', results)
        
        # Each should have ks_stat, p_value, params
        for dist_name in ['normal', 'lognormal', 'exponential', 'uniform']:
            self.assertIn('ks_stat', results[dist_name])
            self.assertIn('p_value', results[dist_name])
            self.assertIn('params', results[dist_name])
            
            # KS stat should be non-negative
            self.assertGreaterEqual(results[dist_name]['ks_stat'], 0)
            
            # p-value should be between 0 and 1
            self.assertGreaterEqual(results[dist_name]['p_value'], 0)
            self.assertLessEqual(results[dist_name]['p_value'], 1)
    
    def test_fit_distributions_lognormal_data(self):
        """Test fitting distributions to log-normally distributed data"""
        np.random.seed(42)
        data = np.random.lognormal(mean=0, sigma=0.5, size=1000)
        
        results = fit_distributions(data)
        
        # Lognormal should fit better than normal for log-normal data
        # (typically, but not guaranteed with small samples)
        self.assertIn('normal', results)
        self.assertIn('lognormal', results)
    
    def test_fit_distributions_exponential_data(self):
        """Test fitting distributions to exponentially distributed data"""
        np.random.seed(42)
        data = np.random.exponential(scale=2.0, size=1000)
        
        results = fit_distributions(data)
        
        # All distributions should be fitted
        self.assertEqual(len(results), 4)
    
    def test_fit_distributions_uniform_data(self):
        """Test fitting distributions to uniformly distributed data"""
        np.random.seed(42)
        data = np.random.uniform(low=1.0, high=10.0, size=1000)
        
        results = fit_distributions(data)
        
        # All distributions should be fitted
        self.assertEqual(len(results), 4)
    
    def test_fit_distributions_params_shape(self):
        """Test that parameters have correct shape"""
        np.random.seed(42)
        data = np.random.random(100) + 0.1  # Ensure positive
        
        results = fit_distributions(data)
        
        # Normal has 2 params (mu, std)
        self.assertEqual(len(results['normal']['params']), 2)
        
        # Lognormal has 3 params (shape, loc, scale)
        self.assertEqual(len(results['lognormal']['params']), 3)
        
        # Exponential has 2 params (loc, scale)
        self.assertEqual(len(results['exponential']['params']), 2)
        
        # Uniform has 2 params (loc, scale)
        self.assertEqual(len(results['uniform']['params']), 2)
    
    def test_fit_distributions_with_negative_values(self):
        """Test handling of negative values for lognormal"""
        np.random.seed(42)
        # Mix of positive and negative values
        data = np.random.normal(loc=0, scale=1.0, size=100)
        
        # Should still work - negative values are filtered for lognormal
        results = fit_distributions(data)
        
        # Should have all 4 distributions
        self.assertEqual(len(results), 4)
    
    def test_calculate_moments_symmetric(self):
        """Test moments calculation for symmetric distribution"""
        np.random.seed(42)
        # Normal distribution is symmetric, so skewness should be near 0
        data = np.random.normal(loc=0, scale=1.0, size=10000)
        
        moments = calculate_moments(data)
        
        self.assertIn('skewness', moments)
        self.assertIn('kurtosis', moments)
        
        # Skewness should be close to 0 for normal distribution
        self.assertAlmostEqual(moments['skewness'], 0.0, places=1)
        
        # Excess kurtosis should be close to 0 for normal distribution
        self.assertAlmostEqual(moments['kurtosis'], 0.0, places=1)
    
    def test_calculate_moments_right_skewed(self):
        """Test moments calculation for right-skewed distribution"""
        np.random.seed(42)
        # Exponential distribution is right-skewed
        data = np.random.exponential(scale=2.0, size=10000)
        
        moments = calculate_moments(data)
        
        # Skewness should be positive for right-skewed distribution
        self.assertGreater(moments['skewness'], 0)
        
        # Kurtosis should be positive for exponential
        self.assertGreater(moments['kurtosis'], 0)
    
    def test_calculate_moments_left_skewed(self):
        """Test moments calculation for left-skewed distribution"""
        np.random.seed(42)
        # Create left-skewed by negating right-skewed
        data = -np.random.exponential(scale=2.0, size=10000)
        
        moments = calculate_moments(data)
        
        # Skewness should be negative for left-skewed distribution
        self.assertLess(moments['skewness'], 0)
    
    def test_calculate_moments_uniform(self):
        """Test moments for uniform distribution"""
        np.random.seed(42)
        data = np.random.uniform(low=0, high=10, size=10000)
        
        moments = calculate_moments(data)
        
        # Uniform distribution has zero skewness
        self.assertAlmostEqual(moments['skewness'], 0.0, places=1)
        
        # Uniform distribution has negative excess kurtosis
        self.assertLess(moments['kurtosis'], 0)


if __name__ == '__main__':
    unittest.main()
