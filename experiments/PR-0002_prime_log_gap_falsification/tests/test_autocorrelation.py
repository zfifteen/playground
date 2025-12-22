import unittest
import numpy as np
from src.autocorrelation import compute_autocorrelation, perform_ljung_box


class TestAutocorrelation(unittest.TestCase):
    
    def test_compute_autocorrelation_simple(self):
        """Test ACF/PACF computation with simple data"""
        np.random.seed(42)
        data = np.random.random(1000)
        
        result = compute_autocorrelation(data, nlags=20)
        
        # Should have acf and pacf keys
        self.assertIn('acf', result)
        self.assertIn('pacf', result)
        
        # ACF should have nlags+1 values (includes lag 0)
        self.assertEqual(len(result['acf']), 21)
        
        # PACF should have nlags+1 values
        self.assertEqual(len(result['pacf']), 21)
        
        # ACF at lag 0 should be 1
        self.assertAlmostEqual(result['acf'][0], 1.0, places=10)
    
    def test_compute_autocorrelation_different_nlags(self):
        """Test with different nlags values"""
        np.random.seed(42)
        data = np.random.random(500)
        
        result = compute_autocorrelation(data, nlags=10)
        
        self.assertEqual(len(result['acf']), 11)
        self.assertEqual(len(result['pacf']), 11)
    
    def test_compute_autocorrelation_white_noise(self):
        """Test with white noise - should have low autocorrelation"""
        np.random.seed(42)
        data = np.random.normal(0, 1, 10000)
        
        result = compute_autocorrelation(data, nlags=20)
        
        # ACF at lag 0 should be 1
        self.assertAlmostEqual(result['acf'][0], 1.0, places=10)
        
        # ACF at other lags should be close to 0 for white noise
        # (allowing some statistical variation)
        for lag in range(1, 21):
            self.assertLess(abs(result['acf'][lag]), 0.1)
    
    def test_compute_autocorrelation_ar_process(self):
        """Test with AR(1) process - should have decaying autocorrelation"""
        np.random.seed(42)
        # Generate AR(1) process using statsmodels
        from statsmodels.tsa.arima_process import ArmaProcess
        n = 10000
        phi = 0.7
        ar = np.array([1, -phi])  # AR(1) coefficient
        ma = np.array([1])  # No MA component
        arma_process = ArmaProcess(ar, ma)
        data = arma_process.generate_sample(nsample=n)
        
        result = compute_autocorrelation(data, nlags=20)
        
        # ACF should decay exponentially for AR(1)
        # ACF[1] should be approximately phi
        self.assertAlmostEqual(result['acf'][1], phi, places=1)
        
        # ACF should be decreasing (in absolute value)
        for i in range(1, 10):
            self.assertGreater(abs(result['acf'][i]), abs(result['acf'][i+1]))
    
    def test_compute_autocorrelation_large_data(self):
        """Test with large dataset - should subsample to 1M points"""
        np.random.seed(42)
        # Create data larger than 1M
        data = np.random.random(2000000)
        
        result = compute_autocorrelation(data, nlags=20)
        
        # Should still work and return results
        self.assertEqual(len(result['acf']), 21)
        self.assertEqual(len(result['pacf']), 21)
    
    def test_perform_ljung_box_simple(self):
        """Test Ljung-Box test with simple data"""
        np.random.seed(42)
        data = np.random.random(1000)
        
        result = perform_ljung_box(data, lags=20)
        
        # Should return a DataFrame
        import pandas as pd
        self.assertIsInstance(result, pd.DataFrame)
        
        # Should have columns for test statistic and p-value
        self.assertIn('lb_stat', result.columns)
        self.assertIn('lb_pvalue', result.columns)
    
    def test_perform_ljung_box_white_noise(self):
        """Test Ljung-Box on white noise - should not reject null hypothesis"""
        np.random.seed(42)
        data = np.random.normal(0, 1, 10000)
        
        result = perform_ljung_box(data, lags=20)
        
        # p-value should be relatively high for white noise
        # (indicating we don't reject the null hypothesis of no autocorrelation)
        self.assertGreater(result['lb_pvalue'].values[0], 0.01)
    
    def test_perform_ljung_box_ar_process(self):
        """Test Ljung-Box on AR process - should reject null hypothesis"""
        np.random.seed(42)
        # Generate AR(1) process with strong autocorrelation using statsmodels
        from statsmodels.tsa.arima_process import ArmaProcess
        n = 10000
        phi = 0.9
        ar = np.array([1, -phi])  # AR(1) coefficient
        ma = np.array([1])  # No MA component
        arma_process = ArmaProcess(ar, ma)
        data = arma_process.generate_sample(nsample=n)
        
        result = perform_ljung_box(data, lags=20)
        
        # p-value should be very low for AR process
        # (indicating we reject the null hypothesis - there is autocorrelation)
        self.assertLess(result['lb_pvalue'].values[0], 0.01)
    
    def test_perform_ljung_box_large_data(self):
        """Test Ljung-Box with large dataset - should subsample"""
        np.random.seed(42)
        data = np.random.random(2000000)
        
        result = perform_ljung_box(data, lags=20)
        
        # Should still work
        import pandas as pd
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_perform_ljung_box_different_lags(self):
        """Test Ljung-Box with different lag values"""
        np.random.seed(42)
        data = np.random.random(1000)
        
        result = perform_ljung_box(data, lags=10)
        
        # Should work with different lag values
        import pandas as pd
        self.assertIsInstance(result, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
