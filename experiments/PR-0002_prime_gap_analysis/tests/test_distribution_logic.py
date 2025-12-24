import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from distribution_tests import test_distributions, compute_practical_significance

def test_cross_band_logic_fix():
    """Verify the fix for cross-band detection logic.
    
    The issue was that lognormal_count and practical_sig_lognormal_count 
    could be satisfied by different bands.
    """
    import distribution_tests
    original_fn = distribution_tests.test_distributions_in_band
    
    # Scenario:
    # Band 1: lognormal best fit, NO practical sig
    # Band 2: lognormal best fit, NO practical sig
    # Band 3: exponential best fit, WITH practical sig favoring lognormal (should not count!)
    
    call_count = 0
    def mock_test_in_band(gaps, band_name):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return {
                'best_fit': 'normal_on_log',
                'practical_sig_favors_lognormal': False,
                'ks_ratio_exp_to_lognormal': 1.1
            }
        else:
            return {
                'best_fit': 'exponential',
                'practical_sig_favors_lognormal': True,
                'ks_ratio_exp_to_lognormal': 2.0
            }
            
    distribution_tests.test_distributions_in_band = mock_test_in_band
    
    try:
        primes = np.array([10**5 + 1, 10**5 + 2, 10**6 + 1, 10**6 + 2, 10**7 + 1, 10**7 + 2])
        results = test_distributions(primes)
        
        print(f"Interpretation: {results['interpretation']}")
        print(f"Lognormal count: {results['lognormal_count']}")
        print(f"Prac sig lognormal count: {results['practical_sig_lognormal_count']}")
        
        # Expectation: 
        # lognormal_count = 2 (Bands 1 & 2)
        # practical_sig_lognormal_count = 0 (Band 3 has sig but best_fit!=lognormal, so it's ignored)
        
        assert "Lognormal structure detected" not in results['interpretation']
        assert results['lognormal_count'] == 2
        assert results['practical_sig_lognormal_count'] == 0
        
        print("✓ test_cross_band_logic_fix passed")
        
    finally:
        distribution_tests.test_distributions_in_band = original_fn

def test_lognormal_detection_success():
    """Verify lognormal detection works when bands align."""
    import distribution_tests
    original_fn = distribution_tests.test_distributions_in_band
    
    call_count = 0
    def mock_test_in_band(gaps, band_name):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return {
                'best_fit': 'normal_on_log',
                'practical_sig_favors_lognormal': True,
                'ks_ratio_exp_to_lognormal': 2.0
            }
        else:
            return {
                'best_fit': 'exponential',
                'practical_sig_favors_lognormal': False,
                'ks_ratio_exp_to_lognormal': 0.5
            }
            
    distribution_tests.test_distributions_in_band = mock_test_in_band
    
    try:
        primes = np.array([10**5 + 1, 10**5 + 2, 10**6 + 1, 10**6 + 2, 10**7 + 1, 10**7 + 2])
        results = test_distributions(primes)
        
        print(f"Interpretation: {results['interpretation']}")
        assert "Lognormal structure detected" in results['interpretation']
        assert results['practical_sig_lognormal_count'] == 2
        
        print("✓ test_lognormal_detection_success passed")
        
    finally:
        distribution_tests.test_distributions_in_band = original_fn

def test_phantom_resonance_prevention():
    """Verify that significance is ignored if best_fit doesn't match."""
    import distribution_tests
    original_fn = distribution_tests.test_distributions_in_band
    
    # Scenario:
    # Band 1: Gamma best fit (but ratio favors lognormal vs exp) -> Should count as 0 for prac_sig_log
    # Band 2: Gamma best fit (but ratio favors lognormal vs exp) -> Should count as 0 for prac_sig_log
    # Band 3: Lognormal best fit (but ratio is neutral) -> Should count as 0 for prac_sig_log
    
    call_count = 0
    def mock_test_in_band(gaps, band_name):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return {
                'best_fit': 'gamma', # Gamma wins
                'practical_sig_favors_lognormal': True, # But lognormal beats exponential
                'ks_ratio_exp_to_lognormal': 2.0
            }
        else:
            return {
                'best_fit': 'normal_on_log',
                'practical_sig_favors_lognormal': False,
                'ks_ratio_exp_to_lognormal': 1.0
            }
            
    distribution_tests.test_distributions_in_band = mock_test_in_band
    
    try:
        primes = np.array([10**5 + 1, 10**5 + 2, 10**6 + 1, 10**6 + 2, 10**7 + 1, 10**7 + 2])
        results = test_distributions(primes)
        
        print(f"Interpretation: {results['interpretation']}")
        
        # practical_sig_lognormal_count should be 0 because in bands 1&2, best_fit != normal_on_log
        assert results['practical_sig_lognormal_count'] == 0
        assert results['lognormal_count'] == 1 # Only Band 3
        
        print("✓ test_phantom_resonance_prevention passed")
        
    finally:
        distribution_tests.test_distributions_in_band = original_fn

def test_practical_significance_safety():
    """Verify compute_practical_significance raises error on invalid threshold."""
    
    res = compute_practical_significance(2.0, threshold=1.5)
    assert res['favors_lognormal'] is True
    assert res['favors_exponential'] is False
    
    try:
        compute_practical_significance(1.0, threshold=0.5)
        print("FAIL: Should have raised ValueError for threshold=0.5")
        assert False, "Did not raise ValueError for contradictory threshold"
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")

if __name__ == "__main__":
    test_cross_band_logic_fix()
    test_lognormal_detection_success()
    test_phantom_resonance_prevention()
    test_practical_significance_safety()