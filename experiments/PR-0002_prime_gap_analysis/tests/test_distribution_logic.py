
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from distribution_tests import test_distributions

def test_cross_band_logic_fix():
    """Verify the fix for cross-band detection logic.
    
    The issue was that lognormal_count and practical_sig_lognormal_count 
    could be satisfied by different bands.
    """
    
    # We need to mock band_results or provide primes that produce specific band_results.
    # It's easier to mock the function that test_distributions calls, 
    # but test_distributions is a self-contained function that calls test_distributions_in_band.
    
    # Let's monkeypatch test_distributions_in_band to return what we want.
    import distribution_tests
    
    original_fn = distribution_tests.test_distributions_in_band
    
    # Scenario:
    # Band 1: lognormal best fit, NO practical sig
    # Band 2: lognormal best fit, NO practical sig
    # Band 3: exponential best fit, WITH practical sig favoring lognormal
    
    call_count = 0
    def mock_test_in_band(gaps, band_name):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                'best_fit': 'normal_on_log',
                'practical_sig_favors_lognormal': False,
                'ks_ratio_exp_to_lognormal': 1.1
            }
        elif call_count == 2:
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
        # Dummy primes, enough to trigger 3 bands
        # Bands are 1e5-1e6, 1e6-1e7, 1e7-1e8
        primes = np.array([10**5 + 1, 10**5 + 2, 10**6 + 1, 10**6 + 2, 10**7 + 1, 10**7 + 2])
        results = test_distributions(primes)
        
        print(f"Interpretation: {results['interpretation']}")
        print(f"Lognormal count: {results['lognormal_count']}")
        print(f"Prac sig lognormal count: {results['practical_sig_lognormal_count']}")
        print(f"Aligned count: {results['lognormal_with_practical_sig_count']}")
        
        # OLD LOGIC would have said "Lognormal structure detected" 
        # because lognormal_count == 2 and practical_sig_lognormal_count == 1
        
        # NEW LOGIC should say "Lognormal detected but practical significance not established"
        # because lognormal_with_practical_sig_count == 0
        
        assert "Lognormal structure detected" not in results['interpretation']
        assert results['lognormal_with_practical_sig_count'] == 0
        assert results['lognormal_count'] == 2
        assert results['practical_sig_lognormal_count'] == 1
        
        print("✓ test_cross_band_logic_fix passed")
        
    finally:
        distribution_tests.test_distributions_in_band = original_fn

def test_lognormal_detection_success():
    """Verify lognormal detection works when bands align."""
    import distribution_tests
    original_fn = distribution_tests.test_distributions_in_band
    
    # Scenario:
    # Band 1: lognormal best fit, WITH practical sig
    # Band 2: lognormal best fit, WITH practical sig
    # Band 3: exponential best fit
    
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
        assert results['lognormal_with_practical_sig_count'] == 2
        
        print("✓ test_lognormal_detection_success passed")
        
    finally:
        distribution_tests.test_distributions_in_band = original_fn

def test_practical_significance_safety():
    """Verify compute_practical_significance raises error on invalid threshold."""
    from distribution_tests import compute_practical_significance
    
    # Valid threshold > 1.0
    res = compute_practical_significance(2.0, threshold=1.5)
    assert res['favors_lognormal'] is True
    assert res['favors_exponential'] is False
    
    # Invalid threshold < 1.0 allowing overlap (e.g. ratio=0.8, thresh=0.9)
    # 0.8 > 0.9 is False. 0.8 < 1/0.9 (~1.11) is True.
    # We need a case where both are true.
    # ratio > thresh AND ratio < 1/thresh
    # Try thresh=0.5. 1/thresh=2.0.
    # ratio=1.0. 1.0 > 0.5 (True). 1.0 < 2.0 (True).
    try:
        compute_practical_significance(1.0, threshold=0.5)
        print("FAIL: Should have raised ValueError for threshold=0.5")
        assert False, "Did not raise ValueError for contradictory threshold"
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")

if __name__ == "__main__":
    test_cross_band_logic_fix()
    test_lognormal_detection_success()
    test_practical_significance_safety()
