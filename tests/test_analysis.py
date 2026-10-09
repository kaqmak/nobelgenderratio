"""
Test script for Nobel Gender Bias Analysis
This script tests the basic functionality with existing data
"""

import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nobelgenderbias import NobelGenderBiasAnalysis


def test_data_loading():
    """Test data loading functionality"""
    print("Testing data loading...")
    
    analysis = NobelGenderBiasAnalysis()
    
    try:
        analysis.load_data()
        print("✓ Data loading successful")
        
        # Check if data was loaded properly
        if analysis.ratio_df is not None:
            print(f"✓ Ratio data loaded: {len(analysis.ratio_df)} rows")
        
        if analysis.nobel_df is not None:
            print(f"✓ Nobel data loaded: {len(analysis.nobel_df)} rows")
            
        return True
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False

def test_data_preparation():
    """Test data preparation for modeling"""
    print("\nTesting data preparation...")
    
    analysis = NobelGenderBiasAnalysis()
    
    try:
        analysis.load_data()
        r_matrix, y_matrix, total_matrix, prize_names = analysis.prepare_stan_data()
        
        print("✓ Data matrices prepared successfully")
        print(f"  - Matrix shape: {r_matrix.shape}")
        print(f"  - Prize fields: {prize_names}")
        print(f"  - Data range: {r_matrix.min():.3f} to {r_matrix.max():.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data preparation failed: {e}")
        return False

def test_model_building():
    """Test PyMC model building"""
    print("\nTesting PyMC model building...")
    
    analysis = NobelGenderBiasAnalysis()
    
    try:
        analysis.load_data()
        r_matrix, y_matrix, total_matrix, prize_names = analysis.prepare_stan_data()
        
        # Build model
        model = analysis.build_pymc_model(r_matrix, y_matrix, total_matrix)
        
        print("✓ PyMC model built successfully")
        print(f"  - Model variables: {list(model.named_vars.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model building failed: {e}")
        return False

def test_visualization(tmp_path, monkeypatch):
    """Test basic visualization functions"""
    print("\nTesting visualization functions...")
    
    analysis = NobelGenderBiasAnalysis()
    
    try:
        analysis.load_data()
        monkeypatch.setattr(
            'nobelgenderbias.analysis.get_root_dir', lambda: tmp_path
        )
        
        # Test ratio plot
        analysis.plot_ratio_vs_year(save_path='test_ratioVsYear.pdf')
        assert (tmp_path / 'figs' / 'test_ratioVsYear.pdf').exists()
        print("✓ Ratio vs year plot created")

        fig, axes = analysis.plot_nobel_laureates_by_gender(save_path=None)
        assert len(axes) == 4
        plt.close(fig)
        print("✓ Laureates by gender plot created")
        
        return True
        
    except Exception as e:
        print(f"✗ Visualization failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("Nobel Gender Bias Analysis - Test Suite")
    print("=" * 50)
    
    tests = [
        test_data_loading,
        test_data_preparation,
        test_model_building,
        test_visualization
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("✓ All tests passed! The conversion appears to be working correctly.")
        print("\nYou can now run the full analysis with:")
        print("python run_analysis.py")
    else:
        print("✗ Some tests failed. Please check the error messages above.")
        print("\nCommon issues:")
        print("- Missing required data files")
        print("- Incorrect file formats")
        print("- Missing Python dependencies")

if __name__ == "__main__":
    run_all_tests()