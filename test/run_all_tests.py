"""
Run All XPlaneGym Tests

This script runs all test scripts in sequence, checking if all functional interfaces of XPlaneGym are working properly.
"""

import sys
import os
import time
from importlib import import_module

# Add project path to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_all_tests():
    """Run all test scripts"""
    
    # List of test modules
    test_modules = [
        "test_connection",
        "test_env_basic",
        "test_discrete_env",
        "test_continuous_env"
    ]
    
    results = {}
    
    print("=" * 60)
    print("XPlaneGym Functionality Tests")
    print("=" * 60)
    
    # Run each test module
    for test_module in test_modules:
        print("\n" + "=" * 60)
        print(f"Running test: {test_module}")
        print("-" * 60)
        
        try:
            # Import test module
            module = import_module(test_module)
            
            # Get short module name (without prefix)
            module_short_name = test_module.split(".")[-1]
            
            # Find and run test functions
            test_functions = [f for f in dir(module) if f.startswith("test_") and callable(getattr(module, f))]
            
            if not test_functions:
                print(f"Warning: No test functions found in module {module_short_name}")
                results[module_short_name] = False
                continue
            
            # Run each test function
            module_result = True
            for test_func_name in test_functions:
                test_func = getattr(module, test_func_name)
                print(f"\nRunning test function: {test_func_name}")
                try:
                    # Call test function
                    func_result = test_func()
                    if not func_result:
                        module_result = False
                        print(f"✗ Test function {test_func_name} failed")
                    else:
                        print(f"✓ Test function {test_func_name} passed")
                except Exception as e:
                    module_result = False
                    print(f"✗ Test function {test_func_name} encountered an exception: {e}")
            
            results[module_short_name] = module_result
            
        except Exception as e:
            module_short_name = test_module.split(".")[-1]
            print(f"✗ Test module {module_short_name} failed to load: {e}")
            results[module_short_name] = False
    
    # Show summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("-" * 60)
    
    all_passed = True
    for module, result in results.items():
        status = "✓ Passed" if result else "✗ Failed"
        print(f"{module}: {status}")
        all_passed = all_passed and result
    
    print("\n" + "=" * 60)
    if all_passed:
        print("Congratulations! All tests passed")
    else:
        print("Warning: Some tests did not pass, please check the logs above")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    run_all_tests() 