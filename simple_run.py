"""
Simple System Runner - No Unicode Issues
"""

import os
import sys
import subprocess

# Fix Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src'))

def test_system():
    """Test if system components work"""
    print("TESTING SYSTEM COMPONENTS")
    print("=" * 30)
    
    # Test 1: Model imports
    try:
        from src.models.unified_transformer import UnifiedMultimodalTransformer, ModelConfig
        print("SUCCESS: Model imports working")
        
        # Test model creation
        config = ModelConfig()
        model = UnifiedMultimodalTransformer(config)
        print(f"SUCCESS: Model created ({sum(p.numel() for p in model.parameters()):,} parameters)")
        
    except Exception as e:
        print(f"FAILED: Model test - {e}")
        return False
    
    # Test 2: Data collection
    try:
        import yfinance as yf
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="2d")
        
        if not data.empty:
            print(f"SUCCESS: Yahoo Finance working ({len(data)} records)")
        else:
            print("FAILED: Yahoo Finance - No data")
            return False
            
    except Exception as e:
        print(f"FAILED: Data collection - {e}")
        return False
    
    print("\nALL TESTS PASSED!")
    return True

def run_api():
    """Run API server"""
    print("\nStarting API server...")
    
    try:
        # Set environment
        env = os.environ.copy()
        env['PYTHONPATH'] = current_dir
        
        # Run API
        api_file = os.path.join(current_dir, 'api', 'main.py')
        subprocess.run([sys.executable, api_file], env=env, cwd=current_dir)
        
    except KeyboardInterrupt:
        print("API server stopped")
    except Exception as e:
        print(f"API error: {e}")

def run_dashboard():
    """Run dashboard"""
    print("\nStarting dashboard...")
    
    try:
        # Set environment
        env = os.environ.copy()
        env['PYTHONPATH'] = current_dir
        
        # Run dashboard
        dashboard_file = os.path.join(current_dir, 'frontend', 'dashboard.py')
        subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_file], env=env)
        
    except KeyboardInterrupt:
        print("Dashboard stopped")
    except Exception as e:
        print(f"Dashboard error: {e}")

def main():
    """Main function"""
    print("UNIFIED MULTIMODAL TRANSFORMER")
    print("Simple System Runner")
    print("=" * 40)
    
    # Test system first
    if not test_system():
        print("\nSystem test failed. Please check the errors above.")
        return
    
    print("\nWhat would you like to run?")
    print("1. Test model only")
    print("2. API server (http://localhost:8000)")
    print("3. Dashboard (http://localhost:8501)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        print("\nModel test completed successfully!")
        print("The system is ready to run.")
        
    elif choice == "2":
        print("\nStarting API server...")
        print("API will be available at: http://localhost:8000")
        print("Press Ctrl+C to stop")
        run_api()
        
    elif choice == "3":
        print("\nStarting dashboard...")
        print("Dashboard will be available at: http://localhost:8501")
        print("Press Ctrl+C to stop")
        run_dashboard()
        
    else:
        print("Invalid choice. Please run again and choose 1, 2, or 3.")

if __name__ == "__main__":
    main()