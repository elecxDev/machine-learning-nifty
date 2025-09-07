"""
Fixed System Runner - Handles Python path issues
"""

import os
import sys
import subprocess
import time
import threading

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def install_packages():
    """Install required packages"""
    packages = [
        "torch", "transformers", "yfinance", "pandas", "numpy", 
        "requests", "fastapi", "uvicorn", "streamlit", "plotly"
    ]
    
    for package in packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
            print(f"✓ {package}")
        except:
            print(f"✗ {package}")

def test_imports():
    """Test if all imports work"""
    try:
        # Test model imports
        sys.path.append(os.path.join(current_dir, 'src'))
        from src.models.unified_transformer import UnifiedMultimodalTransformer, ModelConfig
        print("✓ Model imports working")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def run_api():
    """Run API server"""
    try:
        # Change to API directory and run
        api_dir = os.path.join(current_dir, 'api')
        os.chdir(api_dir)
        
        # Set Python path
        env = os.environ.copy()
        env['PYTHONPATH'] = current_dir
        
        subprocess.run([sys.executable, "main.py"], env=env)
    except Exception as e:
        print(f"API error: {e}")

def run_dashboard():
    """Run Streamlit dashboard"""
    try:
        dashboard_file = os.path.join(current_dir, 'frontend', 'dashboard.py')
        
        # Set Python path
        env = os.environ.copy()
        env['PYTHONPATH'] = current_dir
        
        subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_file], env=env)
    except Exception as e:
        print(f"Dashboard error: {e}")

def main():
    """Main runner"""
    print("🚀 UNIFIED MULTIMODAL TRANSFORMER - FIXED RUNNER")
    print("=" * 50)
    
    # Install packages
    print("📦 Installing packages...")
    install_packages()
    
    # Test imports
    print("\n🔧 Testing imports...")
    if not test_imports():
        print("❌ Import test failed")
        return
    
    print("✅ All imports working!")
    
    # Ask user what to run
    print("\nWhat would you like to run?")
    print("1. Test model only")
    print("2. API server only") 
    print("3. Dashboard only")
    print("4. Both API + Dashboard")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        # Test model
        print("\n🤖 Testing model...")
        os.system(f'python "{os.path.join(current_dir, "test_model.py")}"')
        
    elif choice == "2":
        # Run API only
        print("\n🚀 Starting API server...")
        print("API will be available at: http://localhost:8000")
        run_api()
        
    elif choice == "3":
        # Run dashboard only
        print("\n📊 Starting dashboard...")
        print("Dashboard will be available at: http://localhost:8501")
        run_dashboard()
        
    elif choice == "4":
        # Run both
        print("\n🚀 Starting API server...")
        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()
        
        print("⏳ Waiting for API to start...")
        time.sleep(5)
        
        print("📊 Starting dashboard...")
        print("API: http://localhost:8000")
        print("Dashboard: http://localhost:8501")
        run_dashboard()
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()