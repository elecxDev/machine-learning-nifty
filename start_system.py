"""
Start Sophisticated System - No Unicode Issues
"""

import subprocess
import sys
import os
import time
import threading

def run_api():
    """Run API server"""
    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()
        
        os.chdir("api")
        subprocess.run([sys.executable, "main.py"], env=env)
        
    except KeyboardInterrupt:
        print("API server stopped")

def run_dashboard():
    """Run dashboard"""
    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()
        
        subprocess.run([sys.executable, "-m", "streamlit", "run", "frontend/dashboard.py"], env=env)
        
    except KeyboardInterrupt:
        print("Dashboard stopped")

def check_system():
    """Check system components"""
    print("CHECKING SOPHISTICATED SYSTEM")
    print("=" * 40)
    
    try:
        sys.path.append('src')
        from src.models.unified_transformer import UnifiedMultimodalTransformer, ModelConfig
        
        config = ModelConfig()
        model = UnifiedMultimodalTransformer(config)
        params = sum(p.numel() for p in model.parameters())
        
        print(f"SUCCESS: Multimodal Transformer ({params:,} parameters)")
        print("  - Cross-modal attention")
        print("  - Multi-task learning")
        print("  - FinBERT integration")
        
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def main():
    """Main runner"""
    print("UNIFIED MULTIMODAL TRANSFORMER")
    print("Research-Grade Financial Forecasting")
    print("=" * 50)
    
    if not check_system():
        print("System check failed")
        return
    
    print("\nSYSTEM FEATURES:")
    print("- Cross-market analysis (US, India, Brazil)")
    print("- Multimodal AI (Price + Macro + Text)")
    print("- Explainable predictions")
    print("- Real-time forecasting")
    print("- Risk assessment")
    
    print("\nWHAT TO RUN?")
    print("1. API Server (Backend)")
    print("2. Dashboard (Frontend)")
    print("3. Both (Complete System)")
    
    choice = input("\nChoice (1-3): ").strip()
    
    if choice == "1":
        print("\nStarting API Server...")
        print("URL: http://localhost:8000")
        run_api()
        
    elif choice == "2":
        print("\nStarting Dashboard...")
        print("URL: http://localhost:8501")
        run_dashboard()
        
    elif choice == "3":
        print("\nStarting Complete System...")
        print("API: http://localhost:8000")
        print("Dashboard: http://localhost:8501")
        
        # Start API in background
        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()
        
        print("Waiting for API...")
        time.sleep(5)
        
        print("Starting Dashboard...")
        run_dashboard()
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()