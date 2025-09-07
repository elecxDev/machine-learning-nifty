"""
Full Stack Deployment Script
Runs the complete system: Training → API → Dashboard
"""

import subprocess
import sys
import os
import time
import threading
from pathlib import Path

def run_command(command, cwd=None, description=""):
    """Run a command and handle output"""
    print(f"\n{'='*50}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {command}")
    print(f"{'='*50}")
    
    try:
        if cwd:
            result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True)
        else:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
            if result.stdout:
                print("OUTPUT:", result.stdout[:500])
        else:
            print(f"❌ FAILED: {description}")
            print("ERROR:", result.stderr[:500])
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")
        return False

def install_requirements():
    """Install all required packages"""
    print("📦 Installing requirements...")
    
    packages = [
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "yfinance>=0.2.18",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "streamlit>=1.28.0",
        "plotly>=5.15.0",
        "scikit-learn>=1.3.0"
    ]
    
    for package in packages:
        success = run_command(f"pip install {package}", description=f"Installing {package}")
        if not success:
            print(f"⚠️ Warning: Failed to install {package}")
    
    print("✅ Requirements installation completed")

def train_model():
    """Train the model with historical data"""
    print("🤖 Starting model training...")
    
    # Check if model already exists
    if os.path.exists("models/unified_transformer_trained.pt"):
        response = input("Model already exists. Retrain? (y/N): ")
        if response.lower() != 'y':
            print("Using existing model...")
            return True
    
    # Run training script
    success = run_command(
        "python scripts/train_full_model.py",
        description="Training multimodal transformer"
    )
    
    if success:
        print("✅ Model training completed successfully")
        return True
    else:
        print("❌ Model training failed")
        return False

def start_api_server():
    """Start the FastAPI server"""
    print("🚀 Starting API server...")
    
    def run_api():
        os.chdir("api")
        os.system("python main.py")
    
    # Start API in background thread
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    
    # Wait for API to start
    print("⏳ Waiting for API to start...")
    time.sleep(10)
    
    # Test API health
    import requests
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running")
            return True
        else:
            print("❌ API server health check failed")
            return False
    except:
        print("❌ API server is not responding")
        return False

def start_dashboard():
    """Start the Streamlit dashboard"""
    print("📊 Starting dashboard...")
    
    # Run dashboard
    success = run_command(
        "streamlit run frontend/dashboard.py --server.port 8501",
        description="Starting Streamlit dashboard"
    )
    
    return success

def create_demo_data():
    """Create demo data for testing"""
    print("📊 Creating demo data...")
    
    demo_script = """
import yfinance as yf
import pandas as pd
import pickle
import os

# Create demo data
symbols = ['AAPL', 'GOOGL', 'RELIANCE.NS']
data = {}

for symbol in symbols:
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period='1y')
        data[symbol] = df
        print(f"Downloaded {symbol}: {len(df)} records")
    except Exception as e:
        print(f"Error downloading {symbol}: {e}")

# Save demo data
os.makedirs('data/demo', exist_ok=True)
with open('data/demo/sample_data.pkl', 'wb') as f:
    pickle.dump(data, f)

print("Demo data created successfully")
"""
    
    with open("create_demo.py", "w") as f:
        f.write(demo_script)
    
    success = run_command("python create_demo.py", description="Creating demo data")
    
    # Cleanup
    if os.path.exists("create_demo.py"):
        os.remove("create_demo.py")
    
    return success

def main():
    """Main deployment pipeline"""
    
    print("🚀 UNIFIED MULTIMODAL TRANSFORMER - FULL STACK DEPLOYMENT")
    print("=" * 70)
    
    # Get current directory
    project_root = Path.cwd()
    print(f"Project root: {project_root}")
    
    # Step 1: Install requirements
    print("\n🔧 STEP 1: ENVIRONMENT SETUP")
    install_requirements()
    
    # Step 2: Create demo data
    print("\n📊 STEP 2: DEMO DATA CREATION")
    create_demo_data()
    
    # Step 3: Train model (optional - can use pre-trained)
    print("\n🤖 STEP 3: MODEL TRAINING")
    
    train_choice = input("Train new model? (y/N): ")
    if train_choice.lower() == 'y':
        model_success = train_model()
        if not model_success:
            print("❌ Training failed. Using demo mode...")
    else:
        print("ℹ️ Skipping training. Using demo mode...")
    
    # Step 4: Start API server
    print("\n🚀 STEP 4: API SERVER")
    api_success = start_api_server()
    
    if not api_success:
        print("❌ API server failed to start")
        return False
    
    # Step 5: Start dashboard
    print("\n📊 STEP 5: DASHBOARD")
    print("Starting Streamlit dashboard...")
    print("Dashboard will open in your browser at: http://localhost:8501")
    
    # Instructions for user
    print("\n" + "=" * 70)
    print("🎉 DEPLOYMENT COMPLETE!")
    print("=" * 70)
    
    print("\n📋 SYSTEM STATUS:")
    print("✅ API Server: http://localhost:8000")
    print("✅ Dashboard: http://localhost:8501")
    print("✅ Model: Loaded and ready")
    
    print("\n🔗 QUICK LINKS:")
    print("• API Documentation: http://localhost:8000/docs")
    print("• Health Check: http://localhost:8000/health")
    print("• Dashboard: http://localhost:8501")
    
    print("\n📖 USAGE:")
    print("1. Open dashboard in browser")
    print("2. Select a stock symbol (AAPL, GOOGL, etc.)")
    print("3. View AI predictions and explanations")
    print("4. Explore cross-market capabilities")
    
    print("\n🛠️ DEVELOPMENT:")
    print("• API code: api/main.py")
    print("• Dashboard code: frontend/dashboard.py")
    print("• Model code: src/models/")
    print("• Training: scripts/train_full_model.py")
    
    # Start dashboard
    try:
        start_dashboard()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        return True
    
    return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✅ Full stack deployment completed successfully!")
    else:
        print("\n❌ Deployment failed. Check the logs above.")
    
    print("\n🔄 To restart:")
    print("python scripts/run_full_stack.py")