#!/usr/bin/env python3
"""
🚀 Machine Learning Nifty - One-Command Launch Script
Comprehensive Stock Analysis Platform Entry Point
"""

import sys
import os
import subprocess
import time
import webbrowser
from pathlib import Path

def print_banner():
    """Print welcome banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║           🚀 MACHINE LEARNING NIFTY - AI STOCK ANALYSIS             ║
    ║                                                                      ║
    ║  📈 AI-Powered Buy/Sell/Hold Recommendations                        ║
    ║  🧠 Explainable AI with 20+ Technical Indicators                    ║
    ║  🌍 Multi-Market Support (US, India, Crypto, Global)                ║
    ║  ⚡ Real-Time Analysis with Risk Assessment                          ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    return True

def check_dependencies():
    """Check and install required dependencies"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'streamlit>=1.28.0',
        'yfinance>=0.2.18',
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'plotly>=5.15.0',
        'requests>=2.31.0'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        package_name = package.split('>=')[0]
        try:
            __import__(package_name.replace('-', '_'))
            print(f"✅ {package_name} - OK")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package_name} - Missing")
    
    if missing_packages:
        print(f"\n📦 Installing {len(missing_packages)} missing packages...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install'
            ] + missing_packages)
            print("✅ All dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Please install manually:")
            print("pip install -r requirements.txt")
            return False
    else:
        print("✅ All dependencies are installed!")
    
    return True

def create_requirements_if_missing():
    """Create requirements.txt if it doesn't exist"""
    req_path = Path("requirements.txt")
    if not req_path.exists():
        print("📝 Creating requirements.txt...")
        requirements_content = """torch>=2.0.0
transformers>=4.35.0
yfinance>=0.2.18
pandas>=2.0.0
numpy>=1.24.0
requests>=2.31.0
streamlit>=1.28.0
plotly>=5.15.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
fastapi>=0.104.0
uvicorn>=0.24.0"""
        
        with open(req_path, 'w') as f:
            f.write(requirements_content)
        print("✅ requirements.txt created!")

def launch_main_app():
    """Launch the main Streamlit application"""
    print("\n🚀 Launching AI Stock Analysis Platform...")
    print("📊 Starting Streamlit dashboard...")
    
    # Check if main_app.py exists
    main_app_path = Path("main_app.py")
    if not main_app_path.exists():
        print("❌ main_app.py not found. Please ensure the file exists.")
        return False
    
    try:
        # Set environment variables
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()
        
        # Launch Streamlit
        cmd = [sys.executable, '-m', 'streamlit', 'run', 'main_app.py']
        print(f"🎯 Running command: {' '.join(cmd)}")
        
        # Open browser after a delay
        def open_browser():
            time.sleep(3)
            webbrowser.open('http://localhost:8501')
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Run Streamlit
        subprocess.run(cmd, env=env)
        
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        return False

def show_usage_info():
    """Show usage information"""
    info = """
    📖 USAGE INFORMATION:
    
    🎯 How to Use:
    1. Select a stock using the sidebar (search, popular stocks, or custom symbol)
    2. Configure analysis parameters (period, forecast days, risk tolerance)
    3. View AI recommendation with detailed reasoning
    4. Analyze technical indicators and charts
    5. Review risk assessment and price targets
    
    🔍 Supported Symbols:
    • US Stocks: AAPL, MSFT, GOOGL, TSLA, etc.
    • Indian Stocks: RELIANCE.NS, TCS.NS, HDFCBANK.NS, etc.
    • Crypto: BTC-USD, ETH-USD, BNB-USD, etc.
    • Indices: ^GSPC, ^NSEI, ^BSESN, etc.
    
    🧠 AI Features:
    • Buy/Sell/Hold recommendations with confidence scores
    • 20+ technical indicators analysis
    • Risk assessment and position sizing
    • Explainable AI reasoning for every decision
    
    📊 The application will open automatically in your browser at:
    http://localhost:8501
    
    Press Ctrl+C to stop the application.
    """
    print(info)

def main():
    """Main execution function"""
    print_banner()
    
    # System checks
    if not check_python_version():
        sys.exit(1)
    
    # Create requirements.txt if missing
    create_requirements_if_missing()
    
    # Check and install dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Show usage information
    show_usage_info()
    
    # Launch the application
    success = launch_main_app()
    
    if success:
        print("\n✅ Application session completed successfully!")
    else:
        print("\n❌ Application encountered an error")
        sys.exit(1)

if __name__ == "__main__":
    main()