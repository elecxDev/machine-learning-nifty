"""
Quick Test - Verify everything works without training
"""

import os
import sys

# Fix Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src'))

def test_model():
    """Test model architecture"""
    try:
        from src.models.unified_transformer import UnifiedMultimodalTransformer, ModelConfig
        
        print("✓ Model imports successful")
        
        # Create model
        config = ModelConfig()
        model = UnifiedMultimodalTransformer(config)
        
        print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass
        import torch
        batch_size = 2
        seq_len = 60
        
        price_data = torch.randn(batch_size, seq_len, config.price_features)
        macro_data = torch.randn(batch_size, seq_len, config.macro_features)
        text_data = torch.randn(batch_size, seq_len, config.text_features)
        
        with torch.no_grad():
            outputs = model(price_data, macro_data, text_data)
        
        print(f"✓ Forward pass successful")
        print(f"  Forecast shape: {outputs['forecast'].shape}")
        print(f"  Anomaly shape: {outputs['anomaly_score'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def test_data_collection():
    """Test data collection"""
    try:
        import yfinance as yf
        
        # Test Yahoo Finance
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="5d")
        
        if not data.empty:
            print(f"✓ Yahoo Finance working: {len(data)} records for AAPL")
        else:
            print("✗ Yahoo Finance: No data")
            return False
        
        # Test World Bank API
        import requests
        url = "http://api.worldbank.org/v2/country/USA/indicator/NY.GDP.MKTP.CD?format=json&date=2023"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            print("✓ World Bank API working")
        else:
            print("✗ World Bank API failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Data collection test failed: {e}")
        return False

def test_transformers():
    """Test FinBERT"""
    try:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        
        # Test tokenization
        text = "The stock market is performing well"
        tokens = tokenizer(text, return_tensors='pt')
        
        print(f"✓ FinBERT working: {tokens['input_ids'].shape[1]} tokens")
        return True
        
    except Exception as e:
        print(f"✗ FinBERT test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 QUICK SYSTEM TEST")
    print("=" * 30)
    
    tests = [
        ("Model Architecture", test_model),
        ("Data Collection", test_data_collection), 
        ("FinBERT Integration", test_transformers)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔧 Testing {test_name}...")
        result = test_func()
        results.append(result)
    
    print("\n" + "=" * 30)
    print("📊 TEST RESULTS")
    print("=" * 30)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    if all(results):
        print("\n🎉 ALL TESTS PASSED!")
        print("System is ready to run")
        
        print("\nNext steps:")
        print("1. Run API: python run_system.py (choose option 2)")
        print("2. Run Dashboard: python run_system.py (choose option 3)")
        print("3. Run Both: python run_system.py (choose option 4)")
        
    else:
        print("\n⚠️ Some tests failed")
        print("Check the errors above")

if __name__ == "__main__":
    main()