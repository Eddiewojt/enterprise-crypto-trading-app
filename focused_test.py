#!/usr/bin/env python3
"""
Focused Testing for MongoDB ObjectId Serialization Issues
Tests the specific endpoints that are failing according to test_result.md
"""

import requests
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv('/app/frontend/.env')
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing backend at: {API_BASE_URL}")

class FocusedTester:
    def __init__(self):
        self.base_url = API_BASE_URL
        self.errors = []
        
    def log_error(self, test_name, error):
        """Log test errors"""
        error_msg = f"❌ {test_name}: {str(error)}"
        self.errors.append(error_msg)
        print(error_msg)
        
    def log_success(self, test_name, details=""):
        """Log test success"""
        success_msg = f"✅ {test_name}"
        if details:
            success_msg += f": {details}"
        print(success_msg)
        
    def test_portfolio_trade_creation(self):
        """Test portfolio trade creation to see if it creates problematic data"""
        try:
            print("\n💼 Testing Portfolio Trade Creation...")
            
            # Test paper trade execution
            trade_request = {
                "symbol": "DOGEUSDT",
                "side": "BUY",
                "quantity": 1000.0,
                "price": None  # Use current market price
            }
            
            trade_response = requests.post(f"{self.base_url}/portfolio/trade", 
                                         json=trade_request, 
                                         timeout=15)
            
            if trade_response.status_code == 200:
                trade_data = trade_response.json()
                self.log_success("Portfolio Trade Creation", 
                               f"Trade executed: {trade_data.get('side', 'N/A')} {trade_data.get('quantity', 'N/A')} {trade_data.get('symbol', 'N/A')}")
                print(f"Response data: {json.dumps(trade_data, indent=2)}")
                return True
            else:
                self.log_error("Portfolio Trade Creation", f"HTTP {trade_response.status_code}: {trade_response.text}")
                return False
                
        except Exception as e:
            self.log_error("Portfolio Trade Creation", e)
            return False
            
    def test_portfolio_trades_endpoint(self):
        """Test GET /api/portfolio/trades - reported to have MongoDB ObjectId serialization issues"""
        try:
            print("\n📊 Testing Portfolio Trades Endpoint (GET /api/portfolio/trades)...")
            
            # Test trade history retrieval
            history_response = requests.get(f"{self.base_url}/portfolio/trades", timeout=15)
            
            print(f"Response status: {history_response.status_code}")
            print(f"Response headers: {dict(history_response.headers)}")
            
            if history_response.status_code == 200:
                try:
                    history_data = history_response.json()
                    self.log_success("Portfolio Trades Endpoint", f"Retrieved data successfully")
                    print(f"Response data: {json.dumps(history_data, indent=2)}")
                    return True
                except json.JSONDecodeError as e:
                    self.log_error("Portfolio Trades Endpoint", f"JSON decode error: {e}")
                    print(f"Raw response: {history_response.text}")
                    return False
            elif history_response.status_code == 500:
                self.log_error("Portfolio Trades Endpoint", f"500 Internal Server Error - likely MongoDB ObjectId serialization issue")
                print(f"Error response: {history_response.text}")
                return False
            else:
                self.log_error("Portfolio Trades Endpoint", f"HTTP {history_response.status_code}: {history_response.text}")
                return False
                
        except Exception as e:
            self.log_error("Portfolio Trades Endpoint", e)
            return False
            
    def test_backtest_creation(self):
        """Test backtest creation to see if it creates problematic data"""
        try:
            print("\n🔬 Testing Backtest Creation...")
            
            # Run a simple backtest
            backtest_request = {
                "symbol": "doge",
                "timeframe": "15m",
                "start_date": "2024-12-01",
                "end_date": "2024-12-31",
                "strategy": "rsi",
                "initial_capital": 5000.0
            }
            
            response = requests.post(f"{self.base_url}/backtest", 
                                   json=backtest_request, 
                                   timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                self.log_success("Backtest Creation", f"Backtest completed successfully")
                print(f"Backtest result summary: Symbol={data.get('symbol')}, Return={data.get('total_return_percentage', 'N/A')}%, Trades={data.get('total_trades', 'N/A')}")
                return True
            else:
                self.log_error("Backtest Creation", f"HTTP {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            self.log_error("Backtest Creation", e)
            return False
            
    def test_backtest_results_endpoint(self):
        """Test GET /api/backtest/results - reported to have MongoDB ObjectId serialization issues"""
        try:
            print("\n📈 Testing Backtest Results Endpoint (GET /api/backtest/results)...")
            
            # Test results retrieval
            results_response = requests.get(f"{self.base_url}/backtest/results", timeout=15)
            
            print(f"Response status: {results_response.status_code}")
            print(f"Response headers: {dict(results_response.headers)}")
            
            if results_response.status_code == 200:
                try:
                    results_data = results_response.json()
                    self.log_success("Backtest Results Endpoint", f"Retrieved {len(results_data) if isinstance(results_data, list) else 'N/A'} results")
                    print(f"Response data: {json.dumps(results_data, indent=2)}")
                    return True
                except json.JSONDecodeError as e:
                    self.log_error("Backtest Results Endpoint", f"JSON decode error: {e}")
                    print(f"Raw response: {results_response.text}")
                    return False
            elif results_response.status_code == 500:
                self.log_error("Backtest Results Endpoint", f"500 Internal Server Error - likely MongoDB ObjectId serialization issue")
                print(f"Error response: {results_response.text}")
                return False
            else:
                self.log_error("Backtest Results Endpoint", f"HTTP {results_response.status_code}: {results_response.text}")
                return False
                
        except Exception as e:
            self.log_error("Backtest Results Endpoint", e)
            return False
            
    def test_portfolio_endpoint(self):
        """Test basic portfolio endpoint to see if it works"""
        try:
            print("\n💰 Testing Basic Portfolio Endpoint...")
            
            portfolio_response = requests.get(f"{self.base_url}/portfolio", timeout=15)
            
            if portfolio_response.status_code == 200:
                portfolio_data = portfolio_response.json()
                self.log_success("Basic Portfolio Endpoint", "Portfolio retrieval working")
                print(f"Portfolio data: {json.dumps(portfolio_data, indent=2)}")
                return True
            else:
                self.log_error("Basic Portfolio Endpoint", f"HTTP {portfolio_response.status_code}: {portfolio_response.text}")
                return False
                
        except Exception as e:
            self.log_error("Basic Portfolio Endpoint", e)
            return False
            
    def run_focused_tests(self):
        """Run focused tests on the failing endpoints"""
        print("🎯 Starting Focused Tests for MongoDB ObjectId Serialization Issues")
        print("=" * 80)
        
        # Test basic endpoints first
        print("\n1. Testing Basic Endpoints...")
        portfolio_basic_ok = self.test_portfolio_endpoint()
        
        # Test data creation endpoints
        print("\n2. Testing Data Creation Endpoints...")
        trade_creation_ok = self.test_portfolio_trade_creation()
        backtest_creation_ok = self.test_backtest_creation()
        
        # Test the problematic endpoints
        print("\n3. Testing Problematic Endpoints...")
        portfolio_trades_ok = self.test_portfolio_trades_endpoint()
        backtest_results_ok = self.test_backtest_results_endpoint()
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 FOCUSED TEST RESULTS SUMMARY")
        print("=" * 80)
        
        results = {
            "Basic Portfolio Endpoint": portfolio_basic_ok,
            "Portfolio Trade Creation": trade_creation_ok,
            "Backtest Creation": backtest_creation_ok,
            "Portfolio Trades Endpoint (FAILING)": portfolio_trades_ok,
            "Backtest Results Endpoint (FAILING)": backtest_results_ok
        }
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name}: {status}")
            
        if self.errors:
            print(f"\n🚨 ERRORS ENCOUNTERED:")
            for error in self.errors:
                print(f"  {error}")
                
        return results

if __name__ == "__main__":
    if not BACKEND_URL:
        print("❌ Error: REACT_APP_BACKEND_URL not found in frontend/.env")
        exit(1)
        
    tester = FocusedTester()
    results = tester.run_focused_tests()