#!/usr/bin/env python3
"""
Comprehensive Backend Testing for DOGE Trading App
Tests all API endpoints, Binance integration, technical analysis, and WebSocket functionality
"""

import requests
import json
import time
import asyncio
import websockets
import sys
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv('/app/frontend/.env')
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"Testing backend at: {API_BASE_URL}")

class DOGETradingAppTester:
    def __init__(self):
        self.base_url = API_BASE_URL
        self.test_results = {
            'binance_api_integration': False,
            'real_time_price_tracking': False,
            'technical_analysis_engine': False,
            'trading_signal_generation': False,
            'api_endpoints': False,
            'websocket_connection': False
        }
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
        
    def test_root_endpoint(self):
        """Test the root API endpoint"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if "message" in data:
                    self.log_success("Root Endpoint", f"Status: {response.status_code}")
                    return True
            self.log_error("Root Endpoint", f"Unexpected response: {response.status_code}")
            return False
        except Exception as e:
            self.log_error("Root Endpoint", e)
            return False
            
    def test_binance_api_integration(self):
        """Test Binance API integration by fetching DOGE price"""
        try:
            response = requests.get(f"{self.base_url}/doge/price", timeout=15)
            if response.status_code == 200:
                data = response.json()
                required_fields = ['symbol', 'price', 'change_24h', 'volume', 'high_24h', 'low_24h', 'timestamp']
                
                if all(field in data for field in required_fields):
                    if data['symbol'] == 'DOGEUSDT' and isinstance(data['price'], (int, float)) and data['price'] > 0:
                        self.log_success("Binance API Integration", f"DOGE Price: ${data['price']:.6f}")
                        self.test_results['binance_api_integration'] = True
                        return True
                        
                self.log_error("Binance API Integration", f"Invalid response format: {data}")
                return False
            else:
                self.log_error("Binance API Integration", f"HTTP {response.status_code}: {response.text}")
                return False
        except Exception as e:
            self.log_error("Binance API Integration", e)
            return False
            
    def test_klines_endpoint(self):
        """Test historical candlestick data endpoint"""
        try:
            # Test 15m timeframe
            response = requests.get(f"{self.base_url}/doge/klines?timeframe=15m&limit=50", timeout=15)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    kline = data[0]
                    required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    if all(field in kline for field in required_fields):
                        self.log_success("Klines Endpoint (15m)", f"Retrieved {len(data)} candlesticks")
                        
                        # Test 4h timeframe
                        response_4h = requests.get(f"{self.base_url}/doge/klines?timeframe=4h&limit=50", timeout=15)
                        if response_4h.status_code == 200:
                            data_4h = response_4h.json()
                            if isinstance(data_4h, list) and len(data_4h) > 0:
                                self.log_success("Klines Endpoint (4h)", f"Retrieved {len(data_4h)} candlesticks")
                                return True
                                
                self.log_error("Klines Endpoint", f"Invalid response format: {data}")
                return False
            else:
                self.log_error("Klines Endpoint", f"HTTP {response.status_code}: {response.text}")
                return False
        except Exception as e:
            self.log_error("Klines Endpoint", e)
            return False
            
    def test_technical_analysis_engine(self):
        """Test technical analysis calculations"""
        try:
            # Test 15m analysis
            response = requests.get(f"{self.base_url}/doge/analysis?timeframe=15m", timeout=15)
            if response.status_code == 200:
                data = response.json()
                required_fields = ['symbol', 'timeframe', 'current_price', 'rsi', 'macd', 'signal_line', 'sma_short', 'sma_long', 'analysis']
                
                if all(field in data for field in required_fields):
                    # Validate technical indicators
                    rsi = data['rsi']
                    macd = data['macd']
                    analysis = data['analysis']
                    
                    if (0 <= rsi <= 100 and 
                        isinstance(macd, (int, float)) and
                        'rsi_signal' in analysis and
                        'macd_signal' in analysis and
                        'ma_signal' in analysis):
                        
                        self.log_success("Technical Analysis Engine (15m)", 
                                       f"RSI: {rsi:.2f}, MACD: {macd:.6f}")
                        
                        # Test 4h analysis
                        response_4h = requests.get(f"{self.base_url}/doge/analysis?timeframe=4h", timeout=15)
                        if response_4h.status_code == 200:
                            data_4h = response_4h.json()
                            if all(field in data_4h for field in required_fields):
                                self.log_success("Technical Analysis Engine (4h)", 
                                               f"RSI: {data_4h['rsi']:.2f}, MACD: {data_4h['macd']:.6f}")
                                self.test_results['technical_analysis_engine'] = True
                                return True
                                
                self.log_error("Technical Analysis Engine", f"Invalid response format: {data}")
                return False
            else:
                self.log_error("Technical Analysis Engine", f"HTTP {response.status_code}: {response.text}")
                return False
        except Exception as e:
            self.log_error("Technical Analysis Engine", e)
            return False
            
    def test_trading_signal_generation(self):
        """Test trading signal generation"""
        try:
            response = requests.get(f"{self.base_url}/doge/signals", timeout=15)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    self.log_success("Trading Signal Generation", f"Retrieved {len(data)} signals")
                    
                    # If we have signals, validate their structure
                    if len(data) > 0:
                        signal = data[0]
                        required_fields = ['id', 'symbol', 'signal_type', 'strength', 'indicators', 'timeframe', 'price', 'timestamp']
                        
                        if all(field in signal for field in required_fields):
                            if (signal['signal_type'] in ['BUY', 'SELL'] and
                                60 <= signal['strength'] <= 100 and
                                signal['symbol'] == 'DOGEUSDT'):
                                
                                self.log_success("Signal Validation", 
                                               f"Type: {signal['signal_type']}, Strength: {signal['strength']:.1f}%")
                                self.test_results['trading_signal_generation'] = True
                                return True
                    else:
                        # No signals is also valid (might not have strong enough signals)
                        self.log_success("Trading Signal Generation", "No signals (normal if no strong signals)")
                        self.test_results['trading_signal_generation'] = True
                        return True
                        
                self.log_error("Trading Signal Generation", f"Invalid response format: {data}")
                return False
            else:
                self.log_error("Trading Signal Generation", f"HTTP {response.status_code}: {response.text}")
                return False
        except Exception as e:
            self.log_error("Trading Signal Generation", e)
            return False
            
    async def test_websocket_connection(self):
        """Test WebSocket real-time connection"""
        try:
            ws_url = f"{BACKEND_URL.replace('https://', 'wss://').replace('http://', 'ws://')}/api/ws"
            print(f"Testing WebSocket at: {ws_url}")
            
            async with websockets.connect(ws_url, timeout=10) as websocket:
                self.log_success("WebSocket Connection", "Connected successfully")
                
                # Wait for a message (with timeout)
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(message)
                    
                    if 'type' in data and 'data' in data:
                        if data['type'] in ['price', 'signal']:
                            self.log_success("WebSocket Real-time Data", f"Received {data['type']} update")
                            self.test_results['websocket_connection'] = True
                            self.test_results['real_time_price_tracking'] = True
                            return True
                            
                except asyncio.TimeoutError:
                    # No message received, but connection worked
                    self.log_success("WebSocket Connection", "Connected but no real-time data yet")
                    self.test_results['websocket_connection'] = True
                    return True
                    
        except Exception as e:
            self.log_error("WebSocket Connection", e)
            return False
            
    def run_all_tests(self):
        """Run all backend tests"""
        print("🚀 Starting DOGE Trading App Backend Tests")
        print("=" * 60)
        
        # Test API endpoints
        print("\n📡 Testing API Endpoints...")
        root_ok = self.test_root_endpoint()
        binance_ok = self.test_binance_api_integration()
        klines_ok = self.test_klines_endpoint()
        analysis_ok = self.test_technical_analysis_engine()
        signals_ok = self.test_trading_signal_generation()
        
        if root_ok and binance_ok and klines_ok:
            self.test_results['api_endpoints'] = True
            
        # Test WebSocket
        print("\n🔌 Testing WebSocket Connection...")
        try:
            websocket_ok = asyncio.run(self.test_websocket_connection())
        except Exception as e:
            self.log_error("WebSocket Test", e)
            websocket_ok = False
            
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
            
        print(f"\nTotal Tests: {len(self.test_results)}")
        print(f"Passed: {sum(self.test_results.values())}")
        print(f"Failed: {len(self.test_results) - sum(self.test_results.values())}")
        
        if self.errors:
            print(f"\n🚨 ERRORS ENCOUNTERED:")
            for error in self.errors:
                print(f"  {error}")
                
        return self.test_results

if __name__ == "__main__":
    if not BACKEND_URL:
        print("❌ Error: REACT_APP_BACKEND_URL not found in frontend/.env")
        sys.exit(1)
        
    tester = DOGETradingAppTester()
    results = tester.run_all_tests()
    
    # Exit with error code if any critical tests failed
    critical_tests = ['binance_api_integration', 'api_endpoints']
    if not all(results[test] for test in critical_tests):
        sys.exit(1)