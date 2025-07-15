#!/usr/bin/env python3
"""
Ultra-Fast Cryptocurrency Trading Platform Performance Tests
Tests ZERO DELAY and maximum performance with increased budget
"""

import requests
import json
import time
import asyncio
import concurrent.futures
import threading
from datetime import datetime
from dotenv import load_dotenv
import os
import statistics

# Load environment variables
load_dotenv('/app/frontend/.env')
BACKEND_URL = os.getenv('REACT_APP_BACKEND_URL')
API_BASE_URL = f"{BACKEND_URL}/api"

print(f"🚀 ULTRA-FAST PERFORMANCE TESTING at: {API_BASE_URL}")

class UltraFastPerformanceTester:
    def __init__(self):
        self.base_url = API_BASE_URL
        self.test_results = {}
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
        
    def log_performance(self, test_name, response_time, threshold, details=""):
        """Log performance results"""
        if response_time <= threshold:
            status = "✅ ULTRA-FAST"
            color = "\033[92m"  # Green
        elif response_time <= threshold * 1.5:
            status = "⚠️ ACCEPTABLE"
            color = "\033[93m"  # Yellow
        else:
            status = "❌ TOO SLOW"
            color = "\033[91m"  # Red
        
        reset_color = "\033[0m"
        
        print(f"{color}{status} {test_name}: {response_time:.0f}ms (target: <{threshold}ms){reset_color}")
        if details:
            print(f"   {details}")
            
        return response_time <= threshold

    def test_ultra_fast_multi_coin_prices(self):
        """🎯 CRITICAL TEST: /api/multi-coin/prices should respond in <200ms"""
        try:
            print("\n🚀 ULTRA-FAST API PERFORMANCE TEST 1: Multi-Coin Prices")
            print("=" * 80)
            print("🎯 TARGET: Response time <200ms")
            print("🎯 EXPECTED: Real live data with sources 'Coinbase_Ultra', 'CryptoCompare_Ultra', or 'Binance_Public_Ultra'")
            print("=" * 80)
            
            # Perform multiple requests to get average response time
            response_times = []
            successful_requests = 0
            
            for i in range(5):
                start_time = time.time()
                response = requests.get(f"{self.base_url}/multi-coin/prices", timeout=5)
                end_time = time.time()
                
                response_time_ms = (end_time - start_time) * 1000
                response_times.append(response_time_ms)
                
                if response.status_code == 200:
                    successful_requests += 1
                    
                    if i == 0:  # Analyze first response for data quality
                        data = response.json()
                        print(f"📊 Response Status: {response.status_code}")
                        print(f"📊 Data Structure: {type(data)} with {len(data) if isinstance(data, dict) else 'N/A'} coins")
                        
                        if isinstance(data, dict) and len(data) > 0:
                            # Check for ultra-fast sources
                            sample_coin = list(data.keys())[0]
                            sample_data = data[sample_coin]
                            
                            if 'source' in sample_data:
                                source = sample_data['source']
                                print(f"🎯 DATA SOURCE: {source}")
                                
                                ultra_fast_sources = ['Coinbase_Ultra', 'CryptoCompare_Ultra', 'Binance_Public_Ultra']
                                if source in ultra_fast_sources:
                                    print(f"✅ ULTRA-FAST SOURCE CONFIRMED: {source}")
                                else:
                                    print(f"⚠️ NON-ULTRA SOURCE: {source} (expected: {ultra_fast_sources})")
                            
                            # Check for current market prices
                            if 'DOGEUSDT' in data:
                                doge_price = data['DOGEUSDT'].get('price', 0)
                                print(f"🐕 DOGE Price: ${doge_price:.6f}")
                                
                                # Check if price is in realistic current range (~$0.19-0.20)
                                if 0.15 <= doge_price <= 0.25:
                                    print("✅ DOGE PRICE: In realistic current market range")
                                else:
                                    print(f"⚠️ DOGE PRICE: Outside expected range $0.15-0.25")
                            
                            if 'BTCUSDT' in data:
                                btc_price = data['BTCUSDT'].get('price', 0)
                                print(f"₿ BTC Price: ${btc_price:.2f}")
                                
                                # Check if price is in realistic current range (~$115,000-120,000)
                                if 110000 <= btc_price <= 125000:
                                    print("✅ BTC PRICE: In realistic current market range")
                                else:
                                    print(f"⚠️ BTC PRICE: Outside expected range $110,000-125,000")
                
                print(f"Request {i+1}: {response_time_ms:.0f}ms")
            
            # Calculate performance metrics
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            print(f"\n📊 PERFORMANCE METRICS:")
            print(f"   Average: {avg_response_time:.0f}ms")
            print(f"   Fastest: {min_response_time:.0f}ms")
            print(f"   Slowest: {max_response_time:.0f}ms")
            print(f"   Success Rate: {successful_requests}/5 requests")
            
            # Performance evaluation
            performance_passed = self.log_performance(
                "Multi-Coin Prices API", 
                avg_response_time, 
                200,
                f"Success rate: {successful_requests}/5, Range: {min_response_time:.0f}-{max_response_time:.0f}ms"
            )
            
            if performance_passed and successful_requests >= 4:
                self.test_results['ultra_fast_multi_coin_prices'] = True
                return True
            else:
                self.test_results['ultra_fast_multi_coin_prices'] = False
                return False
                
        except Exception as e:
            self.log_error("Ultra-Fast Multi-Coin Prices", e)
            self.test_results['ultra_fast_multi_coin_prices'] = False
            return False

    def test_ultra_fast_mobile_quick_prices(self):
        """🎯 CRITICAL TEST: /api/mobile/quick-prices should respond in <100ms"""
        try:
            print("\n📱 ULTRA-FAST API PERFORMANCE TEST 2: Mobile Quick Prices")
            print("=" * 80)
            print("🎯 TARGET: Response time <100ms")
            print("🎯 EXPECTED: mobile_optimized: true, update_interval ≤ 3 seconds, top 5 coins prioritized")
            print("=" * 80)
            
            # Perform multiple requests to get average response time
            response_times = []
            successful_requests = 0
            
            for i in range(5):
                start_time = time.time()
                response = requests.get(f"{self.base_url}/mobile/quick-prices", timeout=3)
                end_time = time.time()
                
                response_time_ms = (end_time - start_time) * 1000
                response_times.append(response_time_ms)
                
                if response.status_code == 200:
                    successful_requests += 1
                    
                    if i == 0:  # Analyze first response
                        data = response.json()
                        print(f"📊 Response Status: {response.status_code}")
                        print(f"📊 Data Structure: {type(data)}")
                        
                        # Check for mobile optimization indicators
                        if isinstance(data, dict):
                            if 'mobile_optimized' in data:
                                mobile_opt = data['mobile_optimized']
                                print(f"📱 Mobile Optimized: {mobile_opt}")
                                if mobile_opt:
                                    print("✅ MOBILE OPTIMIZATION: Confirmed")
                                else:
                                    print("❌ MOBILE OPTIMIZATION: Not enabled")
                            
                            if 'update_interval' in data:
                                update_interval = data['update_interval']
                                print(f"⏱️ Update Interval: {update_interval} seconds")
                                if update_interval <= 3:
                                    print("✅ UPDATE INTERVAL: ≤3 seconds (ultra-fast)")
                                else:
                                    print(f"⚠️ UPDATE INTERVAL: {update_interval}s (target: ≤3s)")
                            
                            if 'top_coins' in data:
                                top_coins = data['top_coins']
                                if isinstance(top_coins, list):
                                    print(f"🏆 Top Coins: {len(top_coins)} coins prioritized")
                                    if len(top_coins) == 5:
                                        print("✅ TOP 5 PRIORITIZATION: Confirmed")
                                        coin_names = [coin.get('symbol', 'Unknown') for coin in top_coins[:3]]
                                        print(f"   Priority coins: {', '.join(coin_names)}...")
                                    else:
                                        print(f"⚠️ TOP COINS: {len(top_coins)} coins (expected: 5)")
                
                print(f"Request {i+1}: {response_time_ms:.0f}ms")
            
            # Calculate performance metrics
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            print(f"\n📊 MOBILE PERFORMANCE METRICS:")
            print(f"   Average: {avg_response_time:.0f}ms")
            print(f"   Fastest: {min_response_time:.0f}ms")
            print(f"   Slowest: {max_response_time:.0f}ms")
            print(f"   Success Rate: {successful_requests}/5 requests")
            
            # Performance evaluation (stricter for mobile)
            performance_passed = self.log_performance(
                "Mobile Quick Prices API", 
                avg_response_time, 
                100,
                f"Success rate: {successful_requests}/5, Range: {min_response_time:.0f}-{max_response_time:.0f}ms"
            )
            
            if performance_passed and successful_requests >= 4:
                self.test_results['ultra_fast_mobile_quick_prices'] = True
                return True
            else:
                self.test_results['ultra_fast_mobile_quick_prices'] = False
                return False
                
        except Exception as e:
            self.log_error("Ultra-Fast Mobile Quick Prices", e)
            self.test_results['ultra_fast_mobile_quick_prices'] = False
            return False

    def test_ultra_fast_signals(self):
        """🎯 CRITICAL TEST: /api/signals should respond in <100ms"""
        try:
            print("\n🎯 ULTRA-FAST API PERFORMANCE TEST 3: Signal Generation")
            print("=" * 80)
            print("🎯 TARGET: Response time <100ms")
            print("🎯 EXPECTED: Recent calculated signals based on real price movements")
            print("=" * 80)
            
            # Perform multiple requests to get average response time
            response_times = []
            successful_requests = 0
            
            for i in range(5):
                start_time = time.time()
                response = requests.get(f"{self.base_url}/signals", timeout=3)
                end_time = time.time()
                
                response_time_ms = (end_time - start_time) * 1000
                response_times.append(response_time_ms)
                
                if response.status_code == 200:
                    successful_requests += 1
                    
                    if i == 0:  # Analyze first response
                        data = response.json()
                        print(f"📊 Response Status: {response.status_code}")
                        print(f"📊 Data Structure: {type(data)}")
                        
                        if isinstance(data, list):
                            print(f"🎯 Signals Count: {len(data)} signals")
                            
                            if len(data) > 0:
                                # Analyze signal quality
                                recent_signals = 0
                                strong_signals = 0
                                
                                for signal in data[:5]:  # Check first 5 signals
                                    if isinstance(signal, dict):
                                        signal_type = signal.get('signal_type', 'Unknown')
                                        strength = signal.get('strength', 0)
                                        timestamp = signal.get('timestamp', '')
                                        symbol = signal.get('symbol', 'Unknown')
                                        
                                        # Check if signal is recent (within last hour)
                                        try:
                                            signal_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                            time_diff = datetime.now().astimezone() - signal_time.astimezone()
                                            if time_diff.total_seconds() < 3600:  # 1 hour
                                                recent_signals += 1
                                        except:
                                            pass
                                        
                                        if strength >= 70:
                                            strong_signals += 1
                                        
                                        if signal == data[0]:  # Log first signal details
                                            print(f"🎯 Latest Signal: {signal_type} {symbol} (Strength: {strength}%)")
                                
                                print(f"✅ Recent Signals: {recent_signals}/{min(len(data), 5)} within last hour")
                                print(f"✅ Strong Signals: {strong_signals}/{min(len(data), 5)} with ≥70% strength")
                            else:
                                print("ℹ️ No signals available (normal if no strong signals detected)")
                
                print(f"Request {i+1}: {response_time_ms:.0f}ms")
            
            # Calculate performance metrics
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            print(f"\n📊 SIGNALS PERFORMANCE METRICS:")
            print(f"   Average: {avg_response_time:.0f}ms")
            print(f"   Fastest: {min_response_time:.0f}ms")
            print(f"   Slowest: {max_response_time:.0f}ms")
            print(f"   Success Rate: {successful_requests}/5 requests")
            
            # Performance evaluation
            performance_passed = self.log_performance(
                "Signals API", 
                avg_response_time, 
                100,
                f"Success rate: {successful_requests}/5, Range: {min_response_time:.0f}-{max_response_time:.0f}ms"
            )
            
            if performance_passed and successful_requests >= 4:
                self.test_results['ultra_fast_signals'] = True
                return True
            else:
                self.test_results['ultra_fast_signals'] = False
                return False
                
        except Exception as e:
            self.log_error("Ultra-Fast Signals", e)
            self.test_results['ultra_fast_signals'] = False
            return False

    def test_concurrent_performance(self):
        """🎯 CRITICAL TEST: Backend should handle 10+ concurrent requests"""
        try:
            print("\n⚡ CONCURRENT PERFORMANCE TEST: 10+ Simultaneous Requests")
            print("=" * 80)
            print("🎯 TARGET: Handle 10+ concurrent requests without rate limiting")
            print("🎯 EXPECTED: All responses consistent, no delays, no errors")
            print("=" * 80)
            
            def make_request(endpoint, request_id):
                """Make a single request and return timing info"""
                start_time = time.time()
                try:
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                    end_time = time.time()
                    
                    return {
                        'request_id': request_id,
                        'endpoint': endpoint,
                        'status_code': response.status_code,
                        'response_time': (end_time - start_time) * 1000,
                        'success': response.status_code == 200,
                        'data_size': len(response.text) if response.status_code == 200 else 0
                    }
                except Exception as e:
                    end_time = time.time()
                    return {
                        'request_id': request_id,
                        'endpoint': endpoint,
                        'status_code': 0,
                        'response_time': (end_time - start_time) * 1000,
                        'success': False,
                        'error': str(e)
                    }
            
            # Define test endpoints
            test_endpoints = [
                '/multi-coin/prices',
                '/mobile/quick-prices', 
                '/signals',
                '/doge/price',
                '/btc/price',
                '/eth/price',
                '/supported-coins',
                '/doge/analysis?timeframe=15m',
                '/btc/analysis?timeframe=1h',
                '/portfolio'
            ]
            
            # Create concurrent requests
            concurrent_requests = []
            for i in range(12):  # 12 concurrent requests
                endpoint = test_endpoints[i % len(test_endpoints)]
                concurrent_requests.append((endpoint, i + 1))
            
            print(f"🚀 Launching {len(concurrent_requests)} concurrent requests...")
            
            # Execute concurrent requests
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
                future_to_request = {
                    executor.submit(make_request, endpoint, req_id): (endpoint, req_id)
                    for endpoint, req_id in concurrent_requests
                }
                
                results = []
                for future in concurrent.futures.as_completed(future_to_request):
                    result = future.result()
                    results.append(result)
            
            total_time = time.time() - start_time
            
            # Analyze results
            successful_requests = sum(1 for r in results if r['success'])
            failed_requests = len(results) - successful_requests
            
            response_times = [r['response_time'] for r in results if r['success']]
            if response_times:
                avg_response_time = statistics.mean(response_times)
                min_response_time = min(response_times)
                max_response_time = max(response_times)
            else:
                avg_response_time = min_response_time = max_response_time = 0
            
            print(f"\n📊 CONCURRENT PERFORMANCE RESULTS:")
            print(f"   Total Execution Time: {total_time:.2f}s")
            print(f"   Successful Requests: {successful_requests}/{len(results)}")
            print(f"   Failed Requests: {failed_requests}")
            print(f"   Average Response Time: {avg_response_time:.0f}ms")
            print(f"   Fastest Response: {min_response_time:.0f}ms")
            print(f"   Slowest Response: {max_response_time:.0f}ms")
            
            # Check for consistency
            endpoint_results = {}
            for result in results:
                endpoint = result['endpoint']
                if endpoint not in endpoint_results:
                    endpoint_results[endpoint] = []
                endpoint_results[endpoint].append(result)
            
            consistent_responses = True
            for endpoint, endpoint_results_list in endpoint_results.items():
                successful_for_endpoint = [r for r in endpoint_results_list if r['success']]
                if len(successful_for_endpoint) > 1:
                    # Check if data sizes are consistent (indicating same data returned)
                    data_sizes = [r['data_size'] for r in successful_for_endpoint]
                    if len(set(data_sizes)) > 1:
                        print(f"⚠️ INCONSISTENT RESPONSES for {endpoint}: Different data sizes {set(data_sizes)}")
                        consistent_responses = False
                    else:
                        print(f"✅ CONSISTENT RESPONSES for {endpoint}: {len(successful_for_endpoint)} requests")
            
            # Performance evaluation
            success_rate = successful_requests / len(results)
            performance_acceptable = (
                success_rate >= 0.9 and  # 90% success rate
                avg_response_time <= 500 and  # Average under 500ms
                max_response_time <= 2000 and  # No request over 2 seconds
                consistent_responses
            )
            
            if performance_acceptable:
                self.log_success("Concurrent Performance", 
                               f"{successful_requests}/{len(results)} requests successful, avg {avg_response_time:.0f}ms")
                self.test_results['concurrent_performance'] = True
                return True
            else:
                self.log_error("Concurrent Performance", 
                             f"Performance issues: {success_rate:.1%} success rate, {avg_response_time:.0f}ms avg")
                self.test_results['concurrent_performance'] = False
                return False
                
        except Exception as e:
            self.log_error("Concurrent Performance", e)
            self.test_results['concurrent_performance'] = False
            return False

    def test_zero_delay_real_time_pricing(self):
        """🎯 CRITICAL TEST: Verify zero delay in price updates"""
        try:
            print("\n⚡ ZERO DELAY REAL-TIME PRICING TEST")
            print("=" * 80)
            print("🎯 TARGET: Zero delay in price updates, ultra-fast cache system")
            print("🎯 EXPECTED: 3-second cache duration, multiple concurrent sources")
            print("=" * 80)
            
            # Test cache performance by making rapid successive requests
            cache_test_times = []
            
            print("🔄 Testing cache performance with rapid requests...")
            for i in range(10):
                start_time = time.time()
                response = requests.get(f"{self.base_url}/multi-coin/prices", timeout=5)
                end_time = time.time()
                
                response_time_ms = (end_time - start_time) * 1000
                cache_test_times.append(response_time_ms)
                
                if response.status_code == 200:
                    print(f"Cache test {i+1}: {response_time_ms:.0f}ms")
                else:
                    print(f"Cache test {i+1}: FAILED ({response.status_code})")
                
                # Small delay to test cache behavior
                time.sleep(0.1)
            
            # Analyze cache performance
            avg_cache_time = statistics.mean(cache_test_times)
            
            # Test if subsequent requests are faster (indicating cache hit)
            first_request_time = cache_test_times[0]
            subsequent_avg = statistics.mean(cache_test_times[1:5])  # Next 4 requests
            
            cache_improvement = first_request_time - subsequent_avg
            
            print(f"\n📊 CACHE PERFORMANCE ANALYSIS:")
            print(f"   First Request: {first_request_time:.0f}ms")
            print(f"   Subsequent Avg: {subsequent_avg:.0f}ms")
            print(f"   Cache Improvement: {cache_improvement:.0f}ms")
            print(f"   Overall Average: {avg_cache_time:.0f}ms")
            
            # Test data freshness
            print("\n🔄 Testing data freshness...")
            response1 = requests.get(f"{self.base_url}/multi-coin/prices", timeout=5)
            time.sleep(4)  # Wait longer than cache duration (3 seconds)
            response2 = requests.get(f"{self.base_url}/multi-coin/prices", timeout=5)
            
            if response1.status_code == 200 and response2.status_code == 200:
                data1 = response1.json()
                data2 = response2.json()
                
                # Check if timestamps are different (indicating fresh data)
                if 'DOGEUSDT' in data1 and 'DOGEUSDT' in data2:
                    timestamp1 = data1['DOGEUSDT'].get('timestamp', '')
                    timestamp2 = data2['DOGEUSDT'].get('timestamp', '')
                    
                    if timestamp1 != timestamp2:
                        print("✅ DATA FRESHNESS: Timestamps updated after cache expiry")
                    else:
                        print("⚠️ DATA FRESHNESS: Same timestamp after cache expiry")
            
            # Performance evaluation
            zero_delay_achieved = (
                avg_cache_time <= 50 and  # Ultra-fast average
                cache_improvement > 0 and  # Cache is working
                subsequent_avg <= 30  # Cached requests are very fast
            )
            
            if zero_delay_achieved:
                self.log_success("Zero Delay Real-Time Pricing", 
                               f"Ultra-fast cache: {avg_cache_time:.0f}ms avg, {cache_improvement:.0f}ms improvement")
                self.test_results['zero_delay_real_time_pricing'] = True
                return True
            else:
                self.log_error("Zero Delay Real-Time Pricing", 
                             f"Performance issues: {avg_cache_time:.0f}ms avg, cache improvement: {cache_improvement:.0f}ms")
                self.test_results['zero_delay_real_time_pricing'] = False
                return False
                
        except Exception as e:
            self.log_error("Zero Delay Real-Time Pricing", e)
            self.test_results['zero_delay_real_time_pricing'] = False
            return False

    def run_all_ultra_fast_tests(self):
        """Run all ultra-fast performance tests"""
        print("🚀" * 30)
        print("🚀 ULTRA-FAST CRYPTOCURRENCY TRADING PLATFORM TESTING")
        print("🚀 ZERO DELAY & MAXIMUM PERFORMANCE WITH INCREASED BUDGET")
        print("🚀" * 30)
        
        test_methods = [
            ('Ultra-Fast Multi-Coin Prices (<200ms)', self.test_ultra_fast_multi_coin_prices),
            ('Ultra-Fast Mobile Quick Prices (<100ms)', self.test_ultra_fast_mobile_quick_prices),
            ('Ultra-Fast Signal Generation (<100ms)', self.test_ultra_fast_signals),
            ('Concurrent Performance (10+ requests)', self.test_concurrent_performance),
            ('Zero Delay Real-Time Pricing', self.test_zero_delay_real_time_pricing)
        ]
        
        passed_tests = 0
        total_tests = len(test_methods)
        
        for test_name, test_method in test_methods:
            print(f"\n{'='*80}")
            print(f"🧪 RUNNING: {test_name}")
            print(f"{'='*80}")
            
            try:
                if test_method():
                    passed_tests += 1
                    print(f"✅ PASSED: {test_name}")
                else:
                    print(f"❌ FAILED: {test_name}")
            except Exception as e:
                print(f"💥 ERROR in {test_name}: {e}")
                self.log_error(test_name, e)
        
        # Final summary
        print(f"\n{'🎯' * 30}")
        print(f"🎯 ULTRA-FAST PERFORMANCE TEST SUMMARY")
        print(f"{'🎯' * 30}")
        print(f"✅ PASSED: {passed_tests}/{total_tests} tests")
        print(f"❌ FAILED: {total_tests - passed_tests}/{total_tests} tests")
        
        if passed_tests == total_tests:
            print("🎉 ALL ULTRA-FAST PERFORMANCE TESTS PASSED!")
            print("🚀 ZERO DELAY ACHIEVED - MAXIMUM PERFORMANCE CONFIRMED")
        elif passed_tests >= total_tests * 0.8:
            print("⚠️ MOST TESTS PASSED - GOOD PERFORMANCE WITH MINOR ISSUES")
        else:
            print("❌ PERFORMANCE ISSUES DETECTED - OPTIMIZATION NEEDED")
        
        # Error summary
        if self.errors:
            print(f"\n🚨 ERRORS ENCOUNTERED:")
            for error in self.errors:
                print(f"   {error}")
        
        return passed_tests, total_tests, self.test_results

if __name__ == "__main__":
    tester = UltraFastPerformanceTester()
    passed, total, results = tester.run_all_ultra_fast_tests()
    
    # Exit with appropriate code
    if passed == total:
        exit(0)  # All tests passed
    else:
        exit(1)  # Some tests failed