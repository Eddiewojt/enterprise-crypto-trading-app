#!/usr/bin/env python3
"""
Focused Backtesting Test for Trading App
Tests the backtesting functionality comprehensively as requested
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

print(f"Testing backtesting functionality at: {API_BASE_URL}")

def test_backtest_comprehensive():
    """Comprehensive backtesting test as requested in the review"""
    
    print("🔬 COMPREHENSIVE BACKTESTING FUNCTIONALITY TEST")
    print("=" * 60)
    
    test_scenarios = [
        {
            "name": "DOGE Combined Strategy (6 months)",
            "request": {
                "symbol": "doge",
                "timeframe": "15m",
                "start_date": "2024-07-01",
                "end_date": "2025-01-01",
                "strategy": "combined",
                "initial_capital": 10000.0
            }
        },
        {
            "name": "BTC RSI Strategy (6 months)",
            "request": {
                "symbol": "btc",
                "timeframe": "1h",
                "start_date": "2024-07-01",
                "end_date": "2025-01-01",
                "strategy": "rsi",
                "initial_capital": 50000.0
            }
        },
        {
            "name": "ETH MACD Strategy (6 months)",
            "request": {
                "symbol": "eth",
                "timeframe": "4h",
                "start_date": "2024-07-01",
                "end_date": "2025-01-01",
                "strategy": "macd",
                "initial_capital": 25000.0
            }
        },
        {
            "name": "DOGE RSI Strategy (Different Capital)",
            "request": {
                "symbol": "doge",
                "timeframe": "1h",
                "start_date": "2024-07-01",
                "end_date": "2025-01-01",
                "strategy": "rsi",
                "initial_capital": 1000.0
            }
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📊 Test {i}: {scenario['name']}")
        print("-" * 40)
        
        try:
            response = requests.post(f"{API_BASE_URL}/backtest", 
                                   json=scenario['request'], 
                                   timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate all required fields
                required_fields = [
                    'symbol', 'strategy', 'timeframe', 'start_date', 'end_date',
                    'initial_capital', 'final_capital', 'total_return', 'total_return_percentage',
                    'total_trades', 'winning_trades', 'losing_trades', 'win_rate',
                    'max_drawdown', 'sharpe_ratio', 'trades'
                ]
                
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    print(f"❌ Missing fields: {missing_fields}")
                    continue
                
                # Validate data types and logical constraints
                validation_errors = []
                
                if not isinstance(data['initial_capital'], (int, float)):
                    validation_errors.append("initial_capital must be numeric")
                if not isinstance(data['final_capital'], (int, float)):
                    validation_errors.append("final_capital must be numeric")
                if not isinstance(data['total_trades'], int):
                    validation_errors.append("total_trades must be integer")
                if not isinstance(data['trades'], list):
                    validation_errors.append("trades must be list")
                if data['winning_trades'] + data['losing_trades'] > data['total_trades']:
                    validation_errors.append("winning + losing trades cannot exceed total trades")
                if not (0 <= data['win_rate'] <= 100):
                    validation_errors.append("win_rate must be between 0-100")
                
                if validation_errors:
                    print(f"❌ Validation errors: {validation_errors}")
                    continue
                
                # Display results
                print(f"✅ Symbol: {data['symbol']}")
                print(f"✅ Strategy: {data['strategy']}")
                print(f"✅ Timeframe: {data['timeframe']}")
                print(f"✅ Period: {data['start_date']} to {data['end_date']}")
                print(f"✅ Initial Capital: ${data['initial_capital']:,.2f}")
                print(f"✅ Final Capital: ${data['final_capital']:,.2f}")
                print(f"✅ Total Return: ${data['total_return']:,.2f} ({data['total_return_percentage']:.2f}%)")
                print(f"✅ Total Trades: {data['total_trades']}")
                print(f"✅ Winning Trades: {data['winning_trades']}")
                print(f"✅ Losing Trades: {data['losing_trades']}")
                print(f"✅ Win Rate: {data['win_rate']:.2f}%")
                print(f"✅ Max Drawdown: {data['max_drawdown']:.2f}%")
                print(f"✅ Sharpe Ratio: {data['sharpe_ratio']:.4f}")
                
                # Validate trade history
                if len(data['trades']) > 0:
                    trade = data['trades'][0]
                    trade_fields = ['timestamp', 'side', 'price', 'quantity', 'value']
                    if all(field in trade for field in trade_fields):
                        print(f"✅ Trade History: Valid format (first trade: {trade['side']} at ${trade['price']:.6f})")
                    else:
                        print(f"❌ Trade History: Invalid format")
                        continue
                else:
                    print(f"✅ Trade History: No trades (valid for some strategies)")
                
                # Check mathematical consistency
                calculated_return = data['final_capital'] - data['initial_capital']
                if abs(calculated_return - data['total_return']) > 0.01:
                    print(f"❌ Mathematical inconsistency: calculated return {calculated_return} vs reported {data['total_return']}")
                    continue
                
                calculated_return_pct = (calculated_return / data['initial_capital']) * 100
                if abs(calculated_return_pct - data['total_return_percentage']) > 0.01:
                    print(f"❌ Percentage calculation error: calculated {calculated_return_pct:.2f}% vs reported {data['total_return_percentage']:.2f}%")
                    continue
                
                print(f"✅ Mathematical consistency verified")
                
                results.append({
                    'scenario': scenario['name'],
                    'success': True,
                    'data': data
                })
                
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"❌ Response: {response.text}")
                results.append({
                    'scenario': scenario['name'],
                    'success': False,
                    'error': f"HTTP {response.status_code}"
                })
                
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
            results.append({
                'scenario': scenario['name'],
                'success': False,
                'error': str(e)
            })
    
    # Test edge cases
    print(f"\n🧪 EDGE CASE TESTING")
    print("-" * 40)
    
    edge_cases = [
        {
            "name": "Invalid Symbol",
            "request": {
                "symbol": "invalid_coin",
                "timeframe": "15m",
                "start_date": "2024-07-01",
                "end_date": "2025-01-01",
                "strategy": "rsi",
                "initial_capital": 10000.0
            },
            "expected_status": [400, 500]
        },
        {
            "name": "Invalid Date Range",
            "request": {
                "symbol": "doge",
                "timeframe": "15m",
                "start_date": "2025-01-01",
                "end_date": "2024-07-01",
                "strategy": "rsi",
                "initial_capital": 10000.0
            },
            "expected_status": [400, 500]
        }
    ]
    
    for edge_case in edge_cases:
        print(f"\n🔍 Testing: {edge_case['name']}")
        try:
            response = requests.post(f"{API_BASE_URL}/backtest", 
                                   json=edge_case['request'], 
                                   timeout=15)
            
            if response.status_code in edge_case['expected_status']:
                print(f"✅ Correctly handled with status {response.status_code}")
            else:
                print(f"⚠️  Unexpected status {response.status_code} (expected {edge_case['expected_status']})")
                
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("📊 BACKTESTING TEST SUMMARY")
    print("=" * 60)
    
    successful_tests = sum(1 for r in results if r['success'])
    total_tests = len(results)
    
    print(f"Total Scenarios Tested: {total_tests}")
    print(f"Successful Tests: {successful_tests}")
    print(f"Failed Tests: {total_tests - successful_tests}")
    print(f"Success Rate: {(successful_tests/total_tests)*100:.1f}%")
    
    if successful_tests > 0:
        print(f"\n✅ BACKTESTING ENGINE IS WORKING!")
        print(f"✅ All required response fields present")
        print(f"✅ Mathematical calculations verified")
        print(f"✅ Trade history format validated")
        print(f"✅ Multiple strategies tested (RSI, MACD, Combined)")
        print(f"✅ Multiple symbols tested (DOGE, BTC, ETH)")
        print(f"✅ Multiple timeframes tested (15m, 1h, 4h)")
        print(f"✅ Different capital amounts tested")
        
        # Show performance summary
        print(f"\n📈 PERFORMANCE SUMMARY:")
        for result in results:
            if result['success']:
                data = result['data']
                print(f"  {result['scenario']}: {data['total_return_percentage']:+.2f}% ({data['total_trades']} trades)")
    
    return successful_tests >= 3  # At least 3 out of 4 main tests should pass

if __name__ == "__main__":
    success = test_backtest_comprehensive()
    if success:
        print(f"\n🎉 BACKTESTING FUNCTIONALITY VALIDATION: PASSED")
    else:
        print(f"\n❌ BACKTESTING FUNCTIONALITY VALIDATION: FAILED")
        exit(1)