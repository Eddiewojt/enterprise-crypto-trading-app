"""
Multi-Exchange Manager for Legitimate Trading
Supports multiple legitimate exchanges with automatic detection
"""
import os
import logging
from datetime import datetime
import requests

class ExchangeManager:
    def __init__(self):
        self.supported_exchanges = {
            'binance': {
                'name': 'Binance',
                'test_url': 'https://api.binance.com/api/v3/ping',
                'regions': ['Global (Restricted in some countries)'],
                'compliance': 'High',
                'api_available': True
            },
            'kucoin': {
                'name': 'KuCoin',
                'test_url': 'https://api.kucoin.com/api/v1/timestamp',
                'regions': ['Global (200+ countries)'],
                'compliance': 'High',
                'api_available': True
            },
            'kraken': {
                'name': 'Kraken',
                'test_url': 'https://api.kraken.com/0/public/SystemStatus',
                'regions': ['US, EU, Canada, Japan'],
                'compliance': 'Very High',
                'api_available': True
            },
            'okx': {
                'name': 'OKX',
                'test_url': 'https://www.okx.com/api/v5/public/time',
                'regions': ['Global (Restricted in some countries)'],
                'compliance': 'High',
                'api_available': True
            },
            'bybit': {
                'name': 'Bybit',
                'test_url': 'https://api.bybit.com/v3/public/time',
                'regions': ['Global (Restricted in some countries)'],
                'compliance': 'Medium-High',
                'api_available': True
            }
        }
        self.available_exchanges = []
        self.primary_exchange = None
    
    async def detect_available_exchanges(self):
        """Detect which exchanges are accessible from current location"""
        available = []
        
        for exchange_id, exchange_info in self.supported_exchanges.items():
            try:
                # Test connectivity to exchange
                response = requests.get(exchange_info['test_url'], timeout=10)
                if response.status_code == 200:
                    available.append({
                        'id': exchange_id,
                        'name': exchange_info['name'],
                        'status': 'available',
                        'compliance': exchange_info['compliance'],
                        'regions': exchange_info['regions']
                    })
                    logging.info(f"✅ {exchange_info['name']} - Available")
                else:
                    logging.warning(f"⚠️ {exchange_info['name']} - HTTP {response.status_code}")
            except Exception as e:
                available.append({
                    'id': exchange_id,
                    'name': exchange_info['name'],
                    'status': 'blocked',
                    'error': str(e),
                    'compliance': exchange_info['compliance'],
                    'regions': exchange_info['regions']
                })
                logging.warning(f"❌ {exchange_info['name']} - {str(e)}")
        
        self.available_exchanges = available
        return available
    
    def get_recommended_exchange(self):
        """Get the best available exchange for user's region"""
        available = [ex for ex in self.available_exchanges if ex['status'] == 'available']
        
        if not available:
            return None
        
        # Prioritize by compliance and features
        compliance_order = {'Very High': 4, 'High': 3, 'Medium-High': 2, 'Medium': 1}
        
        best_exchange = max(available, key=lambda x: compliance_order.get(x['compliance'], 0))
        return best_exchange
    
    def get_exchange_setup_instructions(self, exchange_id):
        """Get setup instructions for specific exchange"""
        instructions = {
            'binance': {
                'signup_url': 'https://www.binance.com/en/register',
                'api_docs': 'https://binance-docs.github.io/apidocs/',
                'steps': [
                    'Create account at binance.com',
                    'Complete KYC verification',
                    'Enable API access in account settings',
                    'Generate API key with trading permissions',
                    'Add API credentials to the app'
                ],
                'cost': 'Free account, trading fees 0.1%',
                'features': ['Spot Trading', 'Futures', 'Options', 'Staking']
            },
            'kucoin': {
                'signup_url': 'https://www.kucoin.com/ucenter/signup',
                'api_docs': 'https://docs.kucoin.com/',
                'steps': [
                    'Create account at kucoin.com',
                    'Complete identity verification',
                    'Navigate to API Management',
                    'Create new API key with trading permissions',
                    'Add API credentials to the app'
                ],
                'cost': 'Free account, trading fees 0.1%',
                'features': ['Spot Trading', 'Futures', 'Margin', 'Bot Trading']
            },
            'kraken': {
                'signup_url': 'https://www.kraken.com/sign-up',
                'api_docs': 'https://docs.kraken.com/rest/',
                'steps': [
                    'Create account at kraken.com',
                    'Complete verification process',
                    'Go to Settings → API',
                    'Generate new API key with trading permissions',
                    'Add API credentials to the app'
                ],
                'cost': 'Free account, trading fees 0.16%',
                'features': ['Spot Trading', 'Futures', 'Staking', 'NFTs']
            },
            'okx': {
                'signup_url': 'https://www.okx.com/join',
                'api_docs': 'https://www.okx.com/docs-v5/',
                'steps': [
                    'Create account at okx.com',
                    'Complete KYC verification',
                    'Access API management section',
                    'Create trading API key',
                    'Add API credentials to the app'
                ],
                'cost': 'Free account, trading fees 0.08%',
                'features': ['Spot Trading', 'Derivatives', 'Copy Trading', 'DeFi']
            },
            'bybit': {
                'signup_url': 'https://www.bybit.com/register',
                'api_docs': 'https://bybit-exchange.github.io/docs/',
                'steps': [
                    'Create account at bybit.com',
                    'Complete identity verification',
                    'Navigate to API settings',
                    'Generate new API key',
                    'Add API credentials to the app'
                ],
                'cost': 'Free account, trading fees 0.1%',
                'features': ['Derivatives', 'Spot Trading', 'Copy Trading', 'Bots']
            }
        }
        
        return instructions.get(exchange_id, {})

# Global instance
exchange_manager = ExchangeManager()