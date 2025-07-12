import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum

class BotStrategy(Enum):
    GRID_TRADING = "grid_trading"
    DCA = "dca"  # Dollar Cost Averaging
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    COPY_TRADING = "copy_trading"

@dataclass
class BotConfig:
    """Trading Bot Configuration"""
    bot_id: str
    name: str
    strategy: BotStrategy
    symbol: str
    investment_amount: float
    risk_level: str  # 'low', 'medium', 'high'
    max_drawdown: float
    stop_loss: float
    take_profit: float
    active: bool = True
    created_at: datetime = None

@dataclass
class BotPerformance:
    """Bot Performance Metrics"""
    bot_id: str
    total_trades: int
    winning_trades: int
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    current_position: str
    last_trade: datetime

class TradingBotEngine:
    """Advanced Trading Bot Management System"""
    
    def __init__(self):
        self.active_bots = {}
        self.bot_performances = {}
        self.strategy_handlers = {
            BotStrategy.GRID_TRADING: self._execute_grid_strategy,
            BotStrategy.DCA: self._execute_dca_strategy,
            BotStrategy.MOMENTUM: self._execute_momentum_strategy,
            BotStrategy.MEAN_REVERSION: self._execute_mean_reversion_strategy,
            BotStrategy.ARBITRAGE: self._execute_arbitrage_strategy,
            BotStrategy.COPY_TRADING: self._execute_copy_trading_strategy
        }
    
    async def create_bot(self, config: BotConfig) -> Dict:
        """Create a new trading bot"""
        try:
            config.created_at = datetime.utcnow()
            self.active_bots[config.bot_id] = config
            
            # Initialize performance tracking
            self.bot_performances[config.bot_id] = BotPerformance(
                bot_id=config.bot_id,
                total_trades=0,
                winning_trades=0,
                total_return=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                current_position="none",
                last_trade=datetime.utcnow()
            )
            
            return {
                "status": "success",
                "bot_id": config.bot_id,
                "message": f"Trading bot '{config.name}' created successfully",
                "config": config.__dict__
            }
            
        except Exception as e:
            logging.error(f"Bot creation error: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_bot_recommendations(self, user_profile: Dict) -> List[Dict]:
        """Get personalized bot recommendations"""
        try:
            recommendations = []
            
            risk_level = user_profile.get("risk_tolerance", "medium")
            investment_amount = user_profile.get("investment_amount", 1000)
            experience = user_profile.get("experience", "beginner")
            
            # Beginner-friendly bots
            if experience == "beginner":
                recommendations.append({
                    "strategy": "DCA",
                    "name": "Bitcoin DCA Bot",
                    "description": "Automatically buy Bitcoin at regular intervals",
                    "risk_level": "low",
                    "expected_return": "8-15% annually",
                    "min_investment": 100,
                    "complexity": "easy"
                })
                
                recommendations.append({
                    "strategy": "Grid Trading",
                    "name": "ETH Grid Bot",
                    "description": "Profit from ETH price fluctuations in a range",
                    "risk_level": "medium",
                    "expected_return": "12-25% annually",
                    "min_investment": 500,
                    "complexity": "medium"
                })
            
            # Intermediate bots
            if experience in ["intermediate", "advanced"]:
                recommendations.append({
                    "strategy": "Momentum",
                    "name": "Altcoin Momentum Bot",
                    "description": "Catch trending altcoin moves",
                    "risk_level": "high",
                    "expected_return": "20-50% annually",
                    "min_investment": 1000,
                    "complexity": "advanced"
                })
                
                recommendations.append({
                    "strategy": "Arbitrage",
                    "name": "Cross-Exchange Arbitrage",
                    "description": "Profit from price differences across exchanges",
                    "risk_level": "medium",
                    "expected_return": "15-30% annually",
                    "min_investment": 2000,
                    "complexity": "advanced"
                })
            
            # Advanced bots
            if experience == "advanced":
                recommendations.append({
                    "strategy": "Copy Trading",
                    "name": "Top Trader Copy Bot",
                    "description": "Copy trades from top-performing traders",
                    "risk_level": risk_level,
                    "expected_return": "Variable based on copied trader",
                    "min_investment": 500,
                    "complexity": "medium"
                })
            
            return recommendations
            
        except Exception as e:
            logging.error(f"Bot recommendations error: {e}")
            return []
    
    async def execute_bot_strategy(self, bot_id: str, market_data: Dict) -> Dict:
        """Execute trading strategy for a specific bot"""
        try:
            if bot_id not in self.active_bots:
                return {"error": "Bot not found"}
            
            bot_config = self.active_bots[bot_id]
            
            if not bot_config.active:
                return {"status": "inactive"}
            
            # Execute strategy
            strategy_handler = self.strategy_handlers.get(bot_config.strategy)
            if strategy_handler:
                result = await strategy_handler(bot_config, market_data)
                
                # Update performance metrics
                await self._update_bot_performance(bot_id, result)
                
                return result
            else:
                return {"error": "Strategy not implemented"}
                
        except Exception as e:
            logging.error(f"Bot execution error: {e}")
            return {"error": str(e)}
    
    async def _execute_grid_strategy(self, config: BotConfig, market_data: Dict) -> Dict:
        """Execute Grid Trading Strategy"""
        try:
            current_price = market_data.get("price", 0)
            
            # Grid parameters
            grid_range = 0.05  # 5% range
            grid_levels = 10
            upper_bound = current_price * (1 + grid_range)
            lower_bound = current_price * (1 - grid_range)
            grid_step = (upper_bound - lower_bound) / grid_levels
            
            # Determine action
            action = "hold"
            price_level = round((current_price - lower_bound) / grid_step)
            
            # Buy if price hits lower grid levels
            if price_level <= 3:
                action = "buy"
                quantity = config.investment_amount / (grid_levels * current_price)
            # Sell if price hits upper grid levels
            elif price_level >= 7:
                action = "sell"
                quantity = config.investment_amount / (grid_levels * current_price)
            else:
                quantity = 0
            
            return {
                "strategy": "grid_trading",
                "action": action,
                "quantity": quantity,
                "price": current_price,
                "grid_level": price_level,
                "reason": f"Price at grid level {price_level}",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Grid strategy error: {e}")
            return {"error": str(e)}
    
    async def _execute_dca_strategy(self, config: BotConfig, market_data: Dict) -> Dict:
        """Execute Dollar Cost Averaging Strategy"""
        try:
            current_price = market_data.get("price", 0)
            
            # DCA parameters
            interval_hours = 24  # Daily DCA
            investment_per_interval = config.investment_amount / 30  # Monthly divided by days
            
            # Check if it's time to buy
            performance = self.bot_performances.get(config.bot_id)
            last_trade_time = performance.last_trade if performance else datetime.utcnow() - timedelta(hours=25)
            
            hours_since_last_trade = (datetime.utcnow() - last_trade_time).total_seconds() / 3600
            
            if hours_since_last_trade >= interval_hours:
                action = "buy"
                quantity = investment_per_interval / current_price
                reason = f"Scheduled DCA buy after {hours_since_last_trade:.1f} hours"
            else:
                action = "hold"
                quantity = 0
                reason = f"Next DCA buy in {interval_hours - hours_since_last_trade:.1f} hours"
            
            return {
                "strategy": "dca",
                "action": action,
                "quantity": quantity,
                "price": current_price,
                "investment_amount": investment_per_interval if action == "buy" else 0,
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logging.error(f"DCA strategy error: {e}")
            return {"error": str(e)}
    
    async def _execute_momentum_strategy(self, config: BotConfig, market_data: Dict) -> Dict:
        """Execute Momentum Trading Strategy"""
        try:
            current_price = market_data.get("price", 0)
            price_history = market_data.get("price_history", [current_price])
            
            if len(price_history) < 20:
                return {
                    "strategy": "momentum",
                    "action": "hold",
                    "reason": "Insufficient price history",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Calculate momentum indicators
            short_ma = np.mean(price_history[-5:])  # 5-period MA
            long_ma = np.mean(price_history[-20:])  # 20-period MA
            rsi = self._calculate_rsi(price_history)
            
            # Momentum signals
            ma_signal = short_ma > long_ma
            rsi_signal = 30 < rsi < 70  # Not overbought/oversold
            price_momentum = (current_price - price_history[-5]) / price_history[-5]
            
            # Decision logic
            if ma_signal and rsi_signal and price_momentum > 0.02:  # 2% momentum
                action = "buy"
                quantity = config.investment_amount / current_price
                reason = f"Strong upward momentum: {price_momentum:.2%}"
            elif not ma_signal or rsi > 70:
                action = "sell"
                quantity = config.investment_amount / current_price
                reason = f"Momentum weakening, RSI: {rsi:.1f}"
            else:
                action = "hold"
                quantity = 0
                reason = "No clear momentum signal"
            
            return {
                "strategy": "momentum",
                "action": action,
                "quantity": quantity,
                "price": current_price,
                "indicators": {
                    "short_ma": short_ma,
                    "long_ma": long_ma,
                    "rsi": rsi,
                    "momentum": price_momentum
                },
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Momentum strategy error: {e}")
            return {"error": str(e)}
    
    async def _execute_mean_reversion_strategy(self, config: BotConfig, market_data: Dict) -> Dict:
        """Execute Mean Reversion Strategy"""
        try:
            current_price = market_data.get("price", 0)
            price_history = market_data.get("price_history", [current_price])
            
            if len(price_history) < 20:
                return {
                    "strategy": "mean_reversion",
                    "action": "hold",
                    "reason": "Insufficient price history",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Calculate mean reversion indicators
            mean_price = np.mean(price_history[-20:])
            std_price = np.std(price_history[-20:])
            z_score = (current_price - mean_price) / std_price
            bollinger_upper = mean_price + (2 * std_price)
            bollinger_lower = mean_price - (2 * std_price)
            
            # Mean reversion signals
            if current_price < bollinger_lower and z_score < -1.5:
                action = "buy"
                quantity = config.investment_amount / current_price
                reason = f"Price below lower Bollinger band, Z-score: {z_score:.2f}"
            elif current_price > bollinger_upper and z_score > 1.5:
                action = "sell"
                quantity = config.investment_amount / current_price
                reason = f"Price above upper Bollinger band, Z-score: {z_score:.2f}"
            else:
                action = "hold"
                quantity = 0
                reason = f"Price within normal range, Z-score: {z_score:.2f}"
            
            return {
                "strategy": "mean_reversion",
                "action": action,
                "quantity": quantity,
                "price": current_price,
                "indicators": {
                    "mean_price": mean_price,
                    "z_score": z_score,
                    "bollinger_upper": bollinger_upper,
                    "bollinger_lower": bollinger_lower
                },
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Mean reversion strategy error: {e}")
            return {"error": str(e)}
    
    async def _execute_arbitrage_strategy(self, config: BotConfig, market_data: Dict) -> Dict:
        """Execute Arbitrage Strategy"""
        try:
            # Mock arbitrage opportunity detection
            current_price = market_data.get("price", 0)
            
            # Simulate price differences across exchanges
            exchange_prices = {
                "binance": current_price,
                "coinbase": current_price * np.random.uniform(0.998, 1.002),
                "kraken": current_price * np.random.uniform(0.997, 1.003),
                "ftx": current_price * np.random.uniform(0.999, 1.001)
            }
            
            # Find best arbitrage opportunity
            min_exchange = min(exchange_prices, key=exchange_prices.get)
            max_exchange = max(exchange_prices, key=exchange_prices.get)
            
            profit_percentage = ((exchange_prices[max_exchange] - exchange_prices[min_exchange]) / exchange_prices[min_exchange]) * 100
            
            # Execute if profitable (accounting for fees)
            if profit_percentage > 0.5:  # 0.5% minimum profit
                action = "arbitrage"
                quantity = config.investment_amount / exchange_prices[min_exchange]
                reason = f"Buy on {min_exchange}, sell on {max_exchange} for {profit_percentage:.2f}% profit"
            else:
                action = "hold"
                quantity = 0
                reason = f"No profitable arbitrage opportunity found ({profit_percentage:.2f}%)"
            
            return {
                "strategy": "arbitrage",
                "action": action,
                "quantity": quantity,
                "buy_exchange": min_exchange,
                "sell_exchange": max_exchange,
                "buy_price": exchange_prices[min_exchange],
                "sell_price": exchange_prices[max_exchange],
                "profit_percentage": profit_percentage,
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Arbitrage strategy error: {e}")
            return {"error": str(e)}
    
    async def _execute_copy_trading_strategy(self, config: BotConfig, market_data: Dict) -> Dict:
        """Execute Copy Trading Strategy"""
        try:
            # Mock copy trading from top performers
            top_traders = [
                {"name": "CryptoMaster", "last_action": "buy", "confidence": 0.85},
                {"name": "DiamondHands", "last_action": "hold", "confidence": 0.75},
                {"name": "MoonTrader", "last_action": "sell", "confidence": 0.90}
            ]
            
            # Select trader to copy based on confidence
            selected_trader = max(top_traders, key=lambda x: x["confidence"])
            
            current_price = market_data.get("price", 0)
            
            if selected_trader["last_action"] == "buy" and selected_trader["confidence"] > 0.8:
                action = "buy"
                quantity = config.investment_amount / current_price
                reason = f"Copying {selected_trader['name']}'s BUY signal (confidence: {selected_trader['confidence']:.0%})"
            elif selected_trader["last_action"] == "sell" and selected_trader["confidence"] > 0.8:
                action = "sell"
                quantity = config.investment_amount / current_price
                reason = f"Copying {selected_trader['name']}'s SELL signal (confidence: {selected_trader['confidence']:.0%})"
            else:
                action = "hold"
                quantity = 0
                reason = f"Copying {selected_trader['name']}'s HOLD position"
            
            return {
                "strategy": "copy_trading",
                "action": action,
                "quantity": quantity,
                "price": current_price,
                "copied_trader": selected_trader["name"],
                "trader_confidence": selected_trader["confidence"],
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Copy trading strategy error: {e}")
            return {"error": str(e)}
    
    def _calculate_rsi(self, prices: List[float], window: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < window + 1:
            return 50  # Neutral RSI
        
        deltas = np.diff(prices)
        seed = deltas[:window+1]
        up = seed[seed >= 0].sum() / window
        down = -seed[seed < 0].sum() / window
        
        if down == 0:
            return 100
        
        rs = up / down
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    async def _update_bot_performance(self, bot_id: str, trade_result: Dict):
        """Update bot performance metrics"""
        try:
            if bot_id not in self.bot_performances:
                return
            
            performance = self.bot_performances[bot_id]
            
            if trade_result.get("action") in ["buy", "sell", "arbitrage"]:
                performance.total_trades += 1
                performance.last_trade = datetime.utcnow()
                
                # Simulate trade outcome
                success_rate = 0.65  # 65% win rate
                if np.random.random() < success_rate:
                    performance.winning_trades += 1
                    profit = np.random.uniform(0.01, 0.05)  # 1-5% profit
                    performance.total_return += profit
                else:
                    loss = np.random.uniform(-0.02, -0.005)  # 0.5-2% loss
                    performance.total_return += loss
                
                # Update other metrics
                performance.sharpe_ratio = max(0, performance.total_return / max(0.01, abs(performance.max_drawdown)))
                
        except Exception as e:
            logging.error(f"Performance update error: {e}")
    
    async def get_all_bot_performances(self) -> Dict:
        """Get performance data for all bots"""
        try:
            performances = {}
            
            for bot_id, performance in self.bot_performances.items():
                bot_config = self.active_bots.get(bot_id)
                if bot_config:
                    performances[bot_id] = {
                        "bot_name": bot_config.name,
                        "strategy": bot_config.strategy.value,
                        "total_trades": performance.total_trades,
                        "winning_trades": performance.winning_trades,
                        "win_rate": (performance.winning_trades / max(1, performance.total_trades)) * 100,
                        "total_return": round(performance.total_return * 100, 2),
                        "sharpe_ratio": round(performance.sharpe_ratio, 2),
                        "current_position": performance.current_position,
                        "last_trade": performance.last_trade.isoformat() if performance.last_trade else None,
                        "active": bot_config.active
                    }
            
            return {
                "total_bots": len(performances),
                "active_bots": len([p for p in performances.values() if p["active"]]),
                "performances": performances,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Get performances error: {e}")
            return {"error": str(e)}

# Global trading bot engine
trading_bot_engine = TradingBotEngine()