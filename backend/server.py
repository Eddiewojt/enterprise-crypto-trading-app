from fastapi import FastAPI, APIRouter, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import json
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uuid
from datetime import datetime, timedelta
from binance.client import Client
import pandas as pd
import numpy as np
import threading
from binance.streams import BinanceSocketManager
import websocket
from contextlib import asynccontextmanager
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Global variables for websocket and data storage
active_connections: List[WebSocket] = []
current_price = 0.0
price_history = []
signals = []
is_websocket_running = False
portfolio_data = {}
multi_coin_data = {}

# Binance client setup
try:
    binance_client = Client(
        api_key=os.environ.get('BINANCE_API_KEY', ''),
        api_secret=os.environ.get('BINANCE_API_SECRET', ''),
        testnet=True  # Use testnet to avoid geo-restrictions
    )
    # Test connection
    binance_client.ping()
    BINANCE_AVAILABLE = True
    logging.info("Binance API connection successful")
except Exception as e:
    logging.warning(f"Binance API not available: {e}")
    binance_client = None
    BINANCE_AVAILABLE = False

# Create the main app
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Define Models
class PriceData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    price: float
    volume: float
    change_24h: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class TradingSignal(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    signal_type: str  # 'BUY' or 'SELL'
    strength: float  # 0-100
    indicators: Dict
    timeframe: str
    price: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class Alert(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    signal_id: str
    message: str
    is_read: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class Trade(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    is_paper_trade: bool = True

class Portfolio(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "default_user"
    symbol: str
    quantity: float
    avg_price: float
    total_invested: float
    current_value: float
    pnl: float
    pnl_percentage: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class TradeRequest(BaseModel):
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: Optional[float] = None  # If None, use current market price

# Advanced Technical Analysis Functions
def calculate_rsi(prices, window=14):
    """Calculate RSI (Relative Strength Index)"""
    deltas = np.diff(prices)
    seed = deltas[:window+1]
    up = seed[seed >= 0].sum()/window
    down = -seed[seed < 0].sum()/window
    rs = up/down if down != 0 else 0
    rsi = np.zeros_like(prices)
    rsi[:window] = 100. - 100./(1. + rs)
    
    for i in range(window, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        
        up = (up*(window-1) + upval)/window
        down = (down*(window-1) + downval)/window
        rs = up/down if down != 0 else 0
        rsi[i] = 100. - 100./(1. + rs)
    
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    exp1 = pd.Series(prices).ewm(span=fast).mean()
    exp2 = pd.Series(prices).ewm(span=slow).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd.values, signal_line.values, histogram.values

def calculate_moving_averages(prices, short_window=20, long_window=50):
    """Calculate Simple Moving Averages"""
    sma_short = pd.Series(prices).rolling(window=short_window).mean()
    sma_long = pd.Series(prices).rolling(window=long_window).mean()
    return sma_short.values, sma_long.values

def calculate_bollinger_bands(prices, window=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = pd.Series(prices).rolling(window=window).mean()
    std = pd.Series(prices).rolling(window=window).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band.values, sma.values, lower_band.values

def calculate_stochastic(high_prices, low_prices, close_prices, k_window=14, d_window=3):
    """Calculate Stochastic Oscillator"""
    high_series = pd.Series(high_prices)
    low_series = pd.Series(low_prices)
    close_series = pd.Series(close_prices)
    
    lowest_low = low_series.rolling(window=k_window).min()
    highest_high = high_series.rolling(window=k_window).max()
    
    k_percent = 100 * ((close_series - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    
    return k_percent.values, d_percent.values

def calculate_volume_indicators(prices, volumes):
    """Calculate Volume-based indicators"""
    # Volume Weighted Average Price (VWAP)
    vwap = np.cumsum(np.array(prices) * np.array(volumes)) / np.cumsum(volumes)
    
    # On-Balance Volume (OBV)
    obv = np.zeros(len(prices))
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            obv[i] = obv[i-1] + volumes[i]
        elif prices[i] < prices[i-1]:
            obv[i] = obv[i-1] - volumes[i]
        else:
            obv[i] = obv[i-1]
    
    return vwap, obv

def generate_advanced_signal(prices, volumes, high_prices, low_prices, timeframe='15m'):
    """Generate trading signal based on multiple advanced indicators"""
    if len(prices) < 50:
        return None
    
    # Calculate all indicators
    rsi = calculate_rsi(prices)
    macd, signal_line, histogram = calculate_macd(prices)
    sma_short, sma_long = calculate_moving_averages(prices)
    upper_band, middle_band, lower_band = calculate_bollinger_bands(prices)
    stoch_k, stoch_d = calculate_stochastic(high_prices, low_prices, prices)
    vwap, obv = calculate_volume_indicators(prices, volumes)
    
    current_price = prices[-1]
    current_rsi = rsi[-1]
    current_macd = macd[-1]
    current_signal = signal_line[-1]
    current_histogram = histogram[-1]
    current_sma_short = sma_short[-1]
    current_sma_long = sma_long[-1]
    current_upper_band = upper_band[-1]
    current_lower_band = lower_band[-1]
    current_stoch_k = stoch_k[-1]
    current_stoch_d = stoch_d[-1]
    current_vwap = vwap[-1]
    
    # Advanced signal scoring
    buy_signals = 0
    sell_signals = 0
    
    # RSI signals
    if current_rsi < 30:  # Oversold
        buy_signals += 1.5
    elif current_rsi > 70:  # Overbought
        sell_signals += 1.5
    
    # MACD signals
    if current_macd > current_signal and histogram[-1] > histogram[-2]:
        buy_signals += 1
    elif current_macd < current_signal and histogram[-1] < histogram[-2]:
        sell_signals += 1
    
    # Moving Average signals
    if current_sma_short > current_sma_long and current_price > current_sma_short:
        buy_signals += 1
    elif current_sma_short < current_sma_long and current_price < current_sma_short:
        sell_signals += 1
    
    # Bollinger Bands signals
    if current_price < current_lower_band:  # Price below lower band
        buy_signals += 1
    elif current_price > current_upper_band:  # Price above upper band
        sell_signals += 1
    
    # Stochastic signals
    if current_stoch_k < 20 and current_stoch_d < 20:  # Oversold
        buy_signals += 0.5
    elif current_stoch_k > 80 and current_stoch_d > 80:  # Overbought
        sell_signals += 0.5
    
    # Volume signals
    if current_price > current_vwap:  # Price above VWAP
        buy_signals += 0.5
    elif current_price < current_vwap:  # Price below VWAP
        sell_signals += 0.5
    
    # Volume trend
    if len(volumes) >= 5:
        recent_volume_avg = np.mean(volumes[-5:])
        if volumes[-1] > recent_volume_avg * 1.5:  # High volume
            if current_price > prices[-2]:
                buy_signals += 0.5
            else:
                sell_signals += 0.5
    
    # Determine signal
    total_signals = buy_signals + sell_signals
    if total_signals > 0:
        if buy_signals > sell_signals:
            signal_type = 'BUY'
            strength = min(100, (buy_signals / total_signals) * 100)
        else:
            signal_type = 'SELL'
            strength = min(100, (sell_signals / total_signals) * 100)
        
        # Only return strong signals
        if strength >= 65:  # Increased threshold for advanced signals
            return TradingSignal(
                symbol='DOGEUSDT',
                signal_type=signal_type,
                strength=strength,
                indicators={
                    'rsi': float(current_rsi),
                    'macd': float(current_macd),
                    'signal_line': float(current_signal),
                    'sma_short': float(current_sma_short),
                    'sma_long': float(current_sma_long),
                    'bollinger_upper': float(current_upper_band),
                    'bollinger_lower': float(current_lower_band),
                    'stochastic_k': float(current_stoch_k),
                    'stochastic_d': float(current_stoch_d),
                    'vwap': float(current_vwap)
                },
                timeframe=timeframe,
                price=current_price
            )
    
    return None

# Portfolio Management Functions
async def update_portfolio(symbol: str, side: str, quantity: float, price: float):
    """Update portfolio after a trade"""
    global portfolio_data
    
    if symbol not in portfolio_data:
        portfolio_data[symbol] = {
            'quantity': 0,
            'avg_price': 0,
            'total_invested': 0
        }
    
    portfolio = portfolio_data[symbol]
    
    if side == 'BUY':
        # Update for buy order
        total_cost = quantity * price
        new_quantity = portfolio['quantity'] + quantity
        new_total_invested = portfolio['total_invested'] + total_cost
        
        if new_quantity > 0:
            portfolio['avg_price'] = new_total_invested / new_quantity
        
        portfolio['quantity'] = new_quantity
        portfolio['total_invested'] = new_total_invested
        
    elif side == 'SELL':
        # Update for sell order
        if portfolio['quantity'] >= quantity:
            portfolio['quantity'] -= quantity
            # Reduce total invested proportionally
            portfolio['total_invested'] *= (portfolio['quantity'] / (portfolio['quantity'] + quantity))
        else:
            raise ValueError("Insufficient quantity to sell")
    
    # Save to database
    await save_portfolio_to_db(symbol, portfolio, price)

async def save_portfolio_to_db(symbol: str, portfolio_data: dict, current_price: float):
    """Save portfolio data to MongoDB"""
    current_value = portfolio_data['quantity'] * current_price
    pnl = current_value - portfolio_data['total_invested']
    pnl_percentage = (pnl / portfolio_data['total_invested'] * 100) if portfolio_data['total_invested'] > 0 else 0
    
    portfolio_record = Portfolio(
        symbol=symbol,
        quantity=portfolio_data['quantity'],
        avg_price=portfolio_data['avg_price'],
        total_invested=portfolio_data['total_invested'],
        current_value=current_value,
        pnl=pnl,
        pnl_percentage=pnl_percentage
    )
    
    await db.portfolio.replace_one(
        {"symbol": symbol, "user_id": "default_user"},
        portfolio_record.dict(),
        upsert=True
    )

# Email notification function
async def send_email_notification(subject: str, message: str):
    """Send email notification"""
    try:
        # Using Gmail SMTP (you can change this to your preferred email service)
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        
        # You'll need to set these in your .env file
        sender_email = os.environ.get('SMTP_EMAIL', 'your_email@gmail.com')
        sender_password = os.environ.get('SMTP_PASSWORD', 'your_app_password')
        recipient_email = "eddiewojt1@gmail.com"
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(message, 'plain'))
        
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        
        logging.info(f"Email sent successfully to {recipient_email}")
        
    except Exception as e:
        logging.error(f"Failed to send email: {e}")

# WebSocket connection manager
async def connect_websocket(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)

async def disconnect_websocket(websocket: WebSocket):
    if websocket in active_connections:
        active_connections.remove(websocket)

async def broadcast_to_all(message: dict):
    """Broadcast message to all connected websockets"""
    if active_connections:
        for connection in active_connections[:]:
            try:
                await connection.send_text(json.dumps(message))
            except:
                active_connections.remove(connection)

# Multi-coin WebSocket handler
def handle_socket_message(msg):
    """Handle incoming Binance WebSocket messages"""
    global current_price, price_history, multi_coin_data
    
    if msg['e'] == '24hrTicker':
        symbol = msg['s']
        current_price = float(msg['c'])
        
        # Initialize multi-coin data if not exists
        if symbol not in multi_coin_data:
            multi_coin_data[symbol] = {
                'prices': [],
                'volumes': [],
                'highs': [],
                'lows': []
            }
        
        # Store data for the symbol
        coin_data = multi_coin_data[symbol]
        coin_data['prices'].append(current_price)
        coin_data['volumes'].append(float(msg['v']))
        coin_data['highs'].append(float(msg['h']))
        coin_data['lows'].append(float(msg['l']))
        
        # Keep only last 200 data points
        if len(coin_data['prices']) > 200:
            coin_data['prices'].pop(0)
            coin_data['volumes'].pop(0)
            coin_data['highs'].pop(0)
            coin_data['lows'].pop(0)
        
        price_data = {
            'symbol': symbol,
            'price': current_price,
            'volume': float(msg['v']),
            'change_24h': float(msg['P']),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # For DOGE, also update the main price history
        if symbol == 'DOGEUSDT':
            price_history.append(current_price)
            if len(price_history) > 200:
                price_history.pop(0)
        
        # Generate advanced signals when we have enough data
        if len(coin_data['prices']) >= 50:
            signal = generate_advanced_signal(
                coin_data['prices'], 
                coin_data['volumes'], 
                coin_data['highs'], 
                coin_data['lows']
            )
            if signal:
                signals.append(signal)
                if len(signals) > 50:
                    signals.pop(0)
                
                # Send email notification for strong signals
                if signal.strength >= 80:
                    asyncio.create_task(send_email_notification(
                        f"Strong {signal.signal_type} Signal - {symbol}",
                        f"Strong {signal.signal_type} signal detected for {symbol}\n"
                        f"Strength: {signal.strength:.1f}%\n"
                        f"Price: ${signal.price:.6f}\n"
                        f"Timeframe: {signal.timeframe}\n"
                        f"RSI: {signal.indicators['rsi']:.1f}\n"
                        f"MACD: {signal.indicators['macd']:.6f}\n"
                        f"Time: {signal.timestamp}"
                    ))
                
                # Broadcast signal to all connected clients
                asyncio.create_task(broadcast_to_all({
                    'type': 'signal',
                    'data': signal.dict()
                }))
        
        # Broadcast price update to all connected clients
        asyncio.create_task(broadcast_to_all({
            'type': 'price',
            'data': price_data
        }))

def start_binance_websocket():
    """Start Binance WebSocket in a separate thread"""
    global is_websocket_running
    
    if not is_websocket_running:
        is_websocket_running = True
        
        # Multi-coin symbols to track
        symbols_to_track = ['DOGEUSDT', 'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT']
        
        if BINANCE_AVAILABLE and binance_client:
            async def websocket_coroutine():
                try:
                    bm = BinanceSocketManager(binance_client)
                    # Subscribe to multiple symbols
                    for symbol in symbols_to_track:
                        ts = bm.symbol_ticker_socket(symbol)
                        
                        async with ts as tscm:
                            while True:
                                res = await tscm.recv()
                                handle_socket_message(res)
                                
                except Exception as e:
                    logging.error(f"WebSocket error: {e}")
                    global is_websocket_running
                    is_websocket_running = False
            
            def websocket_thread():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(websocket_coroutine())
                except Exception as e:
                    logging.error(f"WebSocket thread error: {e}")
                    global is_websocket_running
                    is_websocket_running = False
        else:
            # Mock WebSocket data for multiple coins
            def mock_websocket_thread():
                import random
                import time
                
                mock_prices = {
                    'DOGEUSDT': 0.08234,
                    'BTCUSDT': 43000,
                    'ETHUSDT': 2600,
                    'ADAUSDT': 0.45,
                    'BNBUSDT': 320
                }
                
                try:
                    while True:
                        for symbol, base_price in mock_prices.items():
                            price_variation = random.uniform(-0.02, 0.02)
                            current_price = base_price * (1 + price_variation)
                            
                            mock_message = {
                                'e': '24hrTicker',
                                's': symbol,
                                'c': str(current_price),
                                'v': str(random.uniform(100000, 1000000)),
                                'P': str(random.uniform(-5.0, 5.0)),
                                'h': str(current_price * 1.02),
                                'l': str(current_price * 0.98)
                            }
                            
                            handle_socket_message(mock_message)
                        
                        time.sleep(3)  # Update every 3 seconds
                        
                except Exception as e:
                    logging.error(f"Mock WebSocket error: {e}")
                    global is_websocket_running
                    is_websocket_running = False
            
            websocket_thread = mock_websocket_thread
        
        thread = threading.Thread(target=websocket_thread, daemon=True)
        thread.start()

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Advanced DOGE Trading App API"}

@api_router.get("/multi-coin/prices")
async def get_multi_coin_prices():
    """Get current prices for multiple coins"""
    try:
        symbols = ['DOGEUSDT', 'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT']
        prices = {}
        
        if BINANCE_AVAILABLE and binance_client:
            for symbol in symbols:
                try:
                    ticker = binance_client.get_symbol_ticker(symbol=symbol)
                    price_24h = binance_client.get_ticker(symbol=symbol)
                    
                    prices[symbol] = {
                        "symbol": symbol,
                        "price": float(ticker['price']),
                        "change_24h": float(price_24h['priceChangePercent']),
                        "volume": float(price_24h['volume']),
                        "high_24h": float(price_24h['highPrice']),
                        "low_24h": float(price_24h['lowPrice']),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                except:
                    continue
        else:
            # Mock data for multiple coins
            import random
            mock_prices = {
                'DOGEUSDT': 0.08234,
                'BTCUSDT': 43000,
                'ETHUSDT': 2600,
                'ADAUSDT': 0.45,
                'BNBUSDT': 320
            }
            
            for symbol, base_price in mock_prices.items():
                price_variation = random.uniform(-0.02, 0.02)
                current_price = base_price * (1 + price_variation)
                
                prices[symbol] = {
                    "symbol": symbol,
                    "price": round(current_price, 6 if symbol == 'DOGEUSDT' else 2),
                    "change_24h": round(random.uniform(-5.0, 5.0), 2),
                    "volume": round(random.uniform(100000, 1000000), 2),
                    "high_24h": round(current_price * 1.02, 6 if symbol == 'DOGEUSDT' else 2),
                    "low_24h": round(current_price * 0.98, 6 if symbol == 'DOGEUSDT' else 2),
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        return prices
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching multi-coin prices: {str(e)}")

@api_router.get("/doge/price")
async def get_doge_price():
    """Get current DOGE price"""
    try:
        if BINANCE_AVAILABLE and binance_client:
            ticker = binance_client.get_symbol_ticker(symbol="DOGEUSDT")
            price_24h = binance_client.get_ticker(symbol="DOGEUSDT")
            
            return {
                "symbol": "DOGEUSDT",
                "price": float(ticker['price']),
                "change_24h": float(price_24h['priceChangePercent']),
                "volume": float(price_24h['volume']),
                "high_24h": float(price_24h['highPrice']),
                "low_24h": float(price_24h['lowPrice']),
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            # Mock data when Binance is not available
            import random
            base_price = 0.08234
            price_variation = random.uniform(-0.002, 0.002)
            current_price = base_price + price_variation
            
            return {
                "symbol": "DOGEUSDT",
                "price": round(current_price, 6),
                "change_24h": round(random.uniform(-5.0, 5.0), 2),
                "volume": round(random.uniform(1000000, 5000000), 2),
                "high_24h": round(current_price * 1.02, 6),
                "low_24h": round(current_price * 0.98, 6),
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching price: {str(e)}")

@api_router.get("/doge/klines")
async def get_doge_klines(timeframe: str = "15m", limit: int = 100):
    """Get DOGE candlestick data for advanced charting"""
    try:
        if BINANCE_AVAILABLE and binance_client:
            klines = binance_client.get_klines(
                symbol="DOGEUSDT",
                interval=timeframe,
                limit=limit
            )
            
            formatted_klines = []
            for kline in klines:
                formatted_klines.append({
                    "timestamp": kline[0],
                    "open": float(kline[1]),
                    "high": float(kline[2]),
                    "low": float(kline[3]),
                    "close": float(kline[4]),
                    "volume": float(kline[5])
                })
            
            return formatted_klines
        else:
            # Mock candlestick data
            import random
            from datetime import timedelta
            
            formatted_klines = []
            base_price = 0.08234
            current_time = datetime.utcnow()
            
            # Generate mock klines
            for i in range(limit):
                timestamp = int((current_time - timedelta(minutes=15*i)).timestamp() * 1000)
                price_variation = random.uniform(-0.001, 0.001)
                open_price = base_price + price_variation
                close_price = open_price + random.uniform(-0.0005, 0.0005)
                high_price = max(open_price, close_price) + random.uniform(0, 0.0002)
                low_price = min(open_price, close_price) - random.uniform(0, 0.0002)
                
                formatted_klines.append({
                    "timestamp": timestamp,
                    "open": round(open_price, 6),
                    "high": round(high_price, 6),
                    "low": round(low_price, 6),
                    "close": round(close_price, 6),
                    "volume": round(random.uniform(10000, 50000), 2)
                })
            
            return list(reversed(formatted_klines))  # Return in chronological order
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching klines: {str(e)}")

@api_router.get("/doge/signals")
async def get_trading_signals():
    """Get recent trading signals"""
    return signals[-10:]  # Return last 10 signals

@api_router.get("/doge/analysis")
async def get_advanced_analysis(timeframe: str = "15m"):
    """Get advanced technical analysis for DOGE"""
    try:
        if BINANCE_AVAILABLE and binance_client:
            # Get historical data from Binance
            klines = binance_client.get_klines(
                symbol="DOGEUSDT",
                interval=timeframe,
                limit=100
            )
            prices = [float(kline[4]) for kline in klines]  # Close prices
            volumes = [float(kline[5]) for kline in klines]  # Volumes
            highs = [float(kline[2]) for kline in klines]  # High prices
            lows = [float(kline[3]) for kline in klines]  # Low prices
        else:
            # Use mock data
            if 'DOGEUSDT' in multi_coin_data and len(multi_coin_data['DOGEUSDT']['prices']) > 0:
                prices = multi_coin_data['DOGEUSDT']['prices']
                volumes = multi_coin_data['DOGEUSDT']['volumes']
                highs = multi_coin_data['DOGEUSDT']['highs']
                lows = multi_coin_data['DOGEUSDT']['lows']
            else:
                import random
                prices = [0.08234 + random.uniform(-0.001, 0.001) for _ in range(100)]
                volumes = [random.uniform(10000, 50000) for _ in range(100)]
                highs = [p + random.uniform(0, 0.0002) for p in prices]
                lows = [p - random.uniform(0, 0.0002) for p in prices]
        
        if len(prices) < 50:
            return {"error": "Not enough data for advanced analysis"}
        
        # Calculate all indicators
        rsi = calculate_rsi(prices)
        macd, signal_line, histogram = calculate_macd(prices)
        sma_short, sma_long = calculate_moving_averages(prices)
        upper_band, middle_band, lower_band = calculate_bollinger_bands(prices)
        stoch_k, stoch_d = calculate_stochastic(highs, lows, prices)
        vwap, obv = calculate_volume_indicators(prices, volumes)
        
        return {
            "symbol": "DOGEUSDT",
            "timeframe": timeframe,
            "current_price": prices[-1],
            "indicators": {
                "rsi": {
                    "value": float(rsi[-1]),
                    "signal": "oversold" if rsi[-1] < 30 else "overbought" if rsi[-1] > 70 else "neutral"
                },
                "macd": {
                    "macd": float(macd[-1]),
                    "signal": float(signal_line[-1]),
                    "histogram": float(histogram[-1]),
                    "signal_type": "bullish" if macd[-1] > signal_line[-1] else "bearish"
                },
                "moving_averages": {
                    "sma_short": float(sma_short[-1]),
                    "sma_long": float(sma_long[-1]),
                    "signal": "bullish" if sma_short[-1] > sma_long[-1] else "bearish"
                },
                "bollinger_bands": {
                    "upper": float(upper_band[-1]),
                    "middle": float(middle_band[-1]),
                    "lower": float(lower_band[-1]),
                    "position": "above_upper" if prices[-1] > upper_band[-1] else "below_lower" if prices[-1] < lower_band[-1] else "middle"
                },
                "stochastic": {
                    "k": float(stoch_k[-1]),
                    "d": float(stoch_d[-1]),
                    "signal": "oversold" if stoch_k[-1] < 20 else "overbought" if stoch_k[-1] > 80 else "neutral"
                },
                "volume": {
                    "vwap": float(vwap[-1]),
                    "obv": float(obv[-1]),
                    "price_vs_vwap": "above" if prices[-1] > vwap[-1] else "below"
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in technical analysis: {str(e)}")

@api_router.post("/portfolio/trade")
async def execute_paper_trade(trade_request: TradeRequest):
    """Execute a paper trade"""
    try:
        # Use current market price if not specified
        if trade_request.price is None:
            if BINANCE_AVAILABLE and binance_client:
                ticker = binance_client.get_symbol_ticker(symbol=trade_request.symbol)
                trade_request.price = float(ticker['price'])
            else:
                # Use mock price
                mock_prices = {
                    'DOGEUSDT': 0.08234,
                    'BTCUSDT': 43000,
                    'ETHUSDT': 2600,
                    'ADAUSDT': 0.45,
                    'BNBUSDT': 320
                }
                trade_request.price = mock_prices.get(trade_request.symbol, 1.0)
        
        # Create trade record
        trade = Trade(
            symbol=trade_request.symbol,
            side=trade_request.side,
            quantity=trade_request.quantity,
            price=trade_request.price,
            is_paper_trade=True
        )
        
        # Save trade to database
        await db.trades.insert_one(trade.dict())
        
        # Update portfolio
        await update_portfolio(
            trade_request.symbol,
            trade_request.side,
            trade_request.quantity,
            trade_request.price
        )
        
        return {
            "trade_id": trade.id,
            "symbol": trade.symbol,
            "side": trade.side,
            "quantity": trade.quantity,
            "price": trade.price,
            "total_value": trade.quantity * trade.price,
            "timestamp": trade.timestamp,
            "message": f"Paper trade executed successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing trade: {str(e)}")

@api_router.get("/portfolio")
async def get_portfolio():
    """Get current portfolio"""
    try:
        portfolio_cursor = db.portfolio.find({"user_id": "default_user"})
        portfolio_list = await portfolio_cursor.to_list(length=None)
        
        total_invested = 0
        total_current_value = 0
        total_pnl = 0
        
        for item in portfolio_list:
            total_invested += item.get('total_invested', 0)
            total_current_value += item.get('current_value', 0)
            total_pnl += item.get('pnl', 0)
        
        return {
            "holdings": portfolio_list,
            "summary": {
                "total_invested": total_invested,
                "total_current_value": total_current_value,
                "total_pnl": total_pnl,
                "total_pnl_percentage": (total_pnl / total_invested * 100) if total_invested > 0 else 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching portfolio: {str(e)}")

@api_router.get("/portfolio/trades")
async def get_trade_history():
    """Get trade history"""
    try:
        trades_cursor = db.trades.find().sort("timestamp", -1).limit(50)
        trades_list = await trades_cursor.to_list(length=None)
        
        return {
            "trades": trades_list,
            "total_trades": len(trades_list)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching trade history: {str(e)}")

@api_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await connect_websocket(websocket)
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        await disconnect_websocket(websocket)

# Background task to start WebSocket
async def startup_event():
    """Start background tasks"""
    start_binance_websocket()

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup():
    await startup_event()

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()