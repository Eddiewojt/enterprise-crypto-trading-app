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

# Technical Analysis Functions
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

def generate_trading_signal(prices, timeframe='15m'):
    """Generate trading signal based on multiple indicators"""
    if len(prices) < 50:
        return None
    
    # Calculate indicators
    rsi = calculate_rsi(prices)
    macd, signal_line, histogram = calculate_macd(prices)
    sma_short, sma_long = calculate_moving_averages(prices)
    
    current_rsi = rsi[-1]
    current_macd = macd[-1]
    current_signal = signal_line[-1]
    current_histogram = histogram[-1]
    current_sma_short = sma_short[-1]
    current_sma_long = sma_long[-1]
    current_price = prices[-1]
    
    # Signal scoring
    buy_signals = 0
    sell_signals = 0
    
    # RSI signals
    if current_rsi < 30:  # Oversold
        buy_signals += 1
    elif current_rsi > 70:  # Overbought
        sell_signals += 1
    
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
    
    # Volume analysis (simplified)
    if len(prices) >= 5:
        recent_avg = np.mean(prices[-5:])
        if current_price > recent_avg * 1.01:  # 1% increase
            buy_signals += 0.5
        elif current_price < recent_avg * 0.99:  # 1% decrease
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
        if strength >= 60:
            return TradingSignal(
                symbol='DOGEUSDT',
                signal_type=signal_type,
                strength=strength,
                indicators={
                    'rsi': float(current_rsi),
                    'macd': float(current_macd),
                    'signal_line': float(current_signal),
                    'sma_short': float(current_sma_short),
                    'sma_long': float(current_sma_long)
                },
                timeframe=timeframe,
                price=current_price
            )
    
    return None

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

# Binance WebSocket handler
def handle_socket_message(msg):
    """Handle incoming Binance WebSocket messages"""
    global current_price, price_history
    
    if msg['e'] == '24hrTicker':
        current_price = float(msg['c'])
        price_data = {
            'symbol': msg['s'],
            'price': current_price,
            'volume': float(msg['v']),
            'change_24h': float(msg['P']),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Store price history
        price_history.append(current_price)
        if len(price_history) > 200:  # Keep last 200 prices
            price_history.pop(0)
        
        # Generate signals when we have enough data
        if len(price_history) >= 50:
            signal = generate_trading_signal(price_history)
            if signal:
                signals.append(signal)
                if len(signals) > 50:  # Keep last 50 signals
                    signals.pop(0)
                
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
        
        if BINANCE_AVAILABLE and binance_client:
            async def websocket_coroutine():
                try:
                    bm = BinanceSocketManager(binance_client)
                    ts = bm.symbol_ticker_socket('DOGEUSDT')
                    
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
            # Mock WebSocket data when Binance is not available
            def mock_websocket_thread():
                import random
                import time
                
                try:
                    while True:
                        # Generate mock ticker data
                        base_price = 0.08234
                        price_variation = random.uniform(-0.002, 0.002)
                        current_price = base_price + price_variation
                        
                        mock_message = {
                            'e': '24hrTicker',
                            's': 'DOGEUSDT',
                            'c': str(current_price),
                            'v': str(random.uniform(1000000, 5000000)),
                            'P': str(random.uniform(-5.0, 5.0))
                        }
                        
                        handle_socket_message(mock_message)
                        time.sleep(2)  # Update every 2 seconds
                        
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
    return {"message": "DOGE Trading App API"}

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
    """Get DOGE candlestick data"""
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
async def get_technical_analysis(timeframe: str = "15m"):
    """Get technical analysis for DOGE"""
    try:
        if BINANCE_AVAILABLE and binance_client:
            # Get historical data from Binance
            klines = binance_client.get_klines(
                symbol="DOGEUSDT",
                interval=timeframe,
                limit=100
            )
            prices = [float(kline[4]) for kline in klines]  # Close prices
        else:
            # Generate mock price data for analysis
            import random
            base_price = 0.08234
            prices = []
            for i in range(100):
                price_variation = random.uniform(-0.001, 0.001)
                price = base_price + price_variation + (random.uniform(-0.0001, 0.0001) * i)
                prices.append(price)
        
        if len(prices) < 50:
            return {"error": "Not enough data for analysis"}
        
        # Calculate indicators
        rsi = calculate_rsi(prices)
        macd, signal_line, histogram = calculate_macd(prices)
        sma_short, sma_long = calculate_moving_averages(prices)
        
        return {
            "symbol": "DOGEUSDT",
            "timeframe": timeframe,
            "current_price": prices[-1],
            "rsi": float(rsi[-1]),
            "macd": float(macd[-1]),
            "signal_line": float(signal_line[-1]),
            "sma_short": float(sma_short[-1]),
            "sma_long": float(sma_long[-1]),
            "analysis": {
                "rsi_signal": "oversold" if rsi[-1] < 30 else "overbought" if rsi[-1] > 70 else "neutral",
                "macd_signal": "bullish" if macd[-1] > signal_line[-1] else "bearish",
                "ma_signal": "bullish" if sma_short[-1] > sma_long[-1] else "bearish"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in technical analysis: {str(e)}")

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