import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Helper functions
const formatPrice = (price) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 4,
    maximumFractionDigits: 6
  }).format(price);
};

const formatPercentage = (percentage) => {
  return `${percentage >= 0 ? '+' : ''}${percentage.toFixed(2)}%`;
};

const formatNumber = (num) => {
  return new Intl.NumberFormat('en-US').format(num);
};

// Multi-Coin Price Card Component
const MultiCoinPriceCard = ({ multiCoinData, onSelectCoin }) => {
  return (
    <div className="multi-coin-card">
      <h3>Cryptocurrency Prices</h3>
      <div className="coin-grid">
        {Object.entries(multiCoinData).map(([symbol, data]) => {
          const isPositive = data.change_24h >= 0;
          const coinName = symbol.replace('USDT', '');
          
          return (
            <div 
              key={symbol} 
              className={`coin-item ${isPositive ? 'positive' : 'negative'}`}
              onClick={() => onSelectCoin(symbol)}
            >
              <div className="coin-header">
                <span className="coin-symbol">{coinName}</span>
                <span className="coin-change">{formatPercentage(data.change_24h)}</span>
              </div>
              <div className="coin-price">
                {formatPrice(data.price)}
              </div>
              <div className="coin-volume">
                Vol: {formatNumber(data.volume)}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

// Advanced Candlestick Chart Component
const CandlestickChart = ({ klineData, symbol }) => {
  const canvasRef = useRef(null);
  
  useEffect(() => {
    if (!klineData || klineData.length === 0) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const { width, height } = canvas;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Calculate price range
    const prices = klineData.flatMap(k => [k.high, k.low]);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice;
    
    // Chart dimensions
    const chartWidth = width - 80;
    const chartHeight = height - 60;
    const chartX = 40;
    const chartY = 30;
    
    // Draw background
    ctx.fillStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.fillRect(chartX, chartY, chartWidth, chartHeight);
    
    // Draw grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    
    // Horizontal grid lines
    for (let i = 0; i <= 5; i++) {
      const y = chartY + (chartHeight / 5) * i;
      ctx.beginPath();
      ctx.moveTo(chartX, y);
      ctx.lineTo(chartX + chartWidth, y);
      ctx.stroke();
      
      // Price labels
      const price = maxPrice - (priceRange / 5) * i;
      ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
      ctx.font = '10px Arial';
      ctx.fillText(price.toFixed(6), 5, y + 3);
    }
    
    // Vertical grid lines
    for (let i = 0; i <= 10; i++) {
      const x = chartX + (chartWidth / 10) * i;
      ctx.beginPath();
      ctx.moveTo(x, chartY);
      ctx.lineTo(x, chartY + chartHeight);
      ctx.stroke();
    }
    
    // Draw candlesticks
    const candleWidth = chartWidth / klineData.length * 0.8;
    
    klineData.forEach((candle, index) => {
      const x = chartX + (chartWidth / klineData.length) * index + (chartWidth / klineData.length - candleWidth) / 2;
      const highY = chartY + ((maxPrice - candle.high) / priceRange) * chartHeight;
      const lowY = chartY + ((maxPrice - candle.low) / priceRange) * chartHeight;
      const openY = chartY + ((maxPrice - candle.open) / priceRange) * chartHeight;
      const closeY = chartY + ((maxPrice - candle.close) / priceRange) * chartHeight;
      
      const isBullish = candle.close > candle.open;
      const color = isBullish ? '#4ade80' : '#f87171';
      
      // Draw wick
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x + candleWidth / 2, highY);
      ctx.lineTo(x + candleWidth / 2, lowY);
      ctx.stroke();
      
      // Draw body
      ctx.fillStyle = color;
      const bodyTop = Math.min(openY, closeY);
      const bodyHeight = Math.abs(closeY - openY);
      ctx.fillRect(x, bodyTop, candleWidth, bodyHeight || 1);
    });
    
    // Draw title
    ctx.fillStyle = 'white';
    ctx.font = 'bold 14px Arial';
    ctx.fillText(`${symbol} Candlestick Chart`, chartX, 20);
    
  }, [klineData, symbol]);
  
  return (
    <div className="candlestick-chart">
      <canvas 
        ref={canvasRef} 
        width={400} 
        height={300}
        style={{ width: '100%', height: '300px' }}
      />
    </div>
  );
};

// Advanced Technical Analysis Component
const AdvancedTechnicalAnalysis = ({ analysis, selectedSymbol }) => {
  if (!analysis || !analysis.indicators) return null;
  
  const { indicators } = analysis;
  
  const getIndicatorColor = (signal) => {
    if (signal.includes('bullish') || signal.includes('oversold') || signal === 'above') return '#4ade80';
    if (signal.includes('bearish') || signal.includes('overbought') || signal === 'below') return '#f87171';
    return '#fbbf24';
  };
  
  return (
    <div className="advanced-analysis">
      <h3>Advanced Technical Analysis - {selectedSymbol}</h3>
      
      <div className="analysis-grid">
        <div className="indicator-card">
          <h4>RSI (14)</h4>
          <div className="indicator-value" style={{ color: getIndicatorColor(indicators.rsi.signal) }}>
            {indicators.rsi.value.toFixed(1)}
          </div>
          <div className="indicator-signal">{indicators.rsi.signal}</div>
        </div>
        
        <div className="indicator-card">
          <h4>MACD</h4>
          <div className="indicator-value">
            {indicators.macd.macd.toFixed(6)}
          </div>
          <div className="indicator-signal" style={{ color: getIndicatorColor(indicators.macd.signal_type) }}>
            {indicators.macd.signal_type}
          </div>
        </div>
        
        <div className="indicator-card">
          <h4>Moving Averages</h4>
          <div className="indicator-details">
            <div>Short: {indicators.moving_averages.sma_short.toFixed(6)}</div>
            <div>Long: {indicators.moving_averages.sma_long.toFixed(6)}</div>
          </div>
          <div className="indicator-signal" style={{ color: getIndicatorColor(indicators.moving_averages.signal) }}>
            {indicators.moving_averages.signal}
          </div>
        </div>
        
        <div className="indicator-card">
          <h4>Bollinger Bands</h4>
          <div className="indicator-details">
            <div>Upper: {indicators.bollinger_bands.upper.toFixed(6)}</div>
            <div>Lower: {indicators.bollinger_bands.lower.toFixed(6)}</div>
          </div>
          <div className="indicator-signal">
            {indicators.bollinger_bands.position}
          </div>
        </div>
        
        <div className="indicator-card">
          <h4>Stochastic</h4>
          <div className="indicator-details">
            <div>%K: {indicators.stochastic.k.toFixed(1)}</div>
            <div>%D: {indicators.stochastic.d.toFixed(1)}</div>
          </div>
          <div className="indicator-signal" style={{ color: getIndicatorColor(indicators.stochastic.signal) }}>
            {indicators.stochastic.signal}
          </div>
        </div>
        
        <div className="indicator-card">
          <h4>Volume Analysis</h4>
          <div className="indicator-details">
            <div>VWAP: {indicators.volume.vwap.toFixed(6)}</div>
            <div>OBV: {formatNumber(indicators.volume.obv)}</div>
          </div>
          <div className="indicator-signal" style={{ color: getIndicatorColor(indicators.volume.price_vs_vwap) }}>
            Price {indicators.volume.price_vs_vwap} VWAP
          </div>
        </div>
      </div>
    </div>
  );
};

// Portfolio Component
const Portfolio = ({ portfolio, onExecuteTrade }) => {
  const [tradeForm, setTradeForm] = useState({
    symbol: 'DOGEUSDT',
    side: 'BUY',
    quantity: 1000,
    price: ''
  });
  
  const handleTrade = async () => {
    try {
      await onExecuteTrade(tradeForm);
      setTradeForm({ ...tradeForm, quantity: 1000, price: '' });
    } catch (error) {
      console.error('Trade failed:', error);
    }
  };
  
  return (
    <div className="portfolio-container">
      <h3>Portfolio Management</h3>
      
      {/* Paper Trading Form */}
      <div className="trade-form">
        <h4>Paper Trading</h4>
        <div className="form-row">
          <select 
            value={tradeForm.symbol} 
            onChange={(e) => setTradeForm({ ...tradeForm, symbol: e.target.value })}
          >
            <option value="DOGEUSDT">DOGE</option>
            <option value="BTCUSDT">BTC</option>
            <option value="ETHUSDT">ETH</option>
            <option value="ADAUSDT">ADA</option>
            <option value="BNBUSDT">BNB</option>
          </select>
          
          <select 
            value={tradeForm.side} 
            onChange={(e) => setTradeForm({ ...tradeForm, side: e.target.value })}
          >
            <option value="BUY">BUY</option>
            <option value="SELL">SELL</option>
          </select>
        </div>
        
        <div className="form-row">
          <input
            type="number"
            placeholder="Quantity"
            value={tradeForm.quantity}
            onChange={(e) => setTradeForm({ ...tradeForm, quantity: parseFloat(e.target.value) })}
          />
          
          <input
            type="number"
            placeholder="Price (leave empty for market)"
            value={tradeForm.price}
            onChange={(e) => setTradeForm({ ...tradeForm, price: e.target.value })}
          />
        </div>
        
        <button className={`trade-button ${tradeForm.side.toLowerCase()}`} onClick={handleTrade}>
          Execute {tradeForm.side} Order
        </button>
      </div>
      
      {/* Portfolio Summary */}
      {portfolio.summary && (
        <div className="portfolio-summary">
          <h4>Portfolio Summary</h4>
          <div className="summary-grid">
            <div className="summary-item">
              <span className="label">Total Invested</span>
              <span className="value">{formatPrice(portfolio.summary.total_invested)}</span>
            </div>
            <div className="summary-item">
              <span className="label">Current Value</span>
              <span className="value">{formatPrice(portfolio.summary.total_current_value)}</span>
            </div>
            <div className="summary-item">
              <span className="label">P&L</span>
              <span className={`value ${portfolio.summary.total_pnl >= 0 ? 'positive' : 'negative'}`}>
                {formatPrice(portfolio.summary.total_pnl)}
              </span>
            </div>
            <div className="summary-item">
              <span className="label">P&L %</span>
              <span className={`value ${portfolio.summary.total_pnl_percentage >= 0 ? 'positive' : 'negative'}`}>
                {formatPercentage(portfolio.summary.total_pnl_percentage)}
              </span>
            </div>
          </div>
        </div>
      )}
      
      {/* Holdings */}
      {portfolio.holdings && portfolio.holdings.length > 0 && (
        <div className="holdings-list">
          <h4>Holdings</h4>
          {portfolio.holdings.map((holding, index) => (
            <div key={index} className="holding-item">
              <div className="holding-header">
                <span className="symbol">{holding.symbol}</span>
                <span className={`pnl ${holding.pnl >= 0 ? 'positive' : 'negative'}`}>
                  {formatPrice(holding.pnl)} ({formatPercentage(holding.pnl_percentage)})
                </span>
              </div>
              <div className="holding-details">
                <div>Quantity: {holding.quantity}</div>
                <div>Avg Price: {formatPrice(holding.avg_price)}</div>
                <div>Current Value: {formatPrice(holding.current_value)}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// Enhanced Signal Card Component
const EnhancedSignalCard = ({ signal }) => {
  const isBuySignal = signal.signal_type === 'BUY';
  
  return (
    <div className={`enhanced-signal-card ${isBuySignal ? 'buy-signal' : 'sell-signal'}`}>
      <div className="signal-header">
        <div className="signal-type">
          <span className="signal-icon">
            {isBuySignal ? '🚀' : '⚡'}
          </span>
          <span className="signal-text">
            {signal.signal_type}
          </span>
        </div>
        <div className="signal-strength">
          <span className="strength-value">{signal.strength.toFixed(0)}%</span>
          <div className="strength-bar">
            <div 
              className="strength-fill" 
              style={{ width: `${signal.strength}%` }}
            ></div>
          </div>
        </div>
      </div>
      
      <div className="signal-details">
        <div className="signal-price">
          {formatPrice(signal.price)}
        </div>
        <div className="signal-time">
          {new Date(signal.timestamp).toLocaleTimeString()}
        </div>
      </div>
      
      <div className="signal-indicators">
        <div className="indicator-row">
          <span>RSI</span>
          <span className={
            signal.indicators.rsi < 30 ? 'oversold' : 
            signal.indicators.rsi > 70 ? 'overbought' : 'neutral'
          }>
            {signal.indicators.rsi.toFixed(1)}
          </span>
        </div>
        <div className="indicator-row">
          <span>MACD</span>
          <span>{signal.indicators.macd.toFixed(6)}</span>
        </div>
        {signal.indicators.bollinger_upper && (
          <div className="indicator-row">
            <span>Bollinger</span>
            <span>
              {signal.indicators.bollinger_upper.toFixed(6)} / {signal.indicators.bollinger_lower.toFixed(6)}
            </span>
          </div>
        )}
        {signal.indicators.stochastic_k && (
          <div className="indicator-row">
            <span>Stochastic</span>
            <span>{signal.indicators.stochastic_k.toFixed(1)} / {signal.indicators.stochastic_d.toFixed(1)}</span>
          </div>
        )}
      </div>
    </div>
  );
};

// Main App Component
function App() {
  const [multiCoinData, setMultiCoinData] = useState({});
  const [selectedSymbol, setSelectedSymbol] = useState('DOGEUSDT');
  const [klineData, setKlineData] = useState([]);
  const [signals, setSignals] = useState([]);
  const [analysis, setAnalysis] = useState(null);
  const [portfolio, setPortfolio] = useState({ holdings: [], summary: null });
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [selectedTimeframe, setSelectedTimeframe] = useState('15m');
  const [activeTab, setActiveTab] = useState('trading');
  const wsRef = useRef(null);
  
  // Fetch initial data
  useEffect(() => {
    fetchMultiCoinData();
    fetchKlineData();
    fetchSignals();
    fetchAnalysis();
    fetchPortfolio();
  }, []);
  
  // Setup WebSocket connection
  useEffect(() => {
    connectWebSocket();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);
  
  // Fetch data when symbol or timeframe changes
  useEffect(() => {
    fetchKlineData();
    fetchAnalysis();
  }, [selectedSymbol, selectedTimeframe]);
  
  const fetchMultiCoinData = async () => {
    try {
      const response = await axios.get(`${API}/multi-coin/prices`);
      setMultiCoinData(response.data);
    } catch (error) {
      console.error('Error fetching multi-coin data:', error);
    }
  };
  
  const fetchKlineData = async () => {
    try {
      const symbol = selectedSymbol.toLowerCase().replace('usdt', '');
      const response = await axios.get(`${API}/${symbol}/klines?timeframe=${selectedTimeframe}&limit=50`);
      setKlineData(response.data);
    } catch (error) {
      console.error('Error fetching kline data:', error);
    }
  };
  
  const fetchSignals = async () => {
    try {
      const response = await axios.get(`${API}/doge/signals`);
      setSignals(response.data);
    } catch (error) {
      console.error('Error fetching signals:', error);
    }
  };
  
  const fetchAnalysis = async () => {
    try {
      const symbol = selectedSymbol.toLowerCase().replace('usdt', '');
      const response = await axios.get(`${API}/${symbol}/analysis?timeframe=${selectedTimeframe}`);
      setAnalysis(response.data);
    } catch (error) {
      console.error('Error fetching analysis:', error);
    }
  };
  
  const fetchPortfolio = async () => {
    try {
      const response = await axios.get(`${API}/portfolio`);
      setPortfolio(response.data);
    } catch (error) {
      console.error('Error fetching portfolio:', error);
    }
  };
  
  const executeTrade = async (tradeData) => {
    try {
      const response = await axios.post(`${API}/portfolio/trade`, tradeData);
      console.log('Trade executed:', response.data);
      
      // Refresh portfolio data
      fetchPortfolio();
      
      // Show success message
      alert(`Trade executed successfully! ${response.data.message}`);
      
    } catch (error) {
      console.error('Error executing trade:', error);
      alert('Trade execution failed. Please try again.');
    }
  };
  
  const connectWebSocket = () => {
    const wsUrl = `${BACKEND_URL.replace('https://', 'wss://').replace('http://', 'ws://')}/api/ws`;
    wsRef.current = new WebSocket(wsUrl);
    
    wsRef.current.onopen = () => {
      setConnectionStatus('connected');
      console.log('WebSocket connected');
    };
    
    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'price') {
        setMultiCoinData(prev => ({
          ...prev,
          [data.data.symbol]: data.data
        }));
      } else if (data.type === 'signal') {
        setSignals(prev => [...prev, data.data].slice(-10));
      }
    };
    
    wsRef.current.onclose = () => {
      setConnectionStatus('disconnected');
      console.log('WebSocket disconnected');
      setTimeout(() => connectWebSocket(), 5000);
    };
    
    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnectionStatus('error');
    };
  };
  
  return (
    <div className="App">
      <header className="app-header">
        <h1>🚀 Advanced Crypto Trading Platform</h1>
        <div className="header-controls">
          <div className="tab-selector">
            <button 
              className={activeTab === 'trading' ? 'active' : ''}
              onClick={() => setActiveTab('trading')}
            >
              Trading
            </button>
            <button 
              className={activeTab === 'portfolio' ? 'active' : ''}
              onClick={() => setActiveTab('portfolio')}
            >
              Portfolio
            </button>
            <button 
              className={activeTab === 'analysis' ? 'active' : ''}
              onClick={() => setActiveTab('analysis')}
            >
              Analysis
            </button>
          </div>
          <div className={`connection-status ${connectionStatus}`}>
            <span className="status-dot"></span>
            {connectionStatus === 'connected' ? 'Live' : 'Disconnected'}
          </div>
        </div>
      </header>
      
      <main className="app-main">
        {activeTab === 'trading' && (
          <div className="trading-dashboard">
            <div className="left-panel">
              <MultiCoinPriceCard 
                multiCoinData={multiCoinData} 
                onSelectCoin={setSelectedSymbol}
              />
              
              <div className="timeframe-selector">
                <h3>Timeframe</h3>
                <div className="timeframe-buttons">
                  <button 
                    className={selectedTimeframe === '15m' ? 'active' : ''}
                    onClick={() => setSelectedTimeframe('15m')}
                  >
                    15m
                  </button>
                  <button 
                    className={selectedTimeframe === '4h' ? 'active' : ''}
                    onClick={() => setSelectedTimeframe('4h')}
                  >
                    4h
                  </button>
                  <button 
                    className={selectedTimeframe === '1d' ? 'active' : ''}
                    onClick={() => setSelectedTimeframe('1d')}
                  >
                    1d
                  </button>
                </div>
              </div>
              
              <div className="chart-container">
                <CandlestickChart 
                  klineData={klineData} 
                  symbol={selectedSymbol}
                />
              </div>
            </div>
            
            <div className="right-panel">
              <div className="signals-container">
                <h3>Trading Signals</h3>
                <div className="signals-list">
                  {signals.length > 0 ? (
                    signals.slice(-5).reverse().map((signal, index) => (
                      <EnhancedSignalCard key={signal.id || index} signal={signal} />
                    ))
                  ) : (
                    <div className="no-signals">
                      <p>Analyzing market conditions...</p>
                      <div className="loading-spinner"></div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
        
        {activeTab === 'portfolio' && (
          <div className="portfolio-dashboard">
            <Portfolio 
              portfolio={portfolio} 
              onExecuteTrade={executeTrade}
            />
          </div>
        )}
        
        {activeTab === 'analysis' && (
          <div className="analysis-dashboard">
            <AdvancedTechnicalAnalysis 
              analysis={analysis} 
              selectedSymbol={selectedSymbol}
            />
          </div>
        )}
      </main>
      
      <footer className="app-footer">
        <p>⚠️ This is a paper trading platform for educational purposes only. Always do your own research before trading.</p>
      </footer>
    </div>
  );
}

export default App;