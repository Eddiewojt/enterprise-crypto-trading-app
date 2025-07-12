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

// Extended Multi-Coin Price Card Component
const ExtendedMultiCoinCard = ({ multiCoinData, onSelectCoin, selectedSymbol }) => {
  const [sortBy, setSortBy] = useState('change_24h');
  const [sortOrder, setSortOrder] = useState('desc');
  
  const sortedCoins = Object.entries(multiCoinData).sort((a, b) => {
    const aVal = a[1][sortBy] || 0;
    const bVal = b[1][sortBy] || 0;
    return sortOrder === 'desc' ? bVal - aVal : aVal - bVal;
  });
  
  return (
    <div className="extended-multi-coin-card">
      <div className="coin-card-header">
        <h3>Cryptocurrency Market (15 Coins)</h3>
        <div className="sort-controls">
          <select 
            value={sortBy} 
            onChange={(e) => setSortBy(e.target.value)}
            className="sort-select"
          >
            <option value="change_24h">24h Change</option>
            <option value="price">Price</option>
            <option value="volume">Volume</option>
          </select>
          <button 
            onClick={() => setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc')}
            className="sort-button"
          >
            {sortOrder === 'desc' ? '↓' : '↑'}
          </button>
        </div>
      </div>
      
      <div className="coin-grid-extended">
        {sortedCoins.map(([symbol, data]) => {
          const isPositive = data.change_24h >= 0;
          const coinName = symbol.replace('USDT', '');
          const isSelected = symbol === selectedSymbol;
          
          return (
            <div 
              key={symbol} 
              className={`coin-item-extended ${isPositive ? 'positive' : 'negative'} ${isSelected ? 'selected' : ''}`}
              onClick={() => onSelectCoin(symbol)}
            >
              <div className="coin-header">
                <span className="coin-symbol">{coinName}</span>
                <span className="coin-change">{formatPercentage(data.change_24h)}</span>
              </div>
              <div className="coin-price">
                {formatPrice(data.price)}
              </div>
              <div className="coin-details">
                <div className="coin-volume">Vol: {formatNumber(data.volume)}</div>
                <div className="coin-range">
                  H: {formatPrice(data.high_24h)} | L: {formatPrice(data.low_24h)}
                </div>
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
        width={500} 
        height={350}
        style={{ width: '100%', height: '350px' }}
      />
    </div>
  );
};

// Backtesting Component
const BacktestingPanel = ({ selectedSymbol }) => {
  const [backtestForm, setBacktestForm] = useState({
    symbol: selectedSymbol.replace('USDT', ''),
    timeframe: '15m',
    start_date: '2024-01-01',
    end_date: '2024-12-31',
    strategy: 'combined',
    initial_capital: 10000
  });
  
  const [backtestResults, setBacktestResults] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [pastResults, setPastResults] = useState([]);
  
  useEffect(() => {
    setBacktestForm(prev => ({
      ...prev,
      symbol: selectedSymbol.replace('USDT', '')
    }));
    fetchPastResults();
  }, [selectedSymbol]);
  
  const fetchPastResults = async () => {
    try {
      const response = await axios.get(`${API}/backtest/results`);
      setPastResults(response.data);
    } catch (error) {
      console.error('Error fetching past results:', error);
    }
  };
  
  const runBacktest = async () => {
    setIsRunning(true);
    try {
      const response = await axios.post(`${API}/backtest`, backtestForm);
      setBacktestResults(response.data);
      fetchPastResults();
    } catch (error) {
      console.error('Error running backtest:', error);
      alert('Backtest failed. Please try again.');
    } finally {
      setIsRunning(false);
    }
  };
  
  return (
    <div className="backtesting-panel">
      <h3>Strategy Backtesting</h3>
      
      <div className="backtest-form">
        <div className="form-row">
          <label>Symbol:</label>
          <input 
            type="text" 
            value={backtestForm.symbol}
            onChange={(e) => setBacktestForm({...backtestForm, symbol: e.target.value})}
            disabled
          />
        </div>
        
        <div className="form-row">
          <label>Strategy:</label>
          <select 
            value={backtestForm.strategy}
            onChange={(e) => setBacktestForm({...backtestForm, strategy: e.target.value})}
          >
            <option value="rsi">RSI Strategy</option>
            <option value="macd">MACD Strategy</option>
            <option value="combined">Combined Strategy</option>
          </select>
        </div>
        
        <div className="form-row">
          <label>Timeframe:</label>
          <select 
            value={backtestForm.timeframe}
            onChange={(e) => setBacktestForm({...backtestForm, timeframe: e.target.value})}
          >
            <option value="15m">15 minutes</option>
            <option value="1h">1 hour</option>
            <option value="4h">4 hours</option>
            <option value="1d">1 day</option>
          </select>
        </div>
        
        <div className="form-row">
          <label>Start Date:</label>
          <input 
            type="date" 
            value={backtestForm.start_date}
            onChange={(e) => setBacktestForm({...backtestForm, start_date: e.target.value})}
          />
        </div>
        
        <div className="form-row">
          <label>End Date:</label>
          <input 
            type="date" 
            value={backtestForm.end_date}
            onChange={(e) => setBacktestForm({...backtestForm, end_date: e.target.value})}
          />
        </div>
        
        <div className="form-row">
          <label>Initial Capital:</label>
          <input 
            type="number" 
            value={backtestForm.initial_capital}
            onChange={(e) => setBacktestForm({...backtestForm, initial_capital: parseFloat(e.target.value)})}
          />
        </div>
        
        <button 
          onClick={runBacktest} 
          disabled={isRunning}
          className="backtest-button"
        >
          {isRunning ? 'Running Backtest...' : 'Run Backtest'}
        </button>
      </div>
      
      {backtestResults && (
        <div className="backtest-results">
          <h4>Backtest Results</h4>
          <div className="results-grid">
            <div className="result-item">
              <span className="label">Total Return:</span>
              <span className={`value ${backtestResults.total_return >= 0 ? 'positive' : 'negative'}`}>
                {formatPrice(backtestResults.total_return)} ({formatPercentage(backtestResults.total_return_percentage)})
              </span>
            </div>
            
            <div className="result-item">
              <span className="label">Final Capital:</span>
              <span className="value">{formatPrice(backtestResults.final_capital)}</span>
            </div>
            
            <div className="result-item">
              <span className="label">Total Trades:</span>
              <span className="value">{backtestResults.total_trades}</span>
            </div>
            
            <div className="result-item">
              <span className="label">Win Rate:</span>
              <span className="value">{backtestResults.win_rate.toFixed(1)}%</span>
            </div>
            
            <div className="result-item">
              <span className="label">Max Drawdown:</span>
              <span className="value negative">{backtestResults.max_drawdown.toFixed(2)}%</span>
            </div>
            
            <div className="result-item">
              <span className="label">Sharpe Ratio:</span>
              <span className="value">{backtestResults.sharpe_ratio.toFixed(2)}</span>
            </div>
          </div>
        </div>
      )}
      
      {pastResults.length > 0 && (
        <div className="past-results">
          <h4>Recent Backtest Results</h4>
          <div className="results-list">
            {pastResults.slice(0, 5).map((result, index) => (
              <div key={index} className="result-summary">
                <div className="result-header">
                  <span>{result.symbol} - {result.strategy}</span>
                  <span className={result.total_return >= 0 ? 'positive' : 'negative'}>
                    {formatPercentage(result.total_return_percentage)}
                  </span>
                </div>
                <div className="result-details">
                  <span>Trades: {result.total_trades}</span>
                  <span>Win Rate: {result.win_rate.toFixed(1)}%</span>
                  <span>Timeframe: {result.timeframe}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// Notification Settings Component
const NotificationSettings = () => {
  const [settings, setSettings] = useState({
    email_enabled: true,
    sms_enabled: true,
    telegram_enabled: true,
    signal_strength_threshold: 75
  });
  
  return (
    <div className="notification-settings">
      <h3>Notification Settings</h3>
      
      <div className="settings-grid">
        <div className="setting-item">
          <label>
            <input 
              type="checkbox" 
              checked={settings.email_enabled}
              onChange={(e) => setSettings({...settings, email_enabled: e.target.checked})}
            />
            Email Notifications
          </label>
          <span className="setting-description">eddiewojt1@gmail.com</span>
        </div>
        
        <div className="setting-item">
          <label>
            <input 
              type="checkbox" 
              checked={settings.sms_enabled}
              onChange={(e) => setSettings({...settings, sms_enabled: e.target.checked})}
            />
            SMS Notifications
          </label>
          <span className="setting-description">+610437975583</span>
        </div>
        
        <div className="setting-item">
          <label>
            <input 
              type="checkbox" 
              checked={settings.telegram_enabled}
              onChange={(e) => setSettings({...settings, telegram_enabled: e.target.checked})}
            />
            Telegram Notifications
          </label>
          <span className="setting-description">Bot integration ready</span>
        </div>
        
        <div className="setting-item">
          <label>Signal Strength Threshold:</label>
          <input 
            type="range" 
            min="60" 
            max="90" 
            value={settings.signal_strength_threshold}
            onChange={(e) => setSettings({...settings, signal_strength_threshold: e.target.value})}
          />
          <span className="setting-value">{settings.signal_strength_threshold}%</span>
        </div>
      </div>
      
      <div className="notification-status">
        <h4>Notification Status</h4>
        <div className="status-grid">
          <div className="status-item">
            <span className="status-dot email"></span>
            <span>Email: Connected</span>
          </div>
          <div className="status-item">
            <span className="status-dot sms"></span>
            <span>SMS: Demo Mode</span>
          </div>
          <div className="status-item">
            <span className="status-dot telegram"></span>
            <span>Telegram: Demo Mode</span>
          </div>
        </div>
      </div>
    </div>
  );
};

// Enhanced Portfolio Component
const EnhancedPortfolio = ({ portfolio, onExecuteTrade }) => {
  const [tradeForm, setTradeForm] = useState({
    symbol: 'DOGEUSDT',
    side: 'BUY',
    quantity: 1000,
    price: ''
  });
  
  const [supportedCoins, setSupportedCoins] = useState([]);
  const [tradeHistory, setTradeHistory] = useState([]);
  
  useEffect(() => {
    fetchSupportedCoins();
    fetchTradeHistory();
  }, []);
  
  const fetchSupportedCoins = async () => {
    try {
      const response = await axios.get(`${API}/supported-coins`);
      setSupportedCoins(response.data);
    } catch (error) {
      console.error('Error fetching supported coins:', error);
    }
  };
  
  const fetchTradeHistory = async () => {
    try {
      const response = await axios.get(`${API}/portfolio/trades`);
      setTradeHistory(response.data.trades);
    } catch (error) {
      console.error('Error fetching trade history:', error);
    }
  };
  
  const handleTrade = async () => {
    try {
      await onExecuteTrade(tradeForm);
      setTradeForm({ ...tradeForm, quantity: 1000, price: '' });
      fetchTradeHistory();
    } catch (error) {
      console.error('Trade failed:', error);
    }
  };
  
  return (
    <div className="enhanced-portfolio">
      <h3>Advanced Portfolio Management</h3>
      
      {/* Paper Trading Form */}
      <div className="trade-form">
        <h4>Paper Trading - Multi-Coin Support</h4>
        <div className="form-row">
          <select 
            value={tradeForm.symbol} 
            onChange={(e) => setTradeForm({ ...tradeForm, symbol: e.target.value })}
          >
            {supportedCoins.map(coin => (
              <option key={coin.symbol} value={coin.symbol}>
                {coin.name} ({coin.symbol})
              </option>
            ))}
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
          <h4>Current Holdings</h4>
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
      
      {/* Trade History */}
      {tradeHistory.length > 0 && (
        <div className="trade-history">
          <h4>Recent Trades</h4>
          <div className="trades-list">
            {tradeHistory.slice(0, 10).map((trade, index) => (
              <div key={index} className={`trade-item ${trade.side.toLowerCase()}`}>
                <div className="trade-header">
                  <span className="trade-symbol">{trade.symbol}</span>
                  <span className="trade-side">{trade.side}</span>
                  <span className="trade-time">
                    {new Date(trade.timestamp).toLocaleString()}
                  </span>
                </div>
                <div className="trade-details">
                  <span>Qty: {trade.quantity}</span>
                  <span>Price: {formatPrice(trade.price)}</span>
                  <span>Value: {formatPrice(trade.quantity * trade.price)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
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
      const response = await axios.get(`${API}/signals`);
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
        setSignals(prev => [...prev, data.data].slice(-20));
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
        <h1>🚀 Advanced Multi-Coin Trading Platform</h1>
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
              className={activeTab === 'backtesting' ? 'active' : ''}
              onClick={() => setActiveTab('backtesting')}
            >
              Backtesting
            </button>
            <button 
              className={activeTab === 'notifications' ? 'active' : ''}
              onClick={() => setActiveTab('notifications')}
            >
              Notifications
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
              <ExtendedMultiCoinCard 
                multiCoinData={multiCoinData} 
                onSelectCoin={setSelectedSymbol}
                selectedSymbol={selectedSymbol}
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
                    className={selectedTimeframe === '1h' ? 'active' : ''}
                    onClick={() => setSelectedTimeframe('1h')}
                  >
                    1h
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
                <h3>Multi-Coin Trading Signals</h3>
                <div className="signals-list">
                  {signals.length > 0 ? (
                    signals.slice(-8).reverse().map((signal, index) => (
                      <div key={signal.id || index} className={`signal-card ${signal.signal_type.toLowerCase()}`}>
                        <div className="signal-header">
                          <span className="signal-symbol">{signal.symbol}</span>
                          <span className="signal-type">{signal.signal_type}</span>
                          <span className="signal-strength">{signal.strength.toFixed(0)}%</span>
                        </div>
                        <div className="signal-details">
                          <div>Price: {formatPrice(signal.price)}</div>
                          <div>RSI: {signal.indicators.rsi.toFixed(1)}</div>
                          <div>Time: {new Date(signal.timestamp).toLocaleTimeString()}</div>
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="no-signals">
                      <p>Scanning 15 cryptocurrencies...</p>
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
            <EnhancedPortfolio 
              portfolio={portfolio} 
              onExecuteTrade={executeTrade}
            />
          </div>
        )}
        
        {activeTab === 'backtesting' && (
          <div className="backtesting-dashboard">
            <BacktestingPanel selectedSymbol={selectedSymbol} />
          </div>
        )}
        
        {activeTab === 'notifications' && (
          <div className="notifications-dashboard">
            <NotificationSettings />
          </div>
        )}
      </main>
      
      <footer className="app-footer">
        <p>⚠️ Advanced Multi-Coin Trading Platform - Paper Trading Only | 15 Supported Cryptocurrencies | Enhanced Notifications</p>
      </footer>
    </div>
  );
}

export default App;