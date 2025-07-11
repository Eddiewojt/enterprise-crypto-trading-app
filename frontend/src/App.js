import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Helper function to format price
const formatPrice = (price) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 4,
    maximumFractionDigits: 6
  }).format(price);
};

// Helper function to format percentage
const formatPercentage = (percentage) => {
  return `${percentage >= 0 ? '+' : ''}${percentage.toFixed(2)}%`;
};

// Price Card Component
const PriceCard = ({ priceData }) => {
  const isPositive = priceData?.change_24h >= 0;
  
  return (
    <div className="price-card">
      <div className="price-header">
        <h2>DOGE/USDT</h2>
        <div className="price-live">
          <span className="live-dot"></span>
          LIVE
        </div>
      </div>
      
      <div className="price-main">
        <div className="current-price">
          {priceData ? formatPrice(priceData.price) : 'Loading...'}
        </div>
        
        <div className={`price-change ${isPositive ? 'positive' : 'negative'}`}>
          {priceData ? formatPercentage(priceData.change_24h) : '0.00%'}
        </div>
      </div>
      
      {priceData && (
        <div className="price-details">
          <div className="detail-item">
            <span>24h High</span>
            <span>{formatPrice(priceData.high_24h)}</span>
          </div>
          <div className="detail-item">
            <span>24h Low</span>
            <span>{formatPrice(priceData.low_24h)}</span>
          </div>
          <div className="detail-item">
            <span>24h Volume</span>
            <span>{parseInt(priceData.volume).toLocaleString()}</span>
          </div>
        </div>
      )}
    </div>
  );
};

// Signal Card Component
const SignalCard = ({ signal }) => {
  const isBuySignal = signal.signal_type === 'BUY';
  
  return (
    <div className={`signal-card ${isBuySignal ? 'buy-signal' : 'sell-signal'}`}>
      <div className="signal-header">
        <div className="signal-type">
          <span className="signal-icon">
            {isBuySignal ? '📈' : '📉'}
          </span>
          <span className="signal-text">
            {signal.signal_type}
          </span>
        </div>
        <div className="signal-strength">
          <span className="strength-label">Strength</span>
          <span className="strength-value">{signal.strength.toFixed(0)}%</span>
        </div>
      </div>
      
      <div className="signal-details">
        <div className="signal-price">
          Price: {formatPrice(signal.price)}
        </div>
        <div className="signal-timeframe">
          Timeframe: {signal.timeframe}
        </div>
        <div className="signal-time">
          {new Date(signal.timestamp).toLocaleTimeString()}
        </div>
      </div>
      
      <div className="signal-indicators">
        <div className="indicator">
          <span>RSI</span>
          <span>{signal.indicators.rsi.toFixed(1)}</span>
        </div>
        <div className="indicator">
          <span>MACD</span>
          <span>{signal.indicators.macd.toFixed(6)}</span>
        </div>
      </div>
    </div>
  );
};

// Technical Analysis Component
const TechnicalAnalysis = ({ analysis }) => {
  if (!analysis) return null;
  
  const getRSIColor = (rsi) => {
    if (rsi < 30) return 'oversold';
    if (rsi > 70) return 'overbought';
    return 'neutral';
  };
  
  return (
    <div className="technical-analysis">
      <h3>Technical Analysis</h3>
      
      <div className="analysis-indicators">
        <div className="indicator-item">
          <div className="indicator-name">RSI (14)</div>
          <div className={`indicator-value ${getRSIColor(analysis.rsi)}`}>
            {analysis.rsi.toFixed(1)}
          </div>
          <div className="indicator-signal">
            {analysis.analysis.rsi_signal}
          </div>
        </div>
        
        <div className="indicator-item">
          <div className="indicator-name">MACD</div>
          <div className="indicator-value">
            {analysis.macd.toFixed(6)}
          </div>
          <div className="indicator-signal">
            {analysis.analysis.macd_signal}
          </div>
        </div>
        
        <div className="indicator-item">
          <div className="indicator-name">Moving Average</div>
          <div className="indicator-value">
            {analysis.sma_short.toFixed(6)}
          </div>
          <div className="indicator-signal">
            {analysis.analysis.ma_signal}
          </div>
        </div>
      </div>
    </div>
  );
};

// Alert Component
const AlertComponent = ({ signals }) => {
  const [alerts, setAlerts] = useState([]);
  
  useEffect(() => {
    if (signals.length > 0) {
      const latestSignal = signals[signals.length - 1];
      const alertMessage = `${latestSignal.signal_type} signal detected! Strength: ${latestSignal.strength.toFixed(0)}% at ${formatPrice(latestSignal.price)}`;
      
      setAlerts(prev => [{
        id: Date.now(),
        message: alertMessage,
        type: latestSignal.signal_type,
        timestamp: new Date()
      }, ...prev.slice(0, 4)]); // Keep only last 5 alerts
    }
  }, [signals]);
  
  if (alerts.length === 0) return null;
  
  return (
    <div className="alerts-container">
      <h3>Recent Alerts</h3>
      <div className="alerts-list">
        {alerts.map(alert => (
          <div key={alert.id} className={`alert-item ${alert.type.toLowerCase()}`}>
            <div className="alert-icon">
              {alert.type === 'BUY' ? '🟢' : '🔴'}
            </div>
            <div className="alert-content">
              <div className="alert-message">{alert.message}</div>
              <div className="alert-time">
                {alert.timestamp.toLocaleTimeString()}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Main App Component
function App() {
  const [priceData, setPriceData] = useState(null);
  const [signals, setSignals] = useState([]);
  const [analysis, setAnalysis] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [selectedTimeframe, setSelectedTimeframe] = useState('15m');
  const wsRef = useRef(null);
  
  // Fetch initial data
  useEffect(() => {
    fetchPriceData();
    fetchSignals();
    fetchAnalysis();
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
  
  // Fetch analysis when timeframe changes
  useEffect(() => {
    fetchAnalysis();
  }, [selectedTimeframe]);
  
  const fetchPriceData = async () => {
    try {
      const response = await axios.get(`${API}/doge/price`);
      setPriceData(response.data);
    } catch (error) {
      console.error('Error fetching price data:', error);
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
      const response = await axios.get(`${API}/doge/analysis?timeframe=${selectedTimeframe}`);
      setAnalysis(response.data);
    } catch (error) {
      console.error('Error fetching analysis:', error);
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
        setPriceData(data.data);
      } else if (data.type === 'signal') {
        setSignals(prev => [...prev, data.data].slice(-10)); // Keep last 10 signals
      }
    };
    
    wsRef.current.onclose = () => {
      setConnectionStatus('disconnected');
      console.log('WebSocket disconnected');
      // Reconnect after 5 seconds
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
        <h1>🐕 DOGE Trading App</h1>
        <div className={`connection-status ${connectionStatus}`}>
          <span className="status-dot"></span>
          {connectionStatus === 'connected' ? 'Live' : 'Disconnected'}
        </div>
      </header>
      
      <main className="app-main">
        <div className="trading-dashboard">
          <div className="left-panel">
            <PriceCard priceData={priceData} />
            
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
              </div>
            </div>
            
            <TechnicalAnalysis analysis={analysis} />
          </div>
          
          <div className="right-panel">
            <AlertComponent signals={signals} />
            
            <div className="signals-container">
              <h3>Trading Signals</h3>
              <div className="signals-list">
                {signals.length > 0 ? (
                  signals.slice(-5).reverse().map((signal, index) => (
                    <SignalCard key={signal.id || index} signal={signal} />
                  ))
                ) : (
                  <div className="no-signals">
                    <p>No signals yet. Waiting for strong signals...</p>
                    <div className="loading-spinner"></div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>
      
      <footer className="app-footer">
        <p>⚠️ This is for educational purposes only. Always do your own research before trading.</p>
      </footer>
    </div>
  );
}

export default App;