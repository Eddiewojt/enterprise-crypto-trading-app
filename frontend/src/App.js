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

// AI Price Prediction Component
const AIPredictionCard = ({ symbol, selectedTimeframe }) => {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  
  useEffect(() => {
    fetchPrediction();
  }, [symbol]);
  
  const fetchPrediction = async () => {
    setLoading(true);
    try {
      const symbolName = symbol.replace('USDT', '').toLowerCase();
      const response = await axios.get(`${API}/ai/price-prediction/${symbolName}?timeframe=${selectedTimeframe}`);
      setPrediction(response.data);
    } catch (error) {
      console.error('Error fetching AI prediction:', error);
    } finally {
      setLoading(false);
    }
  };
  
  if (loading) return <div className="ai-card loading">🤖 AI analyzing...</div>;
  if (!prediction) return null;
  
  return (
    <div className="ai-prediction-card">
      <div className="ai-header">
        <h4>🤖 AI Price Prediction</h4>
        <span className="confidence-badge">{prediction.confidence.toFixed(0)}% Confidence</span>
      </div>
      
      <div className="predictions-grid">
        <div className="prediction-item">
          <span className="timeframe">1h</span>
          <span className="predicted-price">{formatPrice(prediction.predictions['1h'])}</span>
          <span className={`change ${prediction.predictions['1h'] > prediction.current_price ? 'positive' : 'negative'}`}>
            {formatPercentage(((prediction.predictions['1h'] - prediction.current_price) / prediction.current_price) * 100)}
          </span>
        </div>
        
        <div className="prediction-item">
          <span className="timeframe">24h</span>
          <span className="predicted-price">{formatPrice(prediction.predictions['24h'])}</span>
          <span className={`change ${prediction.predictions['24h'] > prediction.current_price ? 'positive' : 'negative'}`}>
            {formatPercentage(((prediction.predictions['24h'] - prediction.current_price) / prediction.current_price) * 100)}
          </span>
        </div>
        
        <div className="prediction-item">
          <span className="timeframe">7d</span>
          <span className="predicted-price">{formatPrice(prediction.predictions['7d'])}</span>
          <span className={`change ${prediction.predictions['7d'] > prediction.current_price ? 'positive' : 'negative'}`}>
            {formatPercentage(((prediction.predictions['7d'] - prediction.current_price) / prediction.current_price) * 100)}
          </span>
        </div>
      </div>
      
      <div className="trend-indicator">
        <span className={`trend ${prediction.trend}`}>
          {prediction.trend === 'bullish' ? '📈' : prediction.trend === 'bearish' ? '📉' : '➡️'} 
          {prediction.trend.toUpperCase()}
        </span>
        <span className="model-accuracy">Model Accuracy: {(prediction.model_accuracy * 100).toFixed(1)}%</span>
      </div>
    </div>
  );
};

// Sentiment Analysis Component
const SentimentAnalysis = ({ symbol }) => {
  const [sentiment, setSentiment] = useState(null);
  
  useEffect(() => {
    fetchSentiment();
  }, [symbol]);
  
  const fetchSentiment = async () => {
    try {
      const symbolName = symbol.replace('USDT', '').toLowerCase();
      const response = await axios.get(`${API}/ai/sentiment/${symbolName}`);
      setSentiment(response.data);
    } catch (error) {
      console.error('Error fetching sentiment:', error);
    }
  };
  
  if (!sentiment) return null;
  
  const getSentimentColor = (sentiment) => {
    if (sentiment === 'bullish') return '#4ade80';
    if (sentiment === 'bearish') return '#f87171';
    return '#fbbf24';
  };
  
  return (
    <div className="sentiment-analysis">
      <h4>📊 Market Sentiment</h4>
      
      <div className="sentiment-overview">
        <div className="overall-sentiment">
          <span 
            className="sentiment-indicator"
            style={{ color: getSentimentColor(sentiment.overall_sentiment) }}
          >
            {sentiment.overall_sentiment.toUpperCase()}
          </span>
          <span className="sentiment-score">
            Score: {(sentiment.sentiment_score * 100).toFixed(0)}/100
          </span>
        </div>
        
        <div className="confidence-level">
          Confidence: {(sentiment.confidence * 100).toFixed(0)}%
        </div>
      </div>
      
      <div className="sentiment-breakdown">
        <div className="breakdown-item">
          <span className="source">📰 News</span>
          <span className={`sentiment ${sentiment.breakdown.news.sentiment}`}>
            {sentiment.breakdown.news.sentiment}
          </span>
          <span className="score">{(sentiment.breakdown.news.score * 100).toFixed(0)}</span>
        </div>
        
        <div className="breakdown-item">
          <span className="source">🐦 Social</span>
          <span className={`sentiment ${sentiment.breakdown.social_media.sentiment}`}>
            {sentiment.breakdown.social_media.sentiment}
          </span>
          <span className="score">{(sentiment.breakdown.social_media.score * 100).toFixed(0)}</span>
        </div>
        
        <div className="breakdown-item">
          <span className="source">📈 Market</span>
          <span className={`sentiment ${sentiment.breakdown.market.sentiment}`}>
            {sentiment.breakdown.market.sentiment}
          </span>
          <span className="score">{(sentiment.breakdown.market.score * 100).toFixed(0)}</span>
        </div>
      </div>
      
      <div className="sentiment-trend">
        Trend: <span className={`trend ${sentiment.sentiment_trend}`}>{sentiment.sentiment_trend}</span>
      </div>
    </div>
  );
};

// DeFi Opportunities Component
const DeFiOpportunities = () => {
  const [defiData, setDefiData] = useState(null);
  const [selectedCategory, setSelectedCategory] = useState('yield_farming');
  
  useEffect(() => {
    fetchDeFiData();
  }, []);
  
  const fetchDeFiData = async () => {
    try {
      const response = await axios.get(`${API}/defi/opportunities`);
      setDefiData(response.data);
    } catch (error) {
      console.error('Error fetching DeFi data:', error);
    }
  };
  
  if (!defiData) return <div className="defi-loading">Loading DeFi opportunities...</div>;
  
  return (
    <div className="defi-opportunities">
      <div className="defi-header">
        <h3>🌾 DeFi Opportunities</h3>
        <div className="defi-stats">
          <span className="stat">
            <strong>{defiData.total_opportunities}</strong> Opportunities
          </span>
          <span className="stat">
            <strong>{defiData.highest_apy.toFixed(1)}%</strong> Highest APY
          </span>
        </div>
      </div>
      
      <div className="category-tabs">
        <button 
          className={selectedCategory === 'yield_farming' ? 'active' : ''}
          onClick={() => setSelectedCategory('yield_farming')}
        >
          🌾 Yield Farming
        </button>
        <button 
          className={selectedCategory === 'liquidity_pools' ? 'active' : ''}
          onClick={() => setSelectedCategory('liquidity_pools')}
        >
          💧 Liquidity Pools
        </button>
        <button 
          className={selectedCategory === 'staking' ? 'active' : ''}
          onClick={() => setSelectedCategory('staking')}
        >
          🔒 Staking
        </button>
      </div>
      
      <div className="opportunities-list">
        {selectedCategory === 'yield_farming' && defiData.yield_farming && 
          defiData.yield_farming.map((farm, index) => (
            <div key={index} className="opportunity-card">
              <div className="protocol-header">
                <span className="protocol-name">{farm.protocol}</span>
                <span className={`risk-badge ${farm.risk_level}`}>{farm.risk_level} risk</span>
              </div>
              <div className="opportunity-details">
                <div className="pool-info">
                  <span className="pool-name">{farm.pool}</span>
                  <span className="rewards">Rewards: {farm.token_rewards.join(', ')}</span>
                </div>
                <div className="opportunity-metrics">
                  <div className="apy">
                    <span className="apy-value">{farm.apy.toFixed(1)}%</span>
                    <span className="apy-label">APY</span>
                  </div>
                  <div className="lock-period">
                    <span className="lock-value">{farm.lock_period}</span>
                    <span className="lock-label">days lock</span>
                  </div>
                  <div className="min-stake">
                    <span className="stake-value">${farm.minimum_stake}</span>
                    <span className="stake-label">min stake</span>
                  </div>
                </div>
              </div>
            </div>
          ))
        }
        
        {selectedCategory === 'liquidity_pools' && defiData.liquidity_pools &&
          defiData.liquidity_pools.map((pool, index) => (
            <div key={index} className="opportunity-card">
              <div className="protocol-header">
                <span className="protocol-name">{pool.protocol}</span>
                <span className={`il-risk ${pool.impermanent_loss_risk}`}>IL: {pool.impermanent_loss_risk}</span>
              </div>
              <div className="opportunity-details">
                <div className="pool-info">
                  <span className="pool-name">{pool.pair}</span>
                  <span className="tvl">TVL: ${formatNumber(pool.tvl)}</span>
                </div>
                <div className="opportunity-metrics">
                  <div className="apy">
                    <span className="apy-value">{pool.apy.toFixed(1)}%</span>
                    <span className="apy-label">APY</span>
                  </div>
                  <div className="volume">
                    <span className="volume-value">${formatNumber(pool.volume_24h)}</span>
                    <span className="volume-label">24h volume</span>
                  </div>
                  <div className="risk-score">
                    <span className="risk-value">{(pool.risk_score * 100).toFixed(0)}%</span>
                    <span className="risk-label">risk score</span>
                  </div>
                </div>
              </div>
            </div>
          ))
        }
        
        {selectedCategory === 'staking' && defiData.staking &&
          defiData.staking.map((stake, index) => (
            <div key={index} className="opportunity-card">
              <div className="protocol-header">
                <span className="protocol-name">{stake.protocol}</span>
                <span className={`risk-badge ${stake.risk_level}`}>{stake.risk_level} risk</span>
              </div>
              <div className="opportunity-details">
                <div className="pool-info">
                  <span className="pool-name">{stake.token} Staking</span>
                  <span className="validator">{stake.validator_required ? 'Validator Required' : 'Delegated Staking'}</span>
                </div>
                <div className="opportunity-metrics">
                  <div className="apy">
                    <span className="apy-value">{stake.apy.toFixed(1)}%</span>
                    <span className="apy-label">APY</span>
                  </div>
                  <div className="lock-period">
                    <span className="lock-value">{stake.lock_period}</span>
                    <span className="lock-label">days lock</span>
                  </div>
                  <div className="min-stake">
                    <span className="stake-value">{stake.minimum_stake}</span>
                    <span className="stake-label">min {stake.token}</span>
                  </div>
                </div>
              </div>
            </div>
          ))
        }
      </div>
    </div>
  );
};

// Trading Bots Dashboard
const TradingBots = () => {
  const [bots, setBots] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [performance, setPerformance] = useState(null);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [createForm, setCreateForm] = useState({
    name: '',
    strategy: 'dca',
    symbol: 'doge',
    investment_amount: 1000,
    risk_level: 'medium'
  });
  
  useEffect(() => {
    fetchBotRecommendations();
    fetchBotPerformance();
  }, []);
  
  const fetchBotRecommendations = async () => {
    try {
      const response = await axios.get(`${API}/bots/recommendations?experience=intermediate&investment_amount=5000`);
      setRecommendations(response.data.recommendations);
    } catch (error) {
      console.error('Error fetching bot recommendations:', error);
    }
  };
  
  const fetchBotPerformance = async () => {
    try {
      const response = await axios.get(`${API}/bots/performance`);
      setPerformance(response.data);
    } catch (error) {
      console.error('Error fetching bot performance:', error);
    }
  };
  
  const createBot = async () => {
    try {
      const response = await axios.post(`${API}/bots/create`, createForm);
      if (response.data.status === 'success') {
        alert('Trading bot created successfully!');
        setShowCreateForm(false);
        fetchBotPerformance();
        setCreateForm({
          name: '',
          strategy: 'dca',
          symbol: 'doge',
          investment_amount: 1000,
          risk_level: 'medium'
        });
      }
    } catch (error) {
      console.error('Error creating bot:', error);
      alert('Failed to create bot. Please try again.');
    }
  };
  
  return (
    <div className="trading-bots">
      <div className="bots-header">
        <h3>🤖 Trading Bots</h3>
        <button 
          className="create-bot-btn"
          onClick={() => setShowCreateForm(!showCreateForm)}
        >
          {showCreateForm ? 'Cancel' : '+ Create Bot'}
        </button>
      </div>
      
      {showCreateForm && (
        <div className="create-bot-form">
          <h4>Create Trading Bot</h4>
          <div className="form-grid">
            <div className="form-group">
              <label>Bot Name:</label>
              <input 
                type="text"
                value={createForm.name}
                onChange={(e) => setCreateForm({...createForm, name: e.target.value})}
                placeholder="My Trading Bot"
              />
            </div>
            
            <div className="form-group">
              <label>Strategy:</label>
              <select 
                value={createForm.strategy}
                onChange={(e) => setCreateForm({...createForm, strategy: e.target.value})}
              >
                <option value="dca">Dollar Cost Averaging</option>
                <option value="grid_trading">Grid Trading</option>
                <option value="momentum">Momentum Trading</option>
                <option value="mean_reversion">Mean Reversion</option>
                <option value="copy_trading">Copy Trading</option>
              </select>
            </div>
            
            <div className="form-group">
              <label>Symbol:</label>
              <select 
                value={createForm.symbol}
                onChange={(e) => setCreateForm({...createForm, symbol: e.target.value})}
              >
                <option value="doge">DOGE</option>
                <option value="btc">BTC</option>
                <option value="eth">ETH</option>
                <option value="ada">ADA</option>
                <option value="sol">SOL</option>
              </select>
            </div>
            
            <div className="form-group">
              <label>Investment Amount:</label>
              <input 
                type="number"
                value={createForm.investment_amount}
                onChange={(e) => setCreateForm({...createForm, investment_amount: parseFloat(e.target.value)})}
                min="100"
                step="100"
              />
            </div>
            
            <div className="form-group">
              <label>Risk Level:</label>
              <select 
                value={createForm.risk_level}
                onChange={(e) => setCreateForm({...createForm, risk_level: e.target.value})}
              >
                <option value="low">Low Risk</option>
                <option value="medium">Medium Risk</option>
                <option value="high">High Risk</option>
              </select>
            </div>
          </div>
          
          <button className="create-bot-submit" onClick={createBot}>
            Create Bot
          </button>
        </div>
      )}
      
      {performance && performance.total_bots > 0 && (
        <div className="bot-performance">
          <h4>Active Bots Performance</h4>
          <div className="performance-grid">
            {Object.entries(performance.performances).map(([botId, perf]) => (
              <div key={botId} className="bot-card">
                <div className="bot-header">
                  <span className="bot-name">{perf.bot_name}</span>
                  <span className={`bot-status ${perf.active ? 'active' : 'inactive'}`}>
                    {perf.active ? 'Active' : 'Inactive'}
                  </span>
                </div>
                
                <div className="bot-strategy">
                  Strategy: <strong>{perf.strategy.replace('_', ' ').toUpperCase()}</strong>
                </div>
                
                <div className="bot-metrics">
                  <div className="metric">
                    <span className="metric-label">Return</span>
                    <span className={`metric-value ${perf.total_return >= 0 ? 'positive' : 'negative'}`}>
                      {perf.total_return >= 0 ? '+' : ''}{perf.total_return}%
                    </span>
                  </div>
                  
                  <div className="metric">
                    <span className="metric-label">Win Rate</span>
                    <span className="metric-value">{perf.win_rate.toFixed(1)}%</span>
                  </div>
                  
                  <div className="metric">
                    <span className="metric-label">Trades</span>
                    <span className="metric-value">{perf.total_trades}</span>
                  </div>
                  
                  <div className="metric">
                    <span className="metric-label">Sharpe</span>
                    <span className="metric-value">{perf.sharpe_ratio}</span>
                  </div>
                </div>
                
                <div className="bot-actions">
                  <button className="btn-secondary">View Details</button>
                  <button className="btn-primary">
                    {perf.active ? 'Pause' : 'Resume'}
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      <div className="bot-recommendations">
        <h4>Recommended Bots</h4>
        <div className="recommendations-grid">
          {recommendations.map((rec, index) => (
            <div key={index} className="recommendation-card">
              <div className="rec-header">
                <span className="rec-name">{rec.name}</span>
                <span className={`risk-badge ${rec.risk_level}`}>{rec.risk_level}</span>
              </div>
              
              <div className="rec-description">{rec.description}</div>
              
              <div className="rec-metrics">
                <div className="rec-metric">
                  <span className="label">Expected Return</span>
                  <span className="value">{rec.expected_return}</span>
                </div>
                <div className="rec-metric">
                  <span className="label">Min Investment</span>
                  <span className="value">${rec.min_investment}</span>
                </div>
                <div className="rec-metric">
                  <span className="label">Complexity</span>
                  <span className="value">{rec.complexity}</span>
                </div>
              </div>
              
              <button className="create-from-rec">
                Create This Bot
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// NFT Market Analysis
const NFTAnalysis = () => {
  const [nftData, setNftData] = useState(null);
  
  useEffect(() => {
    fetchNFTData();
  }, []);
  
  const fetchNFTData = async () => {
    try {
      const response = await axios.get(`${API}/nft/market-analysis`);
      setNftData(response.data);
    } catch (error) {
      console.error('Error fetching NFT data:', error);
    }
  };
  
  if (!nftData) return <div className="nft-loading">Loading NFT market data...</div>;
  
  return (
    <div className="nft-analysis">
      <div className="nft-header">
        <h3>🎨 NFT Market Analysis</h3>
        <div className="market-sentiment">
          <span className={`sentiment ${nftData.market_overview.market_sentiment}`}>
            {nftData.market_overview.market_sentiment.toUpperCase()}
          </span>
        </div>
      </div>
      
      <div className="nft-overview">
        <div className="overview-stats">
          <div className="stat-card">
            <span className="stat-value">{nftData.market_overview.total_volume_24h.toFixed(1)}Ξ</span>
            <span className="stat-label">24h Volume</span>
          </div>
          <div className="stat-card">
            <span className="stat-value">{formatPercentage(nftData.market_overview.average_change_24h)}</span>
            <span className="stat-label">Avg Change</span>
          </div>
          <div className="stat-card">
            <span className="stat-value">{nftData.market_overview.collections_tracked}</span>
            <span className="stat-label">Collections</span>
          </div>
        </div>
      </div>
      
      <div className="nft-sections">
        <div className="trending-collections">
          <h4>🔥 Trending Collections</h4>
          {nftData.trending.map((collection, index) => (
            <div key={index} className="collection-card trending">
              <div className="collection-header">
                <span className="collection-name">{collection.name}</span>
                <span className="change positive">+{collection.change_24h.toFixed(1)}%</span>
              </div>
              <div className="collection-metrics">
                <div className="metric">
                  <span className="label">Floor</span>
                  <span className="value">{collection.floor_price}Ξ</span>
                </div>
                <div className="metric">
                  <span className="label">Volume</span>
                  <span className="value">{collection.volume_24h}Ξ</span>
                </div>
                <div className="metric">
                  <span className="label">Sales</span>
                  <span className="value">{collection.sales_24h}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
        
        <div className="declining-collections">
          <h4>📉 Declining Collections</h4>
          {nftData.declining.map((collection, index) => (
            <div key={index} className="collection-card declining">
              <div className="collection-header">
                <span className="collection-name">{collection.name}</span>
                <span className="change negative">{collection.change_24h.toFixed(1)}%</span>
              </div>
              <div className="collection-metrics">
                <div className="metric">
                  <span className="label">Floor</span>
                  <span className="value">{collection.floor_price}Ξ</span>
                </div>
                <div className="metric">
                  <span className="label">Volume</span>
                  <span className="value">{collection.volume_24h}Ξ</span>
                </div>
                <div className="metric">
                  <span className="label">Sales</span>
                  <span className="value">{collection.sales_24h}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
      
      {nftData.opportunities && nftData.opportunities.length > 0 && (
        <div className="nft-opportunities">
          <h4>💎 Investment Opportunities</h4>
          {nftData.opportunities.map((opp, index) => (
            <div key={index} className={`opportunity-card ${opp.type}`}>
              <div className="opp-header">
                <span className="opp-type">{opp.type.toUpperCase()}</span>
                <span className={`risk-badge ${opp.risk_level}`}>{opp.risk_level} risk</span>
              </div>
              <div className="opp-collection">{opp.collection}</div>
              <div className="opp-reason">{opp.reason}</div>
              <div className="opp-score">
                Opportunity Score: <strong>{opp.opportunity_score.toFixed(1)}</strong>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// Arbitrage Opportunities
const ArbitrageTracker = () => {
  const [arbitrageData, setArbitrageData] = useState([]);
  const [loading, setLoading] = useState(false);
  
  useEffect(() => {
    fetchArbitrageData();
    const interval = setInterval(fetchArbitrageData, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);
  
  const fetchArbitrageData = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/defi/arbitrage?min_profit=0.3`);
      setArbitrageData(response.data.opportunities);
    } catch (error) {
      console.error('Error fetching arbitrage data:', error);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="arbitrage-tracker">
      <div className="arbitrage-header">
        <h3>⚡ Arbitrage Opportunities</h3>
        <div className="refresh-indicator">
          {loading ? (
            <span className="refreshing">🔄 Scanning...</span>
          ) : (
            <span className="last-update">Updated {new Date().toLocaleTimeString()}</span>
          )}
        </div>
      </div>
      
      {arbitrageData.length > 0 ? (
        <div className="arbitrage-list">
          {arbitrageData.map((arb, index) => (
            <div key={index} className="arbitrage-card">
              <div className="arb-header">
                <span className="symbol">{arb.symbol}</span>
                <span className="profit-badge">
                  +{arb.profit_percentage.toFixed(2)}%
                </span>
              </div>
              
              <div className="arb-details">
                <div className="exchange-info">
                  <div className="buy-info">
                    <span className="action">BUY</span>
                    <span className="exchange">{arb.buy_exchange}</span>
                    <span className="price">{formatPrice(arb.buy_price)}</span>
                  </div>
                  
                  <div className="arrow">→</div>
                  
                  <div className="sell-info">
                    <span className="action">SELL</span>
                    <span className="exchange">{arb.sell_exchange}</span>
                    <span className="price">{formatPrice(arb.sell_price)}</span>
                  </div>
                </div>
                
                <div className="arb-metrics">
                  <div className="metric">
                    <span className="label">Est. Profit</span>
                    <span className="value">${arb.estimated_profit.toFixed(2)}</span>
                  </div>
                  <div className="metric">
                    <span className="label">Volume</span>
                    <span className="value">{formatNumber(arb.volume_available)}</span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="no-arbitrage">
          <p>No profitable arbitrage opportunities found</p>
          <span className="min-threshold">Minimum profit threshold: 0.3%</span>
        </div>
      )}
    </div>
  );
};

// Main App Component with all enterprise features
function App() {
  const [multiCoinData, setMultiCoinData] = useState({});
  const [selectedSymbol, setSelectedSymbol] = useState('DOGEUSDT');
  const [selectedTimeframe, setSelectedTimeframe] = useState('15m');
  const [activeTab, setActiveTab] = useState('ai-trading');
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const wsRef = useRef(null);
  
  // Fetch initial data
  useEffect(() => {
    fetchMultiCoinData();
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
  
  const fetchMultiCoinData = async () => {
    try {
      const response = await axios.get(`${API}/multi-coin/prices`);
      setMultiCoinData(response.data);
    } catch (error) {
      console.error('Error fetching multi-coin data:', error);
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
      }
    };
    
    wsRef.current.onclose = () => {
      setConnectionStatus('disconnected');
      setTimeout(() => connectWebSocket(), 5000);
    };
    
    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnectionStatus('error');
    };
  };
  
  return (
    <div className="App enterprise-app">
      <header className="enterprise-header">
        <div className="header-brand">
          <h1>🚀 Enterprise AI Trading Platform</h1>
          <span className="version-badge">v2.0 Enterprise</span>
        </div>
        
        <div className="header-controls">
          <div className="enterprise-tabs">
            <button 
              className={activeTab === 'ai-trading' ? 'active' : ''}
              onClick={() => setActiveTab('ai-trading')}
            >
              🤖 AI Trading
            </button>
            <button 
              className={activeTab === 'defi' ? 'active' : ''}
              onClick={() => setActiveTab('defi')}
            >
              🌾 DeFi
            </button>
            <button 
              className={activeTab === 'bots' ? 'active' : ''}
              onClick={() => setActiveTab('bots')}
            >
              🤖 Trading Bots
            </button>
            <button 
              className={activeTab === 'nft' ? 'active' : ''}
              onClick={() => setActiveTab('nft')}
            >
              🎨 NFT
            </button>
            <button 
              className={activeTab === 'arbitrage' ? 'active' : ''}
              onClick={() => setActiveTab('arbitrage')}
            >
              ⚡ Arbitrage
            </button>
          </div>
          
          <div className={`connection-status ${connectionStatus}`}>
            <span className="status-dot"></span>
            {connectionStatus === 'connected' ? 'Live Data' : 'Reconnecting...'}
          </div>
        </div>
      </header>
      
      <main className="enterprise-main">
        {activeTab === 'ai-trading' && (
          <div className="ai-trading-dashboard">
            <div className="left-panel">
              <div className="symbol-selector">
                <h3>Select Cryptocurrency</h3>
                <div className="symbol-grid">
                  {Object.entries(multiCoinData).slice(0, 8).map(([symbol, data]) => (
                    <button
                      key={symbol}
                      className={`symbol-btn ${selectedSymbol === symbol ? 'active' : ''}`}
                      onClick={() => setSelectedSymbol(symbol)}
                    >
                      <span className="symbol">{symbol.replace('USDT', '')}</span>
                      <span className={`change ${data.change_24h >= 0 ? 'positive' : 'negative'}`}>
                        {formatPercentage(data.change_24h)}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
              
              <AIPredictionCard symbol={selectedSymbol} selectedTimeframe={selectedTimeframe} />
            </div>
            
            <div className="right-panel">
              <SentimentAnalysis symbol={selectedSymbol} />
            </div>
          </div>
        )}
        
        {activeTab === 'defi' && (
          <div className="defi-dashboard">
            <DeFiOpportunities />
          </div>
        )}
        
        {activeTab === 'bots' && (
          <div className="bots-dashboard">
            <TradingBots />
          </div>
        )}
        
        {activeTab === 'nft' && (
          <div className="nft-dashboard">
            <NFTAnalysis />
          </div>
        )}
        
        {activeTab === 'arbitrage' && (
          <div className="arbitrage-dashboard">
            <ArbitrageTracker />
          </div>
        )}
      </main>
      
      <footer className="enterprise-footer">
        <div className="footer-content">
          <div className="footer-section">
            <h4>Enterprise Features</h4>
            <ul>
              <li>✅ AI Price Prediction</li>
              <li>✅ Real-time Sentiment Analysis</li>
              <li>✅ DeFi Yield Optimization</li>
              <li>✅ Automated Trading Bots</li>
              <li>✅ NFT Market Intelligence</li>
              <li>✅ Cross-Exchange Arbitrage</li>
            </ul>
          </div>
          
          <div className="footer-section">
            <h4>Advanced Analytics</h4>
            <ul>
              <li>✅ Multi-timeframe Analysis</li>
              <li>✅ Pattern Recognition AI</li>
              <li>✅ Portfolio Optimization</li>
              <li>✅ Risk Management</li>
              <li>✅ Social Trading</li>
              <li>✅ Professional Reporting</li>
            </ul>
          </div>
          
          <div className="footer-disclaimer">
            <p>⚠️ Enterprise AI Trading Platform - For Professional Use Only</p>
            <p>Advanced features include machine learning predictions, DeFi optimization, and automated strategies.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;