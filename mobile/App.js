import React, { useState, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  StatusBar,
  SafeAreaView,
  ScrollView,
  TouchableOpacity,
  Alert,
  Platform,
  Dimensions,
  RefreshControl,
  Animated,
} from 'react-native';
import LinearGradient from 'react-native-linear-gradient';
import AsyncStorage from '@react-native-async-storage/async-storage';
import NetInfo from '@react-native-community/netinfo';
import PushNotification from 'react-native-push-notification';
import DeviceInfo from 'react-native-device-info';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { NavigationContainer } from '@react-navigation/native';
import Icon from 'react-native-vector-icons/MaterialIcons';

const { width, height } = Dimensions.get('window');
const Tab = createBottomTabNavigator();

// API Configuration
const API_BASE_URL = 'https://09e672a6-bb5a-400c-8105-643a48e73b61.preview.emergentagent.com/api';

// Main App Component
const EnterpriseCryptoApp = () => {
  const [isConnected, setIsConnected] = useState(true);
  const [appVersion, setAppVersion] = useState('2.0.0');

  useEffect(() => {
    initializeApp();
    setupPushNotifications();
    setupNetworkListener();
  }, []);

  const initializeApp = async () => {
    try {
      const version = DeviceInfo.getVersion();
      setAppVersion(version);
      
      // Load user preferences
      const savedPreferences = await AsyncStorage.getItem('userPreferences');
      if (savedPreferences) {
        console.log('Loaded user preferences:', JSON.parse(savedPreferences));
      }
    } catch (error) {
      console.error('App initialization error:', error);
    }
  };

  const setupPushNotifications = () => {
    PushNotification.configure({
      onRegister: function (token) {
        console.log('FCM Token:', token);
      },
      onNotification: function (notification) {
        console.log('Notification received:', notification);
        
        if (notification.userInteraction) {
          // User tapped notification
          handleNotificationTap(notification);
        }
      },
      permissions: {
        alert: true,
        badge: true,
        sound: true,
      },
      popInitialNotification: true,
      requestPermissions: true,
    });
  };

  const setupNetworkListener = () => {
    NetInfo.addEventListener(state => {
      setIsConnected(state.isConnected);
      if (!state.isConnected) {
        Alert.alert('No Internet', 'Please check your internet connection');
      }
    });
  };

  const handleNotificationTap = (notification) => {
    Alert.alert(
      'Trading Alert',
      notification.message || 'New trading signal available',
      [{ text: 'OK', onPress: () => console.log('Notification acknowledged') }]
    );
  };

  return (
    <NavigationContainer>
      <StatusBar barStyle="light-content" backgroundColor="#1e293b" />
      <Tab.Navigator
        screenOptions={({ route }) => ({
          tabBarIcon: ({ focused, color, size }) => {
            let iconName;
            
            if (route.name === 'Trading') {
              iconName = 'trending-up';
            } else if (route.name === 'AI') {
              iconName = 'psychology';
            } else if (route.name === 'DeFi') {
              iconName = 'agriculture';
            } else if (route.name === 'Bots') {
              iconName = 'smart-toy';
            } else if (route.name === 'Portfolio') {
              iconName = 'account-balance-wallet';
            }

            return <Icon name={iconName} size={size} color={color} />;
          },
          tabBarActiveTintColor: '#3b82f6',
          tabBarInactiveTintColor: 'gray',
          tabBarStyle: {
            backgroundColor: '#1e293b',
            borderTopColor: '#374151',
            height: Platform.OS === 'ios' ? 85 : 65,
            paddingBottom: Platform.OS === 'ios' ? 25 : 10,
          },
          headerStyle: {
            backgroundColor: '#1e293b',
          },
          headerTintColor: '#fff',
          headerTitleStyle: {
            fontWeight: 'bold',
            fontSize: 18,
          },
        })}
      >
        <Tab.Screen 
          name="Trading" 
          component={TradingScreen} 
          options={{ title: '📈 Trading' }}
        />
        <Tab.Screen 
          name="AI" 
          component={AIScreen} 
          options={{ title: '🤖 AI Signals' }}
        />
        <Tab.Screen 
          name="DeFi" 
          component={DeFiScreen} 
          options={{ title: '🌾 DeFi' }}
        />
        <Tab.Screen 
          name="Bots" 
          component={BotsScreen} 
          options={{ title: '🤖 Bots' }}
        />
        <Tab.Screen 
          name="Portfolio" 
          component={PortfolioScreen} 
          options={{ title: '💼 Portfolio' }}
        />
      </Tab.Navigator>
    </NavigationContainer>
  );
};

// Trading Screen Component
const TradingScreen = () => {
  const [cryptoData, setCryptoData] = useState([]);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedCoin, setSelectedCoin] = useState('DOGEUSDT');

  useEffect(() => {
    fetchCryptoData();
    const interval = setInterval(fetchCryptoData, 10000); // Update every 10 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchCryptoData = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/multi-coin/prices`);
      const data = await response.json();
      
      const formattedData = Object.entries(data).map(([symbol, info]) => ({
        symbol,
        ...info,
      }));
      
      setCryptoData(formattedData);
    } catch (error) {
      console.error('Error fetching crypto data:', error);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await fetchCryptoData();
    setRefreshing(false);
  };

  const formatPrice = (price) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 4,
      maximumFractionDigits: 6,
    }).format(price);
  };

  const formatPercentage = (percentage) => {
    return `${percentage >= 0 ? '+' : ''}${percentage.toFixed(2)}%`;
  };

  return (
    <LinearGradient colors={['#1e293b', '#0f172a']} style={styles.container}>
      <SafeAreaView style={styles.safeArea}>
        <ScrollView
          style={styles.scrollView}
          refreshControl={
            <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
          }
        >
          <View style={styles.header}>
            <Text style={styles.headerTitle}>🚀 Live Crypto Prices</Text>
            <Text style={styles.headerSubtitle}>Enterprise Trading Platform</Text>
          </View>

          <View style={styles.cryptoGrid}>
            {cryptoData.map((crypto, index) => (
              <TouchableOpacity
                key={crypto.symbol}
                style={[
                  styles.cryptoCard,
                  selectedCoin === crypto.symbol && styles.selectedCryptoCard,
                ]}
                onPress={() => setSelectedCoin(crypto.symbol)}
              >
                <View style={styles.cryptoHeader}>
                  <Text style={styles.cryptoSymbol}>
                    {crypto.symbol.replace('USDT', '')}
                  </Text>
                  <Text
                    style={[
                      styles.cryptoChange,
                      crypto.change_24h >= 0 ? styles.positive : styles.negative,
                    ]}
                  >
                    {formatPercentage(crypto.change_24h)}
                  </Text>
                </View>
                
                <Text style={styles.cryptoPrice}>
                  {formatPrice(crypto.price)}
                </Text>
                
                <View style={styles.cryptoMetrics}>
                  <Text style={styles.cryptoVolume}>
                    Vol: {crypto.volume?.toLocaleString() || 'N/A'}
                  </Text>
                  <Text style={styles.cryptoHigh}>
                    H: {formatPrice(crypto.high_24h || crypto.price)}
                  </Text>
                </View>
              </TouchableOpacity>
            ))}
          </View>

          {selectedCoin && (
            <View style={styles.selectedCoinDetails}>
              <Text style={styles.selectedCoinTitle}>
                {selectedCoin.replace('USDT', '')} Details
              </Text>
              <Text style={styles.selectedCoinInfo}>
                Real-time price updates • AI-powered analysis • Smart alerts
              </Text>
            </View>
          )}
        </ScrollView>
      </SafeAreaView>
    </LinearGradient>
  );
};

// AI Screen Component
const AIScreen = () => {
  const [aiPrediction, setAiPrediction] = useState(null);
  const [sentiment, setSentiment] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchAIData();
  }, []);

  const fetchAIData = async () => {
    setLoading(true);
    try {
      // Fetch AI prediction
      const predictionResponse = await fetch(`${API_BASE_URL}/ai/price-prediction/doge`);
      const predictionData = await predictionResponse.json();
      setAiPrediction(predictionData);

      // Fetch sentiment analysis
      const sentimentResponse = await fetch(`${API_BASE_URL}/ai/sentiment/doge`);
      const sentimentData = await sentimentResponse.json();
      setSentiment(sentimentData);
    } catch (error) {
      console.error('Error fetching AI data:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <LinearGradient colors={['#1e293b', '#0f172a']} style={styles.container}>
      <SafeAreaView style={styles.safeArea}>
        <ScrollView style={styles.scrollView}>
          <View style={styles.header}>
            <Text style={styles.headerTitle}>🤖 AI Trading Intelligence</Text>
            <Text style={styles.headerSubtitle}>Machine Learning Predictions</Text>
          </View>

          {loading && (
            <View style={styles.loadingContainer}>
              <Text style={styles.loadingText}>🧠 AI Analyzing Market Data...</Text>
            </View>
          )}

          {aiPrediction && (
            <View style={styles.aiCard}>
              <Text style={styles.aiCardTitle}>📈 Price Prediction</Text>
              <Text style={styles.aiSymbol}>{aiPrediction.symbol}</Text>
              
              <View style={styles.predictionGrid}>
                <View style={styles.predictionItem}>
                  <Text style={styles.predictionLabel}>1 Hour</Text>
                  <Text style={styles.predictionValue}>
                    ${aiPrediction.predictions?.['1h']?.toFixed(6) || 'N/A'}
                  </Text>
                </View>
                <View style={styles.predictionItem}>
                  <Text style={styles.predictionLabel}>24 Hours</Text>
                  <Text style={styles.predictionValue}>
                    ${aiPrediction.predictions?.['24h']?.toFixed(6) || 'N/A'}
                  </Text>
                </View>
                <View style={styles.predictionItem}>
                  <Text style={styles.predictionLabel}>7 Days</Text>
                  <Text style={styles.predictionValue}>
                    ${aiPrediction.predictions?.['7d']?.toFixed(6) || 'N/A'}
                  </Text>
                </View>
              </View>

              <View style={styles.aiMetrics}>
                <Text style={styles.confidenceText}>
                  🎯 Confidence: {aiPrediction.confidence?.toFixed(0) || 0}%
                </Text>
                <Text style={[styles.trendText, aiPrediction.trend === 'bullish' ? styles.positive : styles.negative]}>
                  📊 Trend: {aiPrediction.trend?.toUpperCase() || 'ANALYZING'}
                </Text>
              </View>
            </View>
          )}

          {sentiment && (
            <View style={styles.sentimentCard}>
              <Text style={styles.aiCardTitle}>📊 Market Sentiment</Text>
              
              <View style={styles.sentimentOverview}>
                <Text style={styles.sentimentMain}>
                  {sentiment.overall_sentiment?.toUpperCase() || 'NEUTRAL'}
                </Text>
                <Text style={styles.sentimentScore}>
                  Score: {Math.round((sentiment.sentiment_score || 0) * 100)}/100
                </Text>
              </View>

              <View style={styles.sentimentBreakdown}>
                <View style={styles.sentimentItem}>
                  <Text style={styles.sentimentLabel}>📰 News</Text>
                  <Text style={styles.sentimentValue}>
                    {sentiment.breakdown?.news?.sentiment || 'neutral'}
                  </Text>
                </View>
                <View style={styles.sentimentItem}>
                  <Text style={styles.sentimentLabel}>🐦 Social</Text>
                  <Text style={styles.sentimentValue}>
                    {sentiment.breakdown?.social_media?.sentiment || 'neutral'}
                  </Text>
                </View>
                <View style={styles.sentimentItem}>
                  <Text style={styles.sentimentLabel}>📈 Market</Text>
                  <Text style={styles.sentimentValue}>
                    {sentiment.breakdown?.market?.sentiment || 'neutral'}
                  </Text>
                </View>
              </View>
            </View>
          )}

          <TouchableOpacity style={styles.refreshButton} onPress={fetchAIData}>
            <Text style={styles.refreshButtonText}>🔄 Refresh AI Analysis</Text>
          </TouchableOpacity>
        </ScrollView>
      </SafeAreaView>
    </LinearGradient>
  );
};

// DeFi Screen Component
const DeFiScreen = () => {
  const [defiData, setDefiData] = useState(null);
  const [selectedCategory, setSelectedCategory] = useState('yield_farming');

  useEffect(() => {
    fetchDeFiData();
  }, []);

  const fetchDeFiData = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/defi/opportunities`);
      const data = await response.json();
      setDefiData(data);
    } catch (error) {
      console.error('Error fetching DeFi data:', error);
    }
  };

  return (
    <LinearGradient colors={['#1e293b', '#0f172a']} style={styles.container}>
      <SafeAreaView style={styles.safeArea}>
        <ScrollView style={styles.scrollView}>
          <View style={styles.header}>
            <Text style={styles.headerTitle}>🌾 DeFi Opportunities</Text>
            <Text style={styles.headerSubtitle}>Decentralized Finance Yields</Text>
          </View>

          {defiData && (
            <View style={styles.defiStats}>
              <View style={styles.statItem}>
                <Text style={styles.statValue}>{defiData.total_opportunities}</Text>
                <Text style={styles.statLabel}>Opportunities</Text>
              </View>
              <View style={styles.statItem}>
                <Text style={styles.statValue}>{defiData.highest_apy?.toFixed(1)}%</Text>
                <Text style={styles.statLabel}>Highest APY</Text>
              </View>
            </View>
          )}

          <View style={styles.categoryTabs}>
            <TouchableOpacity
              style={[styles.categoryTab, selectedCategory === 'yield_farming' && styles.activeTab]}
              onPress={() => setSelectedCategory('yield_farming')}
            >
              <Text style={[styles.categoryTabText, selectedCategory === 'yield_farming' && styles.activeTabText]}>
                🌾 Yield Farming
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.categoryTab, selectedCategory === 'staking' && styles.activeTab]}
              onPress={() => setSelectedCategory('staking')}
            >
              <Text style={[styles.categoryTabText, selectedCategory === 'staking' && styles.activeTabText]}>
                🔒 Staking
              </Text>
            </TouchableOpacity>
          </View>

          {defiData && selectedCategory === 'yield_farming' && defiData.yield_farming && (
            <View style={styles.opportunitiesList}>
              {defiData.yield_farming.map((farm, index) => (
                <View key={index} style={styles.opportunityCard}>
                  <View style={styles.opportunityHeader}>
                    <Text style={styles.protocolName}>{farm.protocol}</Text>
                    <Text style={[styles.riskBadge, styles[farm.risk_level]]}>
                      {farm.risk_level} risk
                    </Text>
                  </View>
                  
                  <Text style={styles.poolName}>{farm.pool}</Text>
                  
                  <View style={styles.opportunityMetrics}>
                    <View style={styles.metricItem}>
                      <Text style={styles.metricValue}>{farm.apy?.toFixed(1)}%</Text>
                      <Text style={styles.metricLabel}>APY</Text>
                    </View>
                    <View style={styles.metricItem}>
                      <Text style={styles.metricValue}>{farm.lock_period}</Text>
                      <Text style={styles.metricLabel}>Days Lock</Text>
                    </View>
                    <View style={styles.metricItem}>
                      <Text style={styles.metricValue}>${farm.minimum_stake}</Text>
                      <Text style={styles.metricLabel}>Min Stake</Text>
                    </View>
                  </View>
                </View>
              ))}
            </View>
          )}

          {defiData && selectedCategory === 'staking' && defiData.staking && (
            <View style={styles.opportunitiesList}>
              {defiData.staking.map((stake, index) => (
                <View key={index} style={styles.opportunityCard}>
                  <View style={styles.opportunityHeader}>
                    <Text style={styles.protocolName}>{stake.protocol}</Text>
                    <Text style={[styles.riskBadge, styles[stake.risk_level]]}>
                      {stake.risk_level} risk
                    </Text>
                  </View>
                  
                  <Text style={styles.poolName}>{stake.token} Staking</Text>
                  
                  <View style={styles.opportunityMetrics}>
                    <View style={styles.metricItem}>
                      <Text style={styles.metricValue}>{stake.apy?.toFixed(1)}%</Text>
                      <Text style={styles.metricLabel}>APY</Text>
                    </View>
                    <View style={styles.metricItem}>
                      <Text style={styles.metricValue}>{stake.lock_period}</Text>
                      <Text style={styles.metricLabel}>Days Lock</Text>
                    </View>
                    <View style={styles.metricItem}>
                      <Text style={styles.metricValue}>{stake.minimum_stake}</Text>
                      <Text style={styles.metricLabel}>Min {stake.token}</Text>
                    </View>
                  </View>
                </View>
              ))}
            </View>
          )}
        </ScrollView>
      </SafeAreaView>
    </LinearGradient>
  );
};

// Bots Screen Component
const BotsScreen = () => {
  const [botPerformance, setBotPerformance] = useState(null);
  const [showCreateForm, setShowCreateForm] = useState(false);

  useEffect(() => {
    fetchBotPerformance();
  }, []);

  const fetchBotPerformance = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/bots/performance`);
      const data = await response.json();
      setBotPerformance(data);
    } catch (error) {
      console.error('Error fetching bot performance:', error);
    }
  };

  const createBot = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/bots/create`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: 'Mobile DCA Bot',
          strategy: 'dca',
          symbol: 'doge',
          investment_amount: 1000,
          risk_level: 'medium',
        }),
      });
      
      const result = await response.json();
      
      if (result.status === 'success') {
        Alert.alert('Success', 'Trading bot created successfully!');
        setShowCreateForm(false);
        fetchBotPerformance();
      }
    } catch (error) {
      console.error('Error creating bot:', error);
      Alert.alert('Error', 'Failed to create bot. Please try again.');
    }
  };

  return (
    <LinearGradient colors={['#1e293b', '#0f172a']} style={styles.container}>
      <SafeAreaView style={styles.safeArea}>
        <ScrollView style={styles.scrollView}>
          <View style={styles.header}>
            <Text style={styles.headerTitle}>🤖 Trading Bots</Text>
            <Text style={styles.headerSubtitle}>Automated Trading Strategies</Text>
          </View>

          <TouchableOpacity
            style={styles.createBotButton}
            onPress={() => setShowCreateForm(!showCreateForm)}
          >
            <Text style={styles.createBotButtonText}>
              {showCreateForm ? '❌ Cancel' : '➕ Create New Bot'}
            </Text>
          </TouchableOpacity>

          {showCreateForm && (
            <View style={styles.createBotForm}>
              <Text style={styles.formTitle}>Quick Bot Setup</Text>
              <Text style={styles.formDescription}>
                Create a Dollar Cost Averaging bot for DOGE with $1000 investment
              </Text>
              <TouchableOpacity style={styles.submitButton} onPress={createBot}>
                <Text style={styles.submitButtonText}>🚀 Create DCA Bot</Text>
              </TouchableOpacity>
            </View>
          )}

          {botPerformance && botPerformance.total_bots > 0 && (
            <View style={styles.botsList}>
              <Text style={styles.botsListTitle}>Active Bots ({botPerformance.active_bots})</Text>
              
              {Object.entries(botPerformance.performances).map(([botId, perf]) => (
                <View key={botId} style={styles.botCard}>
                  <View style={styles.botHeader}>
                    <Text style={styles.botName}>{perf.bot_name}</Text>
                    <View style={[styles.statusBadge, perf.active ? styles.active : styles.inactive]}>
                      <Text style={styles.statusText}>
                        {perf.active ? '🟢 Active' : '🔴 Inactive'}
                      </Text>
                    </View>
                  </View>
                  
                  <Text style={styles.botStrategy}>
                    Strategy: {perf.strategy.replace('_', ' ').toUpperCase()}
                  </Text>
                  
                  <View style={styles.botMetrics}>
                    <View style={styles.botMetric}>
                      <Text style={styles.metricLabel}>Return</Text>
                      <Text style={[styles.metricValue, perf.total_return >= 0 ? styles.positive : styles.negative]}>
                        {perf.total_return >= 0 ? '+' : ''}{perf.total_return}%
                      </Text>
                    </View>
                    <View style={styles.botMetric}>
                      <Text style={styles.metricLabel}>Win Rate</Text>
                      <Text style={styles.metricValue}>{perf.win_rate?.toFixed(1)}%</Text>
                    </View>
                    <View style={styles.botMetric}>
                      <Text style={styles.metricLabel}>Trades</Text>
                      <Text style={styles.metricValue}>{perf.total_trades}</Text>
                    </View>
                  </View>
                </View>
              ))}
            </View>
          )}

          {(!botPerformance || botPerformance.total_bots === 0) && (
            <View style={styles.noBots}>
              <Text style={styles.noBotsText}>🤖 No Trading Bots Yet</Text>
              <Text style={styles.noBotsSubtext}>
                Create your first automated trading bot to start earning passive income
              </Text>
            </View>
          )}
        </ScrollView>
      </SafeAreaView>
    </LinearGradient>
  );
};

// Portfolio Screen Component
const PortfolioScreen = () => {
  const [portfolio, setPortfolio] = useState({ holdings: [], summary: null });

  useEffect(() => {
    fetchPortfolio();
  }, []);

  const fetchPortfolio = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/portfolio`);
      const data = await response.json();
      setPortfolio(data);
    } catch (error) {
      console.error('Error fetching portfolio:', error);
    }
  };

  const formatPrice = (price) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 6,
    }).format(price);
  };

  return (
    <LinearGradient colors={['#1e293b', '#0f172a']} style={styles.container}>
      <SafeAreaView style={styles.safeArea}>
        <ScrollView style={styles.scrollView}>
          <View style={styles.header}>
            <Text style={styles.headerTitle}>💼 Portfolio</Text>
            <Text style={styles.headerSubtitle}>Your Trading Performance</Text>
          </View>

          {portfolio.summary && (
            <View style={styles.portfolioSummary}>
              <Text style={styles.summaryTitle}>Portfolio Overview</Text>
              
              <View style={styles.summaryGrid}>
                <View style={styles.summaryItem}>
                  <Text style={styles.summaryLabel}>Total Value</Text>
                  <Text style={styles.summaryValue}>
                    {formatPrice(portfolio.summary.total_current_value)}
                  </Text>
                </View>
                <View style={styles.summaryItem}>
                  <Text style={styles.summaryLabel}>Total P&L</Text>
                  <Text style={[
                    styles.summaryValue,
                    portfolio.summary.total_pnl >= 0 ? styles.positive : styles.negative
                  ]}>
                    {formatPrice(portfolio.summary.total_pnl)}
                  </Text>
                </View>
                <View style={styles.summaryItem}>
                  <Text style={styles.summaryLabel}>P&L %</Text>
                  <Text style={[
                    styles.summaryValue,
                    portfolio.summary.total_pnl_percentage >= 0 ? styles.positive : styles.negative
                  ]}>
                    {portfolio.summary.total_pnl_percentage >= 0 ? '+' : ''}
                    {portfolio.summary.total_pnl_percentage?.toFixed(2)}%
                  </Text>
                </View>
                <View style={styles.summaryItem}>
                  <Text style={styles.summaryLabel}>Holdings</Text>
                  <Text style={styles.summaryValue}>
                    {portfolio.holdings?.length || 0}
                  </Text>
                </View>
              </View>
            </View>
          )}

          {portfolio.holdings && portfolio.holdings.length > 0 && (
            <View style={styles.holdingsList}>
              <Text style={styles.holdingsTitle}>Current Holdings</Text>
              
              {portfolio.holdings.map((holding, index) => (
                <View key={index} style={styles.holdingCard}>
                  <View style={styles.holdingHeader}>
                    <Text style={styles.holdingSymbol}>{holding.symbol}</Text>
                    <Text style={[
                      styles.holdingPnl,
                      holding.pnl >= 0 ? styles.positive : styles.negative
                    ]}>
                      {formatPrice(holding.pnl)} ({holding.pnl_percentage?.toFixed(2)}%)
                    </Text>
                  </View>
                  
                  <View style={styles.holdingDetails}>
                    <View style={styles.holdingDetail}>
                      <Text style={styles.detailLabel}>Quantity</Text>
                      <Text style={styles.detailValue}>{holding.quantity}</Text>
                    </View>
                    <View style={styles.holdingDetail}>
                      <Text style={styles.detailLabel}>Avg Price</Text>
                      <Text style={styles.detailValue}>{formatPrice(holding.avg_price)}</Text>
                    </View>
                    <View style={styles.holdingDetail}>
                      <Text style={styles.detailLabel}>Current Value</Text>
                      <Text style={styles.detailValue}>{formatPrice(holding.current_value)}</Text>
                    </View>
                  </View>
                </View>
              ))}
            </View>
          )}

          {(!portfolio.holdings || portfolio.holdings.length === 0) && (
            <View style={styles.noHoldings}>
              <Text style={styles.noHoldingsText}>💼 No Holdings Yet</Text>
              <Text style={styles.noHoldingsSubtext}>
                Start trading to build your portfolio
              </Text>
            </View>
          )}

          <TouchableOpacity style={styles.refreshButton} onPress={fetchPortfolio}>
            <Text style={styles.refreshButtonText}>🔄 Refresh Portfolio</Text>
          </TouchableOpacity>
        </ScrollView>
      </SafeAreaView>
    </LinearGradient>
  );
};

// Styles
const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  safeArea: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
    paddingHorizontal: 16,
  },
  header: {
    paddingVertical: 20,
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 4,
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#94a3b8',
    textAlign: 'center',
  },
  cryptoGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  cryptoCard: {
    width: (width - 48) / 2,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  selectedCryptoCard: {
    borderColor: '#3b82f6',
    backgroundColor: 'rgba(59, 130, 246, 0.1)',
  },
  cryptoHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  cryptoSymbol: {
    fontSize: 16,
    fontWeight: '600',
    color: '#ffffff',
  },
  cryptoChange: {
    fontSize: 12,
    fontWeight: '500',
  },
  positive: {
    color: '#4ade80',
  },
  negative: {
    color: '#f87171',
  },
  cryptoPrice: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 8,
  },
  cryptoMetrics: {
    gap: 4,
  },
  cryptoVolume: {
    fontSize: 10,
    color: '#94a3b8',
  },
  cryptoHigh: {
    fontSize: 10,
    color: '#94a3b8',
  },
  selectedCoinDetails: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
  },
  selectedCoinTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 4,
  },
  selectedCoinInfo: {
    fontSize: 14,
    color: '#94a3b8',
  },
  loadingContainer: {
    padding: 40,
    alignItems: 'center',
  },
  loadingText: {
    fontSize: 16,
    color: '#94a3b8',
    textAlign: 'center',
  },
  aiCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: 'rgba(59, 130, 246, 0.3)',
  },
  aiCardTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 12,
  },
  aiSymbol: {
    fontSize: 14,
    color: '#94a3b8',
    marginBottom: 16,
  },
  predictionGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  predictionItem: {
    alignItems: 'center',
    flex: 1,
  },
  predictionLabel: {
    fontSize: 12,
    color: '#94a3b8',
    marginBottom: 4,
  },
  predictionValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#ffffff',
  },
  aiMetrics: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  confidenceText: {
    fontSize: 14,
    color: '#fbbf24',
    fontWeight: '500',
  },
  trendText: {
    fontSize: 14,
    fontWeight: '600',
  },
  sentimentCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: 'rgba(139, 92, 246, 0.3)',
  },
  sentimentOverview: {
    alignItems: 'center',
    marginBottom: 16,
  },
  sentimentMain: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 4,
  },
  sentimentScore: {
    fontSize: 14,
    color: '#94a3b8',
  },
  sentimentBreakdown: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  sentimentItem: {
    alignItems: 'center',
    flex: 1,
  },
  sentimentLabel: {
    fontSize: 12,
    color: '#94a3b8',
    marginBottom: 4,
  },
  sentimentValue: {
    fontSize: 12,
    fontWeight: '500',
    color: '#ffffff',
    textTransform: 'capitalize',
  },
  refreshButton: {
    backgroundColor: '#3b82f6',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    marginVertical: 20,
  },
  refreshButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#ffffff',
  },
  defiStats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
  },
  statItem: {
    alignItems: 'center',
  },
  statValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 4,
  },
  statLabel: {
    fontSize: 12,
    color: '#94a3b8',
  },
  categoryTabs: {
    flexDirection: 'row',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 12,
    padding: 4,
    marginBottom: 20,
  },
  categoryTab: {
    flex: 1,
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  activeTab: {
    backgroundColor: '#3b82f6',
  },
  categoryTabText: {
    fontSize: 14,
    fontWeight: '500',
    color: '#94a3b8',
  },
  activeTabText: {
    color: '#ffffff',
  },
  opportunitiesList: {
    marginBottom: 20,
  },
  opportunityCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  opportunityHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  protocolName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#ffffff',
  },
  riskBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
    fontSize: 10,
    fontWeight: '500',
    textTransform: 'uppercase',
  },
  low: {
    backgroundColor: 'rgba(74, 222, 128, 0.2)',
    color: '#4ade80',
  },
  medium: {
    backgroundColor: 'rgba(251, 191, 36, 0.2)',
    color: '#fbbf24',
  },
  high: {
    backgroundColor: 'rgba(248, 113, 113, 0.2)',
    color: '#f87171',
  },
  poolName: {
    fontSize: 14,
    color: '#94a3b8',
    marginBottom: 12,
  },
  opportunityMetrics: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  metricItem: {
    alignItems: 'center',
    flex: 1,
  },
  metricValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 4,
  },
  metricLabel: {
    fontSize: 10,
    color: '#94a3b8',
    textTransform: 'uppercase',
  },
  createBotButton: {
    backgroundColor: '#10b981',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    marginBottom: 20,
  },
  createBotButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#ffffff',
  },
  createBotForm: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 12,
    padding: 20,
    marginBottom: 20,
  },
  formTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 8,
  },
  formDescription: {
    fontSize: 14,
    color: '#94a3b8',
    marginBottom: 16,
    lineHeight: 20,
  },
  submitButton: {
    backgroundColor: '#3b82f6',
    borderRadius: 8,
    padding: 12,
    alignItems: 'center',
  },
  submitButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#ffffff',
  },
  botsList: {
    marginBottom: 20,
  },
  botsListTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 12,
  },
  botCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  botHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  botName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#ffffff',
  },
  statusBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
  },
  active: {
    backgroundColor: 'rgba(74, 222, 128, 0.2)',
  },
  inactive: {
    backgroundColor: 'rgba(156, 163, 175, 0.2)',
  },
  statusText: {
    fontSize: 10,
    fontWeight: '500',
  },
  botStrategy: {
    fontSize: 14,
    color: '#94a3b8',
    marginBottom: 12,
  },
  botMetrics: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  botMetric: {
    alignItems: 'center',
    flex: 1,
  },
  noBots: {
    alignItems: 'center',
    paddingVertical: 40,
  },
  noBotsText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 8,
  },
  noBotsSubtext: {
    fontSize: 14,
    color: '#94a3b8',
    textAlign: 'center',
    lineHeight: 20,
    paddingHorizontal: 20,
  },
  portfolioSummary: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  summaryTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 16,
    textAlign: 'center',
  },
  summaryGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  summaryItem: {
    width: '48%',
    alignItems: 'center',
    marginBottom: 16,
  },
  summaryLabel: {
    fontSize: 12,
    color: '#94a3b8',
    marginBottom: 4,
    textAlign: 'center',
  },
  summaryValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#ffffff',
    textAlign: 'center',
  },
  holdingsList: {
    marginBottom: 20,
  },
  holdingsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 12,
  },
  holdingCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  holdingHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  holdingSymbol: {
    fontSize: 16,
    fontWeight: '600',
    color: '#ffffff',
  },
  holdingPnl: {
    fontSize: 14,
    fontWeight: '600',
  },
  holdingDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  holdingDetail: {
    alignItems: 'center',
    flex: 1,
  },
  detailLabel: {
    fontSize: 10,
    color: '#94a3b8',
    marginBottom: 4,
    textTransform: 'uppercase',
  },
  detailValue: {
    fontSize: 12,
    fontWeight: '500',
    color: '#ffffff',
  },
  noHoldings: {
    alignItems: 'center',
    paddingVertical: 40,
  },
  noHoldingsText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 8,
  },
  noHoldingsSubtext: {
    fontSize: 14,
    color: '#94a3b8',
    textAlign: 'center',
  },
});

export default EnterpriseCryptoApp;