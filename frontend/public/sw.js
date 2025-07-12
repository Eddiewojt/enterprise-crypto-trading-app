const CACHE_NAME = 'ultra-minimal-signals-v3.0.0';
const urlsToCache = [
  '/',
  '/static/js/bundle.js',
  '/static/css/main.css',
  '/manifest.json'
];

// Force update - clear all old caches
self.addEventListener('install', (event) => {
  console.log('🔧 Service Worker: Force installing new version...');
  self.skipWaiting(); // Force activate immediately
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          console.log('🗑️ Deleting old cache:', cacheName);
          return caches.delete(cacheName);
        })
      );
    }).then(() => {
      return caches.open(CACHE_NAME);
    }).then((cache) => {
      console.log('📦 Service Worker: Caching new files');
      return cache.addAll(urlsToCache);
    })
  );
});

// Take control immediately
self.addEventListener('activate', (event) => {
  console.log('✅ Service Worker: Force activated - taking control');
  event.waitUntil(
    self.clients.claim() // Take control of all pages immediately
  );
});

// Always fetch fresh content, bypass cache
self.addEventListener('fetch', (event) => {
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        // Clone response for cache
        const responseToCache = response.clone();
        
        // Update cache with fresh content
        caches.open(CACHE_NAME)
          .then((cache) => {
            cache.put(event.request, responseToCache);
          });
        
        return response;
      })
      .catch(() => {
        // Only use cache if network fails
        return caches.match(event.request);
      })
  );
});

console.log('🚀 Ultra-Minimal Signals - Service Worker v3.0.0 Loaded');