// Service worker to clear browser cache on activation
const CACHE_VERSION = 'v1.0.3';

// Install event - cache necessary files
self.addEventListener('install', event => {
  console.log('Service Worker: Installed');
  
  // Skip waiting to ensure the new service worker activates immediately
  self.skipWaiting();
});

// Activate event - clear old caches
self.addEventListener('activate', event => {
  console.log('Service Worker: Activated');
  
  // Clean up old caches
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cache => {
          if (cache !== CACHE_VERSION) {
            console.log('Service Worker: Clearing old cache', cache);
            return caches.delete(cache);
          }
        })
      );
    })
  );
  
  // Claim clients immediately
  return self.clients.claim();
});

// Fetch event - network first strategy
self.addEventListener('fetch', event => {
  // Ignore these types of requests
  if (event.request.method !== 'GET') return;
  
  event.respondWith(
    fetch(event.request)
      .then(response => {
        // Return the network response
        return response;
      })
      .catch(() => {
        // If network fails, try the cache
        return caches.match(event.request);
      })
  );
}); 