// Service worker to clear browser caches on activation.
//
// We intentionally *don't* implement offline caching here. The goal is to
// force clients to re-fetch the latest static assets after deployments.
const CACHE_PREFIX = 'mcads-';
const CACHE_NAME = `${CACHE_PREFIX}v1.0.3`;

// Install event - activate immediately.
self.addEventListener('install', event => {
  // Skip waiting to ensure the new service worker activates immediately
  self.skipWaiting();
});

// Activate event - clear old caches
self.addEventListener('activate', event => {
  // Clean up old caches
  event.waitUntil(
    caches.keys().then(cacheNames =>
      Promise.all(
        cacheNames
          // Only delete caches we own (avoid nuking unrelated caches on the same origin).
          .filter(cacheName => cacheName.startsWith(CACHE_PREFIX) && cacheName !== CACHE_NAME)
          .map(cacheName => caches.delete(cacheName))
      )
    )
  );
  
  // Claim clients immediately
  return self.clients.claim();
});
