/**
 * Service Worker for Smart Factory PWA
 * Only caches http/https requests - avoids chrome-extension and file scheme errors
 */

const CACHE_NAME = 'smart-factory-v1';

// Only cache requests with supported schemes (http, https)
function canCache(request) {
  try {
    const url = request.url || '';
    if (!url.startsWith('http://') && !url.startsWith('https://')) return false;
    // Never cache API or dynamic endpoints
    if (url.includes('/api/')) return false;
    return true;
  } catch (e) {
    return false;
  }
}

self.addEventListener('install', function (event) {
  self.skipWaiting();
});

self.addEventListener('activate', function (event) {
  event.waitUntil(
    caches.keys().then(function (cacheNames) {
      return Promise.all(
        cacheNames
          .filter(function (name) { return name !== CACHE_NAME; })
          .map(function (name) { return caches.delete(name); })
      );
    }).then(function () { return self.clients.claim(); })
  );
});

self.addEventListener('fetch', function (event) {
  const request = event.request;

  // Never try to cache non-http(s) requests (chrome-extension, file, etc.)
  if (!canCache(request)) {
    event.respondWith(fetch(request));
    return;
  }

  // Network first - no aggressive caching to avoid stale data
  event.respondWith(
    fetch(request)
      .then(function (response) {
        // Optionally cache successful same-origin GET requests only
        if (request.method === 'GET' && response.ok && canCache(request)) {
          const responseClone = response.clone();
          caches.open(CACHE_NAME).then(function (cache) {
            cache.put(request, responseClone).catch(function () {});
          });
        }
        return response;
      })
      .catch(function () {
        return caches.match(request).then(function (cached) {
          return cached || new Response('Offline', { status: 503 });
        });
      })
  );
});
