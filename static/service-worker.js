const CACHE_NAME = 'navify-cache-v1';
const urlsToCache = [
  '/',
  '/DJ.png',
  '/static/manifest.json',
  '/static/style.css',
  '/static/app.js',
];

self.addEventListener('install', function(event) {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(function(cache) {
        return cache.addAll(urlsToCache);
      })
  );
});

self.addEventListener('fetch', function(event) {
  event.respondWith(
    caches.match(event.request)
      .then(function(response) {
        return response || fetch(event.request);
      })
  );
}); 