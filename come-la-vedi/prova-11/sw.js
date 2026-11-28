"use strict";
const CACHE="sir-come-la-vedi-prova-11-v1";
const ROOT=new URL("./",self.registration.scope);
const SHELL=[
  new URL("./",ROOT).href,
  new URL("index.html",ROOT).href,
  new URL("manifest.webmanifest",ROOT).href,
  new URL("icons/icon-192.png",ROOT).href,
  new URL("icons/icon-512.png",ROOT).href,
  new URL("icons/apple-touch-icon.png",ROOT).href,
  new URL("../prova-7/original.jpg",ROOT).href,
  new URL("../prova-7/bg.jpg",ROOT).href
];
self.addEventListener("install",event=>event.waitUntil((async()=>{const cache=await caches.open(CACHE);await cache.addAll(SHELL);await self.skipWaiting()})()));
self.addEventListener("activate",event=>event.waitUntil((async()=>{for(const key of await caches.keys())if(key.startsWith("sir-come-la-vedi-prova-11-")&&key!==CACHE)await caches.delete(key);await self.clients.claim()})()));
self.addEventListener("fetch",event=>{
  const request=event.request;
  if(request.method!=="GET")return;
  const url=new URL(request.url);
  if(url.origin!==self.location.origin)return;
  const basePath=ROOT.pathname;
  const inApp=url.pathname.startsWith(basePath);
  const isDemo=url.pathname.endsWith("/come-la-vedi/prova-7/original.jpg")||url.pathname.endsWith("/come-la-vedi/prova-7/bg.jpg");
  if(!inApp&&!isDemo)return;
  event.respondWith((async()=>{
    const cache=await caches.open(CACHE);
    const navigates=request.mode==="navigate"||url.pathname===new URL("index.html",ROOT).pathname;
    if(navigates){try{const response=await fetch(request);if(response.ok)await cache.put(request,response.clone());return response}catch{return await cache.match(request)||await cache.match(new URL("index.html",ROOT).href)}}
    const saved=await cache.match(request);if(saved)return saved;
    try{const response=await fetch(request);if(response.ok)await cache.put(request,response.clone());return response}catch{return new Response("Risorsa non disponibile offline",{status:503,statusText:"Offline"})}
  })());
});

