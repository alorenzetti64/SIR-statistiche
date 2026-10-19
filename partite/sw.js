/* PARTITE — Service Worker per notifiche push iOS e browser compatibili */
"use strict";
self.addEventListener("install",event=>{
  event.waitUntil(self.skipWaiting());
});
self.addEventListener("activate",event=>{
  event.waitUntil(self.clients.claim());
});
self.addEventListener("push",event=>{
  let payload={
    title:"PARTITE",
    body:"C'è un nuovo aggiornamento.",
    url:"./",
    tag:"partite-update"
  };
  try{
    if(event.data)payload={...payload,...event.data.json()};
  }catch(e){
    if(event.data)payload.body=event.data.text().slice(0,500);
  }
  const title=String(payload.title||"PARTITE").slice(0,120);
  const body=String(payload.body||"").slice(0,280);
  const options={
    body,
    icon:"./apple-touch-icon.png",
    badge:"./apple-touch-icon.png",
    tag:String(payload.tag||"partite-update").slice(0,200),
    data:{url:String(payload.url||"./")},
    requireInteraction:false
  };
  event.waitUntil(self.registration.showNotification(title,options));
});
self.addEventListener("notificationclick",event=>{
  event.notification.close();
  const appScope=new URL(self.registration.scope);
  let target=new URL("./",appScope);
  try{
    const proposed=new URL(event.notification.data?.url||"./",appScope);
    if(proposed.origin===appScope.origin&&proposed.pathname.startsWith(appScope.pathname))target=proposed;
  }catch{}
  event.waitUntil((async()=>{
    const windows=await clients.matchAll({type:"window",includeUncontrolled:true});
    for(const client of windows){
      const current=new URL(client.url);
      if(current.origin===target.origin&&current.pathname.startsWith(appScope.pathname)){
        await client.navigate(target.href);
        return client.focus();
      }
    }
    return clients.openWindow(target.href);
  })());
});