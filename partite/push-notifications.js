'use strict';
/* PARTITE · Attivazione volontaria di vere notifiche push */
(function(){
 const ENDPOINT='https://vqrpkkrqynvlzjufiocn.supabase.co/functions/v1/partite-push';
 const banner=document.getElementById('pushPanel');
 const button=document.getElementById('pushToggle');
 const testButton=document.getElementById('pushTest');
 const state=document.getElementById('pushState');
 const help=document.getElementById('pushHelp');
 const bell=document.getElementById('pushBell');
 const menu=document.getElementById('pushMenu');
 if(!banner||!button||!testButton||!state||!help||!bell||!menu)return;
 function setOpen(open){
  banner.hidden=!open;
  bell.setAttribute('aria-expanded',String(open));
 }
 bell.addEventListener('click',()=>setOpen(banner.hidden));
 document.addEventListener('click',event=>{
  if(!banner.hidden&&!menu.contains(event.target))setOpen(false);
 });
 document.addEventListener('keydown',event=>{
  if(event.key==='Escape'&&!banner.hidden){
   setOpen(false);
   bell.focus();
  }
 });
 let busy=false;
 let currentSubscription=null;
 const pushCapable=()=> 'serviceWorker' in navigator && 'PushManager' in window && 'Notification' in window;
 const isAppleMobile=()=>/iPhone|iPad|iPod/.test(navigator.userAgent)||
  (navigator.platform==='MacIntel'&&navigator.maxTouchPoints>1);
 const installed=()=>window.matchMedia('(display-mode: standalone)').matches||navigator.standalone===true;
 const cleanKey=(key)=>{
  const padding='='.repeat((4-key.length%4)%4);
  const raw=atob((key+padding).replace(/-/g,'+').replace(/_/g,'/'));
  return Uint8Array.from(raw,c=>c.charCodeAt(0));
 };
 function status(message,bad=false){
  state.textContent=message;
  state.classList.toggle('is-error',bad);
 }
 function render(){
  const subscribed=!!currentSubscription&&pushCapable()&&Notification.permission==='granted';
  banner.classList.toggle('push-active',subscribed);
  bell.classList.toggle('push-bell-active',subscribed);
  bell.setAttribute('aria-label',subscribed?'Notifiche attive. Apri impostazioni':'Apri impostazioni notifiche');
  button.disabled=busy;
  testButton.classList.toggle('hidden',!subscribed);
  testButton.disabled=busy;
  button.textContent=busy?'Attendere…':subscribed?'Disattiva notifiche':'Attiva notifiche';
  if(busy)return;
  if(isAppleMobile()&&!installed()){
   status('Per riceverle su iPhone, apri PARTITE dall’icona nella schermata Home.');
  } else if(!pushCapable()){
   status('Le notifiche push non sono disponibili su questo dispositivo o browser.',true);
  } else if(Notification.permission==='denied'){
   status('Notifiche bloccate: puoi riattivarle dalle impostazioni del dispositivo.',true);
  } else if(subscribed){
   status('Notifiche attive su questo dispositivo.');
  } else if(Notification.permission==='granted'){
   status('Le notifiche sono consentite ma non ancora attivate in PARTITE.');
  } else {
   status('Le notifiche sono disattivate.');
  }
 }
 async function api(action,fields={}){
  const response=await fetch(ENDPOINT,{
   method:'POST',
   headers:{'Content-Type':'application/json'},
   body:JSON.stringify({action,...fields}),
   cache:'no-store'
  });
  let data={};
  try{data=await response.json()}catch{}
  if(!response.ok)throw Error(data.error||'Errore di collegamento ('+response.status+')');
  return data;
 }
 async function serviceWorker(){
  if(!('serviceWorker' in navigator))throw Error('Questo browser non supporta le notifiche.');
  const reg=await navigator.serviceWorker.register('./sw.js?v=1',{scope:'./'});
  return reg;
 }
 async function getCurrentSubscription(){
  if(!pushCapable()||Notification.permission!=='granted')return null;
  const reg=await serviceWorker();
  const sub=await reg.pushManager.getSubscription();
  return sub||null;
 }
 async function refresh(){
  try{currentSubscription=await getCurrentSubscription()}catch(e){
   console.info('PARTITE push non ancora disponibile',e);
   currentSubscription=null;
  }
  render();
 }
 async function turnOn(){
  if(isAppleMobile()&&!installed()){
   status('Prima aggiungi PARTITE alla schermata Home da Safari, poi aprila dall’icona.',true);
   return;
  }
  if(!pushCapable()){render();return}
  if(Notification.permission==='denied'){
   status('Apri Impostazioni dell’iPhone → Notifiche → PARTITE e consenti gli avvisi.',true);
   return;
  }
  busy=true;render();
  try{
   // Il consenso deve partire direttamente dal tocco sul pulsante.
   const permission=Notification.permission==='granted'?'granted':await Notification.requestPermission();
   if(permission!=='granted'){
    status(permission==='denied'?'L’iPhone ha bloccato le notifiche.':'Notifiche non autorizzate.',true);
    return;
   }
   const reg=await serviceWorker();
   const response=await fetch(ENDPOINT+'?action=public-key',{cache:'no-store'});
   if(!response.ok)throw Error('Chiave di notifica non disponibile.');
   const data=await response.json();
   if(!data.publicKey)throw Error('Chiave pubblica assente.');
   let sub=await reg.pushManager.getSubscription();
   if(!sub){
    sub=await reg.pushManager.subscribe({
     userVisibleOnly:true,
     applicationServerKey:cleanKey(data.publicKey)
    });
   }
   const result=await api('subscribe',{
    subscription:sub.toJSON(),
    deviceLabel:navigator.userAgent.slice(0,95)
   });
   if(!result.subscribed)throw Error('Registrazione del dispositivo non riuscita.');
   currentSubscription=sub;
   status('Notifiche attivate. Puoi premere “Prova notifica”.');
  }catch(error){
   console.error('PARTITE push enable',error);
   const message=String(error?.message||error);
   status(/AbortError|NotAllowedError/.test(message)?
    'Notifiche non autorizzate dal dispositivo. Controlla le impostazioni di PARTITE.':
    'Non riesco ad attivare le notifiche: '+message,true);
  }finally{busy=false;render()}
 }
 async function turnOff(){
  busy=true;render();
  try{
   const sub=currentSubscription||await getCurrentSubscription();
   if(sub){
    await api('unsubscribe',{endpoint:sub.endpoint});
    await sub.unsubscribe();
   }
   currentSubscription=null;
   status('Notifiche disattivate.');
  }catch(e){
   console.error('PARTITE push disable',e);
   status('Non riesco a disattivare le notifiche. Riprova.',true);
  }finally{busy=false;render()}
 }
 async function test(){
  if(!currentSubscription)return;
  busy=true;render();
  try{
   const result=await api('test',{endpoint:currentSubscription.endpoint});
   if(result.sent>0)status('Notifica di prova inviata. Controlla il tuo iPhone.');
   else status('Il servizio non è riuscito a consegnare la notifica. Riprova tra poco.',true);
  }catch(e){
   status('Prova non riuscita: '+String(e.message||e),true);
  }finally{busy=false;button.disabled=false;testButton.disabled=false}
 }
 button.addEventListener('click',()=>{
  if(busy)return;
  if(currentSubscription&&pushCapable()&&Notification.permission==='granted')turnOff();
  else turnOn();
 });
 testButton.addEventListener('click',()=>{if(!busy)test()});
 document.addEventListener('visibilitychange',()=>{
  if(document.visibilityState==='visible'&&!busy)refresh();
 });
 // Notification click deep links: show the precise day and, if requested, shared events.
 const params=new URLSearchParams(window.location.search);
 const day=params.get('day');
 const section=params.get('section');
 if(day&&/^\d{4}-\d{2}-\d{2}$/.test(day)&&typeof setCalendarOpen==='function'){
  setCalendarOpen(true,day);
 }
 if(section==='events'&&typeof chooseSection==='function')chooseSection('events',false);
 refresh();
})();