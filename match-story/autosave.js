/* Bozze protette: salvataggio cloud indipendente dalla pubblicazione PLAYERS. */
(function(){
'use strict';
const NATIVE_PREFIX='sir-match-story-native-autosave-v1:';
const DB_NAME='sir-match-story-local-recovery-v1';
let databasePromise;
function db(){
 if(!('indexedDB' in window))return Promise.reject(Error('IndexedDB non disponibile'));
 if(!databasePromise)databasePromise=new Promise((resolve,reject)=>{
  const req=indexedDB.open(DB_NAME,1);
  req.onupgradeneeded=()=>{if(!req.result.objectStoreNames.contains('presentations'))req.result.createObjectStore('presentations')};
  req.onsuccess=()=>resolve(req.result);
  req.onerror=()=>reject(req.error);
 });
 return databasePromise;
}
async function storageAction(type,key,value){
 const database=await db();
 return new Promise((resolve,reject)=>{
  const tx=database.transaction('presentations',type==='read'?'readonly':'readwrite');
  const store=tx.objectStore('presentations');
  const request=type==='read'?store.get(key):type==='put'?store.put(value,key):store.delete(key);
  request.onsuccess=()=>{if(type==='read')resolve(request.result||null)};
  tx.oncomplete=()=>{if(type!=='read')resolve(true)};
  tx.onerror=()=>reject(tx.error||Error('Archiviazione locale non riuscita'));
  tx.onabort=()=>reject(tx.error||Error('Archiviazione locale interrotta'));
 });
}
const presentation={
 read:id=>storageAction('read',id).catch(()=>null),
 backup:(id,pages)=>storageAction('put',id,{savedAt:new Date().toISOString(),pages}).catch(()=>false),
 clear:id=>storageAction('delete',id).catch(()=>false)
};
const native={
 pending(reportId){
  try{const raw=localStorage.getItem(NATIVE_PREFIX+reportId);return raw?JSON.parse(raw):null}catch{return null}
 },
 clear(reportId){try{localStorage.removeItem(NATIVE_PREFIX+reportId)}catch{}},
 create({getTitle,getHtml,setTitle,setHtml,status,save,onSaved}){
  let context=null,version=0,syncedVersion=0,timer=null,retryTimer=null,working=null;
  let observer=null,paused=false,localFailed=false;
  const notify=value=>{if(status)status.textContent=value};
  function snapshot(){
   if(!context)return null;
   return {...context,title:getTitle(),content:getHtml(),savedAt:new Date().toISOString()};
  }
  function preserve(){
   if(!context||version===syncedVersion)return;
   try{localStorage.setItem(NATIVE_PREFIX+context.reportId,JSON.stringify(snapshot()));localFailed=false}
   catch{localFailed=true;notify('Bozza troppo grande per la copia locale; attendi “Salvato online” prima di chiudere.')}
  }
  function queue(delay=2400){
   clearTimeout(timer);
   timer=setTimeout(()=>{flush().catch(()=>{})},delay);
  }
  function mark(){
   if(!context||paused)return;
   version++;
   preserve();
   if(!localFailed)notify('Modifiche in attesa di salvataggio…');
   queue();
  }
  async function flush(){
   clearTimeout(timer);timer=null;
   if(!context||version===syncedVersion)return true;
   if(working){await working;if(version!==syncedVersion)return flush();return true}
   const current=snapshot();
   if(!current.title.trim()){
    notify('Scrivi un titolo per salvare automaticamente il capitolo.');
    return false;
   }
   const currentVersion=version, currentContext=context;
   notify('Salvataggio automatico online…');
   const promise=(async()=>{
    try{
     await save({id:current.reportId,owner:current.owner,section_id:current.sectionId,title:current.title.trim(),content:current.content,position:current.position});
     if(context!==currentContext)return false;
     syncedVersion=currentVersion;
     await onSaved?.(current);
     if(version===syncedVersion){
      native.clear(context.reportId);
      notify('✓ Salvato online · '+new Date().toLocaleTimeString('it-IT',{hour:'2-digit',minute:'2-digit'}));
     }else{preserve();queue(900)}
     clearTimeout(retryTimer);
     return true;
    }catch(e){
     if(context===currentContext){
      notify('⚠ Salvataggio online non riuscito: '+(e.message||'errore di rete')+'. Copia locale conservata.');
      clearTimeout(retryTimer);
      retryTimer=setTimeout(()=>{flush().catch(()=>{})},20000);
     }
     return false;
    }finally{working=null}
   })();
   working=promise;
   return promise;
  }
  function begin(data){
   clearTimeout(timer);clearTimeout(retryTimer);
   context={reportId:data.reportId,owner:data.owner,sectionId:data.sectionId,position:data.position};
   version=syncedVersion=0;
   paused=true;
   const pending=native.pending(data.reportId);
   let restored=false;
   if(pending&&pending.owner===data.owner&&pending.sectionId===data.sectionId&&
      Date.parse(pending.savedAt||0)>Date.parse(data.updatedAt||0)){
     setTitle(pending.title||'');setHtml(pending.content||'');restored=true;
     version=1;
   }else if(pending&&pending.owner===data.owner&&pending.sectionId===data.sectionId){
     native.clear(data.reportId);
   }
   paused=false;
   observer?.disconnect();
   observer=new MutationObserver(()=>mark());
   observer.observe(data.editor,{subtree:true,childList:true,characterData:true,attributes:true,attributeFilter:['style','class','src','data-card-title']});
   if(restored){notify('Bozza recuperata dal dispositivo · sincronizzazione in corso…');queue(350)}
   else notify('Salvataggio automatico attivo');
   return restored;
  }
  function end(){clearTimeout(timer);clearTimeout(retryTimer);observer?.disconnect();observer=null;context=null;version=syncedVersion=0}
  function beforeUnload(e){
   if(!context)return;
   if(version!==syncedVersion){preserve();e.preventDefault();e.returnValue=''}
  }
  document.addEventListener('visibilitychange',()=>{if(document.hidden&&context&&version!==syncedVersion){preserve();flush().catch(()=>{})}});
  window.addEventListener('pagehide',preserve);
  window.addEventListener('beforeunload',beforeUnload);
  return {begin,end,mark,flush,hasPending:()=>!!context&&version!==syncedVersion,pending:native.pending,clear:native.clear};
 }
};
window.SirMatchAutosave={native,presentation};
})();
