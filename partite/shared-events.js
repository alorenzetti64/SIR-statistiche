'use strict';
/* PARTITE · EVENTI CONDIVISI, separati da CASA e dal calendario dei match */
const SHARED_ENDPOINT='https://vqrpkkrqynvlzjufiocn.supabase.co/rest/v1/partite_shared_events';
const SHARED_PUBLIC_KEY='sb_publishable_RGTsinQvL4hyv94mzNgNHA_uKqOfDuf';
const SHARED_HEADERS={'apikey':SHARED_PUBLIC_KEY,'Accept':'application/json','Content-Type':'application/json'};
let sharedEvents=[];
let sharedLoading=true;
let sharedLoadError='';
let activeSection='matches';
let sharedRequestInProgress=false;
let toastTimer=null;
const eventDate=s=>atDay(s);
const eventEnds=e=>e.end_date||e.start_date;
const eventIsPast=e=>eventEnds(e)<todayKey();
const eventStartTime=e=>(e.start_time||'').slice(0,5);
const eventEndTime=e=>(e.end_time||'').slice(0,5);
const eventDateShort=s=>new Intl.DateTimeFormat('it-IT',{day:'numeric',month:'short',year:'numeric'}).format(eventDate(s));
function eventOnDay(e,date){return e.start_date<=date&&eventEnds(e)>=date}
function eventsOnDay(date){return sharedEvents.filter(e=>eventOnDay(e,date))}
function eventDuration(e){
 if(e.all_day){
  return e.start_date===eventEnds(e)?'Tutto il giorno':'Dal '+eventDateShort(e.start_date)+' al '+eventDateShort(eventEnds(e))+' · Tutto il giorno';
 }
 const from=eventStartTime(e);
 const to=eventEndTime(e);
 if(e.start_date!==eventEnds(e))return eventDateShort(e.start_date)+' · '+from+' → '+eventDateShort(eventEnds(e))+(to?' · '+to:'');
 return eventDateShort(e.start_date)+(from?' · '+from:'')+(to?'–'+to:'');
}
function eventDayTime(e,day){
 if(e.all_day)return'Tutto il giorno';
 if(e.start_date===eventEnds(e))return(eventStartTime(e)||'')+(eventEndTime(e)?' – '+eventEndTime(e):'');
 if(day===e.start_date)return'Dalle '+(eventStartTime(e)||'00:00');
 if(day===eventEnds(e))return'Fino alle '+(eventEndTime(e)||'23:59');
 return'Evento in corso';
}
function sharedEventCard(e){
 const month=shortMonthFmt.format(eventDate(e.start_date)).replace('.','').toUpperCase();
 return '<article class="shared-event-card '+(eventIsPast(e)?'shared-past':'')+'">'+
 '<div class="shared-event-date"><strong>'+esc(e.start_date.slice(8))+'</strong><span>'+esc(month)+'</span></div>'+
 '<div class="shared-event-info"><span class="shared-event-type">EVENTO</span><h3>'+esc(e.title)+'</h3>'+
 '<p>'+esc(eventDuration(e))+(e.location?' · '+esc(e.location):'')+'</p>'+
 (e.description?'<p class="shared-description">'+esc(e.description)+'</p>':'')+'</div>'+
 '<div class="shared-event-buttons"><button type="button" data-edit-event="'+esc(e.id)+'" aria-label="Modifica evento '+esc(e.title)+'">Modifica</button>'+
 '<button type="button" class="remove-event" data-remove-event="'+esc(e.id)+'" aria-label="Elimina evento '+esc(e.title)+'">Elimina</button></div></article>';
}
function renderSharedEvents(){
 const list=el('sharedEventsList'),badge=el('sharedEventsBadge');
 const ordered=[...sharedEvents].sort((a,b)=>{
  const ap=eventIsPast(a)?1:0,bp=eventIsPast(b)?1:0;
  if(ap!==bp)return ap-bp;
  return (a.start_date+' '+eventStartTime(a)).localeCompare(b.start_date+' '+eventStartTime(b));
 });
 badge.textContent=sharedEvents.length?String(sharedEvents.length):'';
 if(activeSection==='events')el('upcomingCount').textContent=sharedEvents.length+(sharedEvents.length===1?' EVENTO':' EVENTI');
 if(sharedLoadError){list.innerHTML='<div class="events-error">'+esc(sharedLoadError)+' <button type="button" id="retrySharedEvents">Riprova</button></div>';return}
 if(sharedLoading){list.innerHTML='<div class="events-empty">Caricamento eventi...</div>';return}
 if(!ordered.length){list.innerHTML='<div class="events-empty">Nessun evento condiviso.</div>';return}
 list.innerHTML=ordered.map(sharedEventCard).join('');
}
function chooseSection(section,scroll=true){
 activeSection=section==='events'?'events':'matches';
 const matches=activeSection==='matches';
 el('showMatchesSection').classList.toggle('active',matches);
 el('showSharedSection').classList.toggle('active',!matches);
 el('showMatchesSection').setAttribute('aria-pressed',String(matches));
 el('showSharedSection').setAttribute('aria-pressed',String(!matches));
 el('programList').classList.toggle('hidden',!matches);
 el('sharedSection').classList.toggle('hidden',matches);
 el('sectionTitle').textContent=matches?'Partite in programma':'Eventi condivisi';
 el('upcomingCount').textContent=matches?futureGames().length+' GARE':sharedEvents.length+(sharedEvents.length===1?' EVENTO':' EVENTI');
 if(!matches)renderSharedEvents();
 if(scroll)el('programHeading').scrollIntoView({behavior:'smooth',block:'start'});
}
function notifyShared(message){
 const toast=el('eventToast');toast.textContent=message;toast.classList.remove('hidden');
 if(toastTimer)clearTimeout(toastTimer);
 toastTimer=setTimeout(()=>toast.classList.add('hidden'),3200);
}
async function sharedRequest(path='',options={}){
 const res=await fetch(SHARED_ENDPOINT+path,{
  ...options,
  headers:{...SHARED_HEADERS,...(options.headers||{})},
  cache:'no-store'
 });
 if(!res.ok){
  const data=await res.json().catch(()=>({}));
  throw Error(data.message||data.error||'Impossibile comunicare con l’archivio eventi ('+res.status+').');
 }
 if(res.status===204)return null;
 const text=await res.text();
 return text?JSON.parse(text):null;
}
async function loadSharedEvents(options={}){
 if(sharedRequestInProgress)return;
 sharedRequestInProgress=true;
 const silent=!!options.silent;
 if(!silent){sharedLoading=true;sharedLoadError='';renderSharedEvents()}
 try{
  const rows=await sharedRequest('?select=*&order=start_date.asc,start_time.asc&limit=500');
  if(!Array.isArray(rows))throw Error('Dati non disponibili');
  sharedEvents=rows.filter(e=>e&&typeof e.title==='string'&&typeof e.start_date==='string');
  sharedLoadError='';
  sharedLoading=false;
  renderSharedEvents();
  if(typeof drawCalendar==='function')drawCalendar();
 }catch(e){
  sharedLoading=false;sharedLoadError='Non riesco a caricare gli eventi.';
  renderSharedEvents();
  if(!silent)console.error('Shared events load',e);
 }finally{sharedRequestInProgress=false}
}
function showSharedModal(event=null){
 const form=el('sharedEventForm');form.reset();
 el('eventId').value=event?.id||'';
 el('eventTitle').value=event?.title||'';
 el('eventStartDate').value=event?.start_date||todayKey();
 el('eventStartTime').value=eventStartTime(event||{});
 el('eventEndDate').value=event?.end_date||'';
 el('eventEndTime').value=eventEndTime(event||{});
 el('eventAllDay').checked=!!event?.all_day;
 el('eventLocation').value=event?.location||'';
 el('eventDescription').value=event?.description||'';
 el('eventDialogTitle').textContent=event?'Modifica evento':'Nuovo evento';
 el('saveSharedEvent').textContent=event?'Salva modifiche':'Salva evento';
 el('deleteSharedEvent').classList.toggle('hidden',!event);
 el('eventStatus').textContent='';
 updateAllDayInputState();
 el('eventOverlay').classList.remove('hidden');
 document.body.classList.add('event-modal-open');
 el('eventTitle').focus();
}
function closeSharedModal(){
 if(el('saveSharedEvent').disabled)return;
 el('eventOverlay').classList.add('hidden');
 document.body.classList.remove('event-modal-open');
 el('eventStatus').textContent='';
}
function updateAllDayInputState(){
 const allDay=el('eventAllDay').checked;
 for(const id of ['eventStartTime','eventEndTime'])el(id).disabled=allDay;
 el('eventStartTime').required=!allDay;
 el('eventOverlay').querySelector('.event-dialog').classList.toggle('is-all-day',allDay);
}
function nextDay(date){
 const d=eventDate(date);d.setDate(d.getDate()+1);return dateKey(d);
}
function addTwoHours(startDate,startTime){
 const d=new Date(startDate+'T'+startTime+':00');
 d.setHours(d.getHours()+2);
 return {date:dateKey(d),time:String(d.getHours()).padStart(2,'0')+':'+String(d.getMinutes()).padStart(2,'0')};
}
function sharedFormData(){
 const title=el('eventTitle').value.trim();
 const start_date=el('eventStartDate').value;
 const all_day=el('eventAllDay').checked;
 const start_time=all_day?null:el('eventStartTime').value||null;
 let end_date=el('eventEndDate').value||start_date;
 let end_time=all_day?null:el('eventEndTime').value||null;
 if(!title||!start_date)throw Error('Inserisci titolo e data di inizio.');
 if(!all_day&&!start_time)throw Error('Inserisci l’ora di inizio oppure seleziona Tutto il giorno.');
 if(end_date<start_date)throw Error('La data di fine non può precedere quella di inizio.');
 if(!all_day&&start_time){
  if(!end_time){
   const auto=addTwoHours(start_date,start_time);
   end_time=auto.time;
   if(!el('eventEndDate').value)end_date=auto.date;
  }else if(end_date===start_date&&end_time<start_time){
   // Come CASA: l'orario prima dell'inizio indica un evento che termina il giorno dopo.
   if(!el('eventEndDate').value)end_date=nextDay(start_date);
   else throw Error('L’ora di fine precede l’inizio: scegli il giorno successivo come data di fine.');
  }
 }
 return {
  title,start_date,start_time,end_date,end_time,all_day,
  location:el('eventLocation').value.trim()||null,
  description:el('eventDescription').value.trim()||null,
  updated_at:new Date().toISOString()
 };
}
async function submitSharedEvent(ev){
 ev.preventDefault();
 const status=el('eventStatus'),save=el('saveSharedEvent');
 let row;try{row=sharedFormData()}catch(e){status.textContent=e.message;return}
 const id=el('eventId').value;
 save.disabled=true;status.textContent='Salvataggio...';
 try{
  const data=JSON.stringify(row);
  if(id)await sharedRequest('?id=eq.'+encodeURIComponent(id),{method:'PATCH',body:data,headers:{Prefer:'return=minimal'}});
  else await sharedRequest('',{method:'POST',body:data,headers:{Prefer:'return=minimal'}});
  save.disabled=false;closeSharedModal();notifyShared(id?'Evento aggiornato':'Evento condiviso');
  chooseSection('events',false);
  await loadSharedEvents();
 }catch(e){save.disabled=false;status.textContent=e.message;console.error('Shared save',e)}
}
async function removeSharedEvent(id){
 const e=sharedEvents.find(x=>x.id===id);if(!e)return;
 if(!confirm('Eliminare l’evento «'+e.title+'»?'))return;
 try{
  await sharedRequest('?id=eq.'+encodeURIComponent(id),{method:'DELETE',headers:{Prefer:'return=minimal'}});
  if(!el('eventOverlay').classList.contains('hidden'))closeSharedModal();
  notifyShared('Evento eliminato');
  await loadSharedEvents();
 }catch(error){notifyShared('Non riesco a eliminare l’evento.');console.error('Shared delete',error)}
}
el('createSharedEvent').addEventListener('click',()=>showSharedModal());
el('closeEventDialog').addEventListener('click',closeSharedModal);
el('cancelSharedEvent').addEventListener('click',closeSharedModal);
el('eventOverlay').addEventListener('click',e=>{if(e.target===el('eventOverlay'))closeSharedModal()});
el('eventAllDay').addEventListener('change',updateAllDayInputState);
el('eventStartDate').addEventListener('change',()=>{
 const sd=el('eventStartDate').value,ed=el('eventEndDate');
 if(ed.value&&ed.value<sd)ed.value=sd;
});
el('sharedEventForm').addEventListener('submit',submitSharedEvent);
el('deleteSharedEvent').addEventListener('click',()=>removeSharedEvent(el('eventId').value));
el('showMatchesSection').addEventListener('click',()=>chooseSection('matches'));
el('showSharedSection').addEventListener('click',()=>chooseSection('events'));
el('sharedEventsList').addEventListener('click',e=>{
 if(e.target.id==='retrySharedEvents'){loadSharedEvents();return}
 const edit=e.target.closest('[data-edit-event]');
 if(edit){const event=sharedEvents.find(x=>x.id===edit.dataset.editEvent);if(event)showSharedModal(event);return}
 const del=e.target.closest('[data-remove-event]');
 if(del)removeSharedEvent(del.dataset.removeEvent);
});
el('dayPanel').addEventListener('click',e=>{
 const edit=e.target.closest('[data-edit-event]');
 if(edit){const event=sharedEvents.find(x=>x.id===edit.dataset.editEvent);if(event)showSharedModal(event)}
});
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!el('eventOverlay').classList.contains('hidden'))closeSharedModal()});
document.addEventListener('visibilitychange',()=>{if(document.visibilityState==='visible')loadSharedEvents({silent:true})});
window.setInterval(()=>{if(document.visibilityState==='visible')loadSharedEvents({silent:true})},60000);
renderSharedEvents();
loadSharedEvents();
