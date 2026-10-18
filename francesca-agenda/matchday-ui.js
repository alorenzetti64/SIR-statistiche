'use strict';
function drawCalendar(){
 const y=monthCursor.getFullYear(),m=monthCursor.getMonth();
 el('monthTitle').textContent=monthFmt.format(monthCursor);
 const first=new Date(y,m,1,12),start=(first.getDay()+6)%7;
 const slots=Math.ceil((start+new Date(y,m+1,0).getDate())/7)*7;
 const counter=games.filter(g=>g.date.slice(0,7)===dateKey(first).slice(0,7)).length;
 el('monthCount').textContent=counter+(counter===1?' partita nel mese':' partite nel mese');
 let html='';
 for(let i=0;i<slots;i++){
  const date=new Date(y,m,i-start+1,12),key=dateKey(date),daily=games.filter(g=>g.date===key);
  const selected=key===chosenDate;
  const classes=['date-cell',selected?'selected':'',key===todayKey()?'is-today':'',date.getMonth()!==m?'other-month':'',daily.length?'has-match':''].filter(Boolean).join(' ');
  const marks=daily.slice(0,4).map(g=>'<i class="mark '+klass(g)+'"></i>').join('');
  const label=longFmt.format(date)+(daily.length?', '+daily.length+(daily.length===1?' partita':' partite'):'');
  html+='<button type="button" class="'+classes+'" data-day="'+key+'" aria-label="'+esc(label)+'" aria-pressed="'+selected+'"><span class="date-number">'+date.getDate()+'</span><span class="indicators">'+marks+'</span></button>';
 }
 el('calendarDays').innerHTML=html;
 drawDay();
}
function drawDay(){
 const dayGames=games.filter(g=>g.date===chosenDate);
 let html='<span class="day-kicker">GIORNO SELEZIONATO</span><h3>'+esc(longFmt.format(atDay(chosenDate)))+'</h3>'+
 '<div class="day-count">'+(dayGames.length?dayGames.length+(dayGames.length===1?' partita':' partite'):'Nessuna gara in calendario')+'</div>';
 if(dayGames.length){
  html+=dayGames.map(g=>'<div class="day-match"><div class="kind">'+esc(g.kind)+'</div><h4>'+esc(matchTitle(g.title))+'</h4><p>'+esc(g.time)+' · '+esc(fixture(g).venue)+'</p><a href="'+esc(linkGame(g))+'" target="_blank" rel="noopener noreferrer">Apri su Google Calendar ↗</a></div>').join('');
 }else{
  html+='<div class="day-empty"><div class="symbol">—</div>Nessuna partita in questa data. I pallini nel calendario indicano i giorni in cui si gioca.</div>';
 }
 el('dayPanel').innerHTML=html;
}
const typeFilters=[
 ['all','Tutte'],['superlega','SuperLega'],['champions','Champions'],
 ['coppa','Coppe e tornei'],['amichevole','Amichevoli']
];
function drawFilters(){
 el('scopeFilters').innerHTML=[['future','In programma'],['all','Tutte le partite'],['past','Già giocate']]
 .map(x=>'<button type="button" class="pill-filter '+(scope===x[0]?'active':'')+'" data-scope="'+x[0]+'" aria-pressed="'+(scope===x[0])+'">'+x[1]+'</button>').join('');
 el('kindFilters').innerHTML=typeFilters.map(x=>'<button type="button" class="pill-filter '+(competition===x[0]?'active':'')+'" data-kind="'+x[0]+'" aria-pressed="'+(competition===x[0])+'">'+x[1]+'</button>').join('');
}
function shownGames(){
 return games.filter(g=>{
  if(scope==='future'&&!future(g))return false;
  if(scope==='past'&&future(g))return false;
  if(competition!=='all'&&klass(g)!==competition)return false;
  if(searchTerm&&!(g.title+' '+g.kind).toLocaleLowerCase('it').includes(searchTerm))return false;
  return true;
 });
}
function drawProgram(){
 const selected=shownGames();
 el('listTotal').textContent=selected.length+(selected.length===1?' partita':' partite');
 if(!selected.length){el('list').innerHTML='<div class="empty-result"><b>Nessuna partita</b>Modifica i filtri per consultare altre gare.</div>';return}
 const months=new Map();
 for(const g of selected){const month=g.date.slice(0,7);if(!months.has(month))months.set(month,[]);months.get(month).push(g)}
 el('list').innerHTML=[...months].map(([month,group])=>{
  const title='<h3 class="month-group-title">'+esc(monthFmt.format(atDay(month+'-01')))+'</h3>';
  return '<section class="month-group">'+title+'<div class="match-list">'+group.map(g=>{
   const f=fixture(g),past=!future(g),category=klass(g);
   const wd=weekdayFmt.format(atDay(g.date)).replace('.','');
   return '<article class="match-row '+(past?'past':'')+'">'+
    '<div class="match-date"><strong>'+esc(g.date.slice(8))+'</strong><span>'+esc(wd)+'</span></div>'+
    '<div class="match-info"><div class="match-meta"><span class="match-category '+category+'">'+esc(g.kind)+'</span><span>'+esc(f.venue)+'</span>'+(g.uncertain?'<span>Da confermare</span>':'')+'</div>'+
    '<h3>'+esc(matchTitle(g.title))+'</h3><p>'+esc(longFmt.format(atDay(g.date)))+'</p></div>'+
    '<div class="row-right"><div class="match-hour"><strong>'+esc(g.time)+'</strong><span>ore</span></div>'+
    '<a class="match-link" href="'+esc(linkGame(g))+'" target="_blank" rel="noopener noreferrer" title="Apri su Google Calendar" aria-label="Apri '+esc(g.title)+' su Google Calendar">↗</a></div>'+
    '</article>';
  }).join('')+'</div></section>';
 }).join('');
}
function changeView(name){
 viewMode=name==='list'?'list':'calendar';
 document.querySelectorAll('[data-view]').forEach(b=>{
  const active=b.dataset.view===viewMode;
  b.classList.toggle('active',active);
  b.setAttribute('aria-pressed',String(active));
 });
 el('calendarView').classList.toggle('hidden',viewMode!=='calendar');
 el('listView').classList.toggle('hidden',viewMode!=='list');
 if(viewMode==='calendar')drawCalendar();else drawProgram();
}
function drawAll(){
 drawHero();drawStats();drawPreview();drawCalendar();drawFilters();drawProgram();
}
el('prevMonth').addEventListener('click',()=>{monthCursor=new Date(monthCursor.getFullYear(),monthCursor.getMonth()-1,1,12);drawCalendar()});
el('nextMonth').addEventListener('click',()=>{monthCursor=new Date(monthCursor.getFullYear(),monthCursor.getMonth()+1,1,12);drawCalendar()});
el('goToday').addEventListener('click',()=>{chosenDate=todayKey();monthCursor=atDay(chosenDate);monthCursor.setDate(1);drawCalendar()});
el('calendarDays').addEventListener('click',ev=>{
 const b=ev.target.closest('[data-day]');
 if(!b)return;
 chosenDate=b.dataset.day;
 monthCursor=atDay(chosenDate);monthCursor.setDate(1);
 drawCalendar();
});
document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>changeView(b.dataset.view)));
el('scopeFilters').addEventListener('click',ev=>{
 const btn=ev.target.closest('[data-scope]');
 if(!btn)return;
 scope=btn.dataset.scope;drawFilters();drawProgram();
});
el('kindFilters').addEventListener('click',ev=>{
 const btn=ev.target.closest('[data-kind]');
 if(!btn)return;
 competition=btn.dataset.kind;drawFilters();drawProgram();
});
el('searchMatches').addEventListener('input',ev=>{searchTerm=ev.target.value.trim().toLocaleLowerCase('it');drawProgram()});
function jumpTo(view){
 changeView(view);
 el('viewTabs').scrollIntoView({behavior:'smooth',block:'start'});
}
el('showFullList').addEventListener('click',()=>jumpTo('list'));
el('showCalendar').addEventListener('click',()=>jumpTo('calendar'));
drawAll();
changeView('calendar');
(async function reloadPublishedData(){
 try{
  const response=await fetch('./partite.json?v='+Date.now(),{cache:'no-store'});
  if(!response.ok)throw Error('Impossibile leggere il programma');
  const data=await response.json();
  if(!Array.isArray(data.matches)||!data.matches.length)throw Error('Elenco vuoto');
  games=normalize(data.matches);
  if(!games.some(g=>g.date.slice(0,7)===dateKey(monthCursor).slice(0,7))){
   chosenDate=futureGames()[0]?.date||todayKey();
   monthCursor=atDay(chosenDate);monthCursor.setDate(1);
  }
  drawAll();
  const updated=data.updated_at||'2026-10-19';
  const formatted=/^\d{4}-\d{2}-\d{2}$/.test(updated)?niceDate(updated):updated;
  el('dataUpdated').textContent='Fonte: Google Calendar SUPERLEGA · importato il '+formatted;
 }catch(e){console.info('Usata la copia delle partite incorporata');}
})();
