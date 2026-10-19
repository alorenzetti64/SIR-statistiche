'use strict';
function drawCalendar(){
 const y=monthCursor.getFullYear(),m=monthCursor.getMonth();
 el('monthTitle').textContent=monthFmt.format(monthCursor);
 const first=new Date(y,m,1,12),start=(first.getDay()+6)%7;
 const slots=Math.ceil((start+new Date(y,m+1,0).getDate())/7)*7;
 const count=games.filter(g=>g.date.slice(0,7)===dateKey(first).slice(0,7)).length;
 el('monthCount').textContent=count+(count===1?' partita':' partite');
 let html='';
 for(let i=0;i<slots;i++){
  const d=new Date(y,m,i-start+1,12),key=dateKey(d),daily=games.filter(g=>g.date===key);
  const selected=key===chosenDate;
  const classes=['date-cell',selected?'selected':'',key===todayKey()?'is-today':'',d.getMonth()!==m?'other-month':'',daily.length?'has-match':''].filter(Boolean).join(' ');
  const marks=daily.slice(0,4).map(g=>'<i aria-hidden="true" class="mark '+klass(g)+'"></i>').join('');
  const label=longFmt.format(d)+(daily.length?', '+daily.length+(daily.length===1?' partita':' partite'):'');
  html+='<button type="button" class="'+classes+'" data-day="'+key+'" aria-label="'+esc(label)+'" aria-pressed="'+selected+'"><span class="date-number">'+d.getDate()+'</span><span class="indicators">'+marks+'</span></button>';
 }
 el('calendarDays').innerHTML=html;
 drawDay();
}
function drawDay(){
 const dayGames=games.filter(g=>g.date===chosenDate);
 let html='<span class="day-kicker">GIORNO SELEZIONATO</span><h3>'+esc(longFmt.format(atDay(chosenDate)))+'</h3>'+
 '<div class="day-count">'+(dayGames.length?dayGames.length+(dayGames.length===1?' partita':' partite'):'Nessuna partita')+'</div>';
 if(dayGames.length){
  html+=dayGames.map(g=>{
   const f=fixture(g);
   return '<div class="day-match"><div class="kind">'+esc(g.kind)+'</div><h4>'+esc(matchTitle(g.title))+'</h4><p>Ore '+esc(g.time)+(f.venue?' · '+esc(f.venue):'')+(g.uncertain?' · Da confermare':'')+'</p></div>';
  }).join('');
 }else{
  html+='<div class="day-empty"><div class="symbol">—</div>Nessuna partita prevista.</div>';
 }
 el('dayPanel').innerHTML=html;
}
function drawProgram(){
 // L'elenco mostra solo le gare non ancora iniziate.
 // Il calendario usa invece "games" senza filtri, quindi conserva tutte le date.
 const upcoming=futureGames();
 const countLabel=el('upcomingCount');
 if(countLabel)countLabel.textContent=upcoming.length+(upcoming.length===1?' GARA':' GARE');
 const months=new Map();
 for(const g of upcoming){
  const key=g.date.slice(0,7);
  if(!months.has(key))months.set(key,[]);
  months.get(key).push(g);
 }
 if(!upcoming.length){
  el('list').innerHTML='<div class="empty-result"><b>Nessuna partita in programma</b></div>';
  return;
 }
 el('list').innerHTML=[...months].map(([month,items])=>{
  const head='<div class="month-group-head"><h3 class="month-group-title">'+esc(monthFmt.format(atDay(month+'-01')))+'</h3>'+
   '<button type="button" class="month-jump" data-open-month="'+month+'" aria-label="Apri il calendario di '+esc(monthFmt.format(atDay(month+'-01')))+'"><span aria-hidden="true">▦</span> CALENDARIO</button></div>';
  const rows=items.map(g=>{
   const f=fixture(g),past=!future(g),category=klass(g);
   const weekday=weekdayFmt.format(atDay(g.date)).replace('.','');
   return '<button type="button" class="match-row match-action '+(past?'past':'')+'" data-open-day="'+g.date+'" aria-label="Apri nel calendario '+esc(g.title)+', '+esc(niceDate(g.date))+', ore '+esc(g.time)+'">'+
    '<span class="match-date"><strong>'+esc(g.date.slice(8))+'</strong><span>'+esc(weekday)+'</span></span>'+
    '<span class="match-info"><span class="match-meta"><span class="match-category '+category+'">'+esc(g.kind)+'</span>'+
    (f.venue?'<span>'+esc(f.venue)+'</span>':'')+
    (g.uncertain?'<span>Da confermare</span>':'')+'</span>'+
    '<span class="row-title">'+esc(matchTitle(g.title))+'</span></span>'+
    '<span class="row-right"><span class="match-hour"><strong>'+esc(g.time)+'</strong><span>ORE</span></span><span class="match-arrow" aria-hidden="true">›</span></span></button>';
  }).join('');
  return '<section class="month-group" id="month-'+month+'">'+head+'<div class="match-list">'+rows+'</div></section>';
 }).join('');
}
function setCalendarOpen(open,date=null){
 calendarOpen=!!open;
 if(date){
  chosenDate=date.length===7?((games.find(g=>g.date.startsWith(date))||{}).date||date+'-01'):date;
  monthCursor=atDay(chosenDate);monthCursor.setDate(1);
 }
 const c=el('calendarView'),btn=el('calendarToggle');
 c.classList.toggle('hidden',!calendarOpen);
 btn.classList.toggle('active',calendarOpen);
 btn.setAttribute('aria-expanded',String(calendarOpen));
 btn.querySelector('.calendar-toggle-text').textContent=calendarOpen?'Chiudi calendario':'Apri calendario';
 if(calendarOpen){drawCalendar();c.scrollIntoView({behavior:'smooth',block:'start'});}
}
function drawAll(){drawHero();drawCount();drawProgram();drawCalendar()}
el('calendarToggle').addEventListener('click',()=>setCalendarOpen(!calendarOpen));
el('calendarClose').addEventListener('click',()=>setCalendarOpen(false));
el('prevMonth').addEventListener('click',()=>{monthCursor=new Date(monthCursor.getFullYear(),monthCursor.getMonth()-1,1,12);drawCalendar()});
el('nextMonth').addEventListener('click',()=>{monthCursor=new Date(monthCursor.getFullYear(),monthCursor.getMonth()+1,1,12);drawCalendar()});
el('goToday').addEventListener('click',()=>setCalendarOpen(true,todayKey()));
el('calendarDays').addEventListener('click',ev=>{
 const b=ev.target.closest('[data-day]');
 if(!b)return;
 chosenDate=b.dataset.day;
 monthCursor=atDay(chosenDate);monthCursor.setDate(1);
 drawCalendar();
});
el('list').addEventListener('click',ev=>{
 const item=ev.target.closest('[data-open-day]');
 if(item){setCalendarOpen(true,item.dataset.openDay);return}
 const month=ev.target.closest('[data-open-month]');
 if(month)setCalendarOpen(true,month.dataset.openMonth);
});
drawAll();
(async function reloadPublishedData(){
 try{
  const response=await fetch('./partite.json?v='+Date.now(),{cache:'no-store'});
  if(!response.ok)throw Error('Programma non disponibile');
  const data=await response.json();
  if(!Array.isArray(data.matches)||!data.matches.length)throw Error('Elenco vuoto');
  games=normalize(data.matches);
  const next=futureGames()[0];
  if(next&&!calendarOpen){chosenDate=next.date;monthCursor=atDay(chosenDate);monthCursor.setDate(1)}
  drawAll();
  const updated=data.updated_at||'2026-10-19';
  const formatted=/^\d{4}-\d{2}-\d{2}$/.test(updated)?niceDate(updated):updated;
  el('dataUpdated').textContent='SUPERLEGA · Aggiornato il '+formatted;
 }catch(e){console.info('Visualizzato il programma incorporato',e)}
})();
