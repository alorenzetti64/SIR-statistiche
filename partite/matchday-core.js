'use strict';
const CAL_ID='p2au0socbhks3n0sp29i9iumvg@group.calendar.google.com';
const el=(id)=>document.getElementById(id);
const esc=(s)=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const longFmt=new Intl.DateTimeFormat('it-IT',{weekday:'long',day:'numeric',month:'long',year:'numeric'});
const monthFmt=new Intl.DateTimeFormat('it-IT',{month:'long',year:'numeric'});
const shortMonthFmt=new Intl.DateTimeFormat('it-IT',{month:'short'});
const weekdayFmt=new Intl.DateTimeFormat('it-IT',{weekday:'short'});
const dayFmt=new Intl.DateTimeFormat('it-IT',{weekday:'long'});
const atDay=(v)=>new Date(Number(v.slice(0,4)),Number(v.slice(5,7))-1,Number(v.slice(8,10)),12);
const dateKey=(d)=>d.getFullYear()+'-'+String(d.getMonth()+1).padStart(2,'0')+'-'+String(d.getDate()).padStart(2,'0');
const todayKey=()=>dateKey(new Date());
const niceDate=(s)=>longFmt.format(atDay(s));
function normalize(input){
 if(!Array.isArray(input))return [];
 const done=new Set();
 return input.filter(g=>g&&typeof g.title==='string'&&/^\d{4}-\d{2}-\d{2}$/.test(g.date))
 .map(g=>({date:g.date,time:/^\d{2}:\d{2}$/.test(g.time||'')?g.time:'00:00',title:g.title,kind:String(g.kind||'Partita'),id:String(g.id||''),uncertain:!!g.uncertain}))
 .filter(g=>{const id=g.id||g.title+g.date+g.time;if(done.has(id))return false;done.add(id);return true})
 .sort((a,b)=>(a.date+' '+a.time).localeCompare(b.date+' '+b.time));
}
let initial=[];
try{initial=JSON.parse(el('match-data').textContent).matches||[]}catch{}
let games=normalize(initial);
const future=(g)=>new Date(g.date+'T'+g.time+':00').getTime()>=Date.now();
const futureGames=()=>games.filter(future);
let chosenDate=futureGames()[0]?.date||todayKey();
let monthCursor=atDay(chosenDate);
monthCursor.setDate(1);
let viewMode='calendar',scope='future',competition='all',searchTerm='';
function klass(g){
 const s=g.kind.toLowerCase();
 if(s.includes('champions'))return 'champions';
 if(s.includes('coppa')||s.includes('mondiale')||s.includes('world'))return 'coppa';
 if(s.includes('amichevole'))return 'amichevole';
 return 'superlega';
}
function matchTitle(title){
 let s=String(title).trim();
 if(s.includes(':'))s=s.slice(s.lastIndexOf(':')+1).trim();
 return s.replace(/\s*(?:ore|alle)\s*\d{1,2}(?:[:.]\d{2})?\b.*$/i,'').trim()||title;
}
function fixture(g){
 const text=matchTitle(g.title);
 const home=text.match(/\bPerugia\s*[-–—]\s*(.+)$/i);
 if(home)return{left:'PERUGIA',right:home[1].trim().toUpperCase(),venue:'In casa'};
 const away=text.match(/^(.+?)\s*[-–—]\s*Perugia\b/i);
 if(away)return{left:away[1].trim().toUpperCase(),right:'PERUGIA',venue:'In trasferta'};
 return{left:'PERUGIA',right:text.toUpperCase(),venue:'Partita'};
}
function linkGame(g){
 if(!g.id)return 'https://calendar.google.com/calendar/u/0/r/day/'+g.date.replaceAll('-','/');
 try{return 'https://www.google.com/calendar/event?eid='+encodeURIComponent(btoa(g.id+' '+CAL_ID).replace(/=+$/,''))+'&ctz=Europe/Rome'}
 catch{return 'https://calendar.google.com/calendar/u/0/r'}
}
function relativeDate(s){
 const d=atDay(s),now=atDay(todayKey());
 const n=Math.round((Date.UTC(d.getFullYear(),d.getMonth(),d.getDate())-Date.UTC(now.getFullYear(),now.getMonth(),now.getDate()))/86400000);
 if(n===0)return'Oggi';if(n===1)return'Domani';if(n>1)return'Tra '+n+' giorni';
 return n===-1?'Ieri':Math.abs(n)+' giorni fa';
}
function drawHero(){
 const game=futureGames()[0],root=el('mainHero'),title=el('heroText'),date=el('heroDate');
 root.classList.toggle('no-game',!game);
 if(!game){
  title.innerHTML='<div class="kicker hero-type"><span class="kicker-line"></span> IL PROGRAMMA</div><h1>OGNI PARTITA.<br><span class="vs">UNA STORIA.</span></h1><p class="hero-desc">La stagione sportiva, un appuntamento alla volta.</p><div class="hero-pills"><span class="hero-tag primary">STAGIONE 2026—27</span></div>';
  date.innerHTML='<span class="day">26/27</span><div class="month">MATCHDAY</div>';return;
 }
 const f=fixture(game),dt=atDay(game.date);
 title.innerHTML='<div class="kicker hero-type"><span class="kicker-line"></span> PROSSIMA PARTITA · '+esc(relativeDate(game.date).toUpperCase())+'</div>'+
 '<h1>'+esc(f.left)+'<br><span class="vs">VS</span> <span class="opponent">'+esc(f.right)+'</span></h1>'+
 '<p class="hero-desc">'+esc(game.title)+'</p><div class="hero-pills">'+
 '<span class="hero-tag primary">'+esc(game.kind.toUpperCase())+'</span>'+
 '<span class="hero-tag">'+esc(f.venue.toUpperCase())+'</span>'+
 '<span class="hero-tag">ORE '+esc(game.time)+'</span></div>';
 date.innerHTML='<span class="day">'+esc(game.date.slice(8))+'</span><div class="month">'+esc(shortMonthFmt.format(dt).replace('.','').toUpperCase())+'</div><div class="weekday">'+esc(dayFmt.format(dt).toUpperCase())+'</div>';
}
function drawStats(){
 const next=futureGames(),thisMonth=todayKey().slice(0,7);
 const numbers=[
  {head:'Gare in programma',value:next.length,foot:'Da oggi in avanti'},
  {head:'Questo mese',value:next.filter(g=>g.date.slice(0,7)===thisMonth).length,foot:monthFmt.format(new Date()).toUpperCase()},
  {head:'Champions future',value:next.filter(g=>klass(g)==='champions').length,foot:'Nel calendario attuale'}
 ];
 el('stats').innerHTML=numbers.map(x=>'<article class="mini-stat"><div class="top">'+esc(x.head)+'</div><div class="value">'+String(x.value).padStart(2,'0')+'</div><div class="foot">'+esc(x.foot)+'</div></article>').join('');
 el('matchCount').textContent=games.length+' PARTITE';
}
function drawPreview(){
 const next=futureGames().slice(0,3);
 el('preview').innerHTML=next.length?next.map(g=>{
  const f=fixture(g);
  return '<a class="preview-card" href="'+esc(linkGame(g))+'" target="_blank" rel="noopener noreferrer" aria-label="Apri '+esc(g.title)+' su Google Calendar"><div class="card-date"><div class="small-date">'+esc(g.date.slice(8))+'<span>'+esc(shortMonthFmt.format(atDay(g.date)).replace('.','').toUpperCase())+'</span></div><span class="tag">'+esc(g.kind)+'</span></div><h3>'+esc(matchTitle(g.title))+'</h3><div class="meta"><span>'+esc(f.venue)+' · '+esc(g.time)+'</span><span class="arrow">↗</span></div></a>';
 }).join(''):'<div class="preview-empty">Nessuna gara futura presente nel programma importato.</div>';
}
