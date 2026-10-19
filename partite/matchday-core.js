'use strict';
const el=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const longFmt=new Intl.DateTimeFormat('it-IT',{weekday:'long',day:'numeric',month:'long',year:'numeric'});
const monthFmt=new Intl.DateTimeFormat('it-IT',{month:'long',year:'numeric'});
const shortMonthFmt=new Intl.DateTimeFormat('it-IT',{month:'short'});
const weekdayFmt=new Intl.DateTimeFormat('it-IT',{weekday:'short'});
const dayFmt=new Intl.DateTimeFormat('it-IT',{weekday:'long'});
const atDay=s=>new Date(Number(s.slice(0,4)),Number(s.slice(5,7))-1,Number(s.slice(8,10)),12);
const dateKey=d=>d.getFullYear()+'-'+String(d.getMonth()+1).padStart(2,'0')+'-'+String(d.getDate()).padStart(2,'0');
const todayKey=()=>dateKey(new Date());
const niceDate=s=>longFmt.format(atDay(s));
function normalize(input){
 if(!Array.isArray(input))return [];
 const done=new Set();
 return input.filter(g=>g&&typeof g.title==='string'&&/^\d{4}-\d{2}-\d{2}$/.test(g.date))
 .map(g=>({date:g.date,time:/^\d{2}:\d{2}$/.test(g.time||'')?g.time:'00:00',title:g.title,kind:String(g.kind||'Partita'),id:String(g.id||''),uncertain:!!g.uncertain}))
 .filter(g=>{const key=g.id||g.title+g.date+g.time;if(done.has(key))return false;done.add(key);return true})
 .sort((a,b)=>(a.date+' '+a.time).localeCompare(b.date+' '+b.time));
}
let initial=[];
try{initial=JSON.parse(el('match-data').textContent).matches||[]}catch(e){console.warn('Dati partite incorporati non disponibili',e)}
let games=normalize(initial);
const future=g=>new Date(g.date+'T'+g.time+':00').getTime()>=Date.now();
const futureGames=()=>games.filter(future);
let chosenDate=futureGames()[0]?.date||todayKey();
let monthCursor=atDay(chosenDate);monthCursor.setDate(1);
let calendarOpen=false;
function klass(g){
 const s=g.kind.toLowerCase();
 if(s.includes('champions'))return'champions';
 if(s.includes('coppa')||s.includes('mondiale')||s.includes('world'))return'coppa';
 if(s.includes('amichevole'))return'amichevole';
 return'superlega';
}
function matchTitle(t){
 let s=String(t).trim();
 s=s.replace(/\s*(?:ore|alle)\s*\d{1,2}(?:[:.]\d{2})?\b.*$/i,'').replace(/\s+\d{1,2}[:.]\d{2}\b.*$/,'').trim();
 s=s.replace(/\s*[-–—]?\s*inizio riscaldamento\b.*$/i,'').trim();
 const colon=s.indexOf(':');
 if(colon!==-1)s=s.slice(colon+1).trim();
 return s||t;
}
function fixture(g){
 const title=matchTitle(g.title);
 const h=title.match(/\bPerugia\s*[-–—]\s*(.+)$/i);
 if(h)return{left:'PERUGIA',right:h[1].trim().toUpperCase(),venue:'In casa',versus:true};
 const a=title.match(/^(.+?)\s*[-–—]\s*Perugia\b/i);
 if(a)return{left:a[1].trim().toUpperCase(),right:'PERUGIA',venue:'In trasferta',versus:true};
 return{left:title.toUpperCase(),right:'',venue:'',versus:false};
}
function relativeDate(s){
 const d=atDay(s),now=atDay(todayKey());
 const n=Math.round((Date.UTC(d.getFullYear(),d.getMonth(),d.getDate())-Date.UTC(now.getFullYear(),now.getMonth(),now.getDate()))/86400000);
 if(n===0)return'Oggi';if(n===1)return'Domani';if(n>1)return'Tra '+n+' giorni';return n===-1?'Ieri':Math.abs(n)+' giorni fa';
}
function drawHero(){
 const g=futureGames()[0],root=el('mainHero'),title=el('heroText'),date=el('heroDate');
 root.classList.toggle('no-game',!g);
 if(!g){
  title.innerHTML='<div class="kicker hero-type"><span class="kicker-line"></span> LE PARTITE</div><h1 class="hero-fixture">STAGIONE 2026/27</h1><div class="hero-pills"><span class="hero-tag primary">PROGRAMMA</span></div>';
  date.innerHTML='<span class="day hero-end-season">26/27</span>';
  return;
 }
 const f=fixture(g),dt=atDay(g.date);
 const headline=f.versus?'<h1 class="hero-fixture"><span>'+esc(f.left)+'</span><span class="hero-versus">VS</span><span class="opponent">'+esc(f.right)+'</span></h1>':'<h1 class="hero-fixture"><span>'+esc(f.left)+'</span></h1>';
 title.innerHTML='<div class="kicker hero-type"><span class="kicker-line"></span> PROSSIMA PARTITA <span class="hero-rel">'+esc(relativeDate(g.date).toUpperCase())+'</span></div>'+
 headline+'<div class="hero-pills"><span class="hero-tag primary">'+esc(g.kind.toUpperCase())+'</span>'+
 (f.venue?'<span class="hero-tag">'+esc(f.venue.toUpperCase())+'</span>':'')+
 (g.uncertain?'<span class="hero-tag">DA CONFERMARE</span>':'')+'</div>';
 date.innerHTML='<span class="day">'+esc(g.date.slice(8))+'</span><div class="month">'+esc(shortMonthFmt.format(dt).replace('.','').toUpperCase())+'</div><div class="weekday">'+esc(dayFmt.format(dt).toUpperCase())+'</div><div class="hero-hour">'+esc(g.time)+'</div>';
}
function drawCount(){const counter=el('matchCount');if(counter)counter.textContent=games.length+' PARTITE'}
