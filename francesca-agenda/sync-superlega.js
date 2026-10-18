(function(){
"use strict";
const ID="__superlega_ical";
let linked=false;
const style=document.createElement("style");
style.textContent=".feed-line{display:flex;gap:12px;align-items:center;flex-wrap:wrap;border:1px solid #dce7d8;background:#f0f7ef;color:#35614c;border-radius:12px;padding:12px 15px;margin:-12px 0 20px;font-size:12px}.feed-line.warning{background:#fcf9f1;color:#886b35;border-color:#efe1c2}.feed-line b{font-weight:750}.feed-guide{line-height:1.6;color:#556e61;font-size:13px}.feed-guide a{color:#1f634b}.feed-tip{background:#f0f6ef;border-radius:10px;padding:12px}.feed-form-input{width:100%;font:inherit;padding:12px;border:1px solid #cbdccc;border-radius:10px}.feed-controls{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}.feed-controls button{flex:1;min-width:85px}.feed-controls .secondary{background:#eaf1eb;color:#234c37}.feed-controls .danger{background:#fff0ec;color:#9e4938}@media(max-width:620px){.cloud-strip{gap:8px}.cloud-strip button{font-size:11px}}";
document.head.appendChild(style);
const btn=document.createElement("button");
btn.id="feedOpen";btn.type="button";btn.textContent="Collega SUPERLEGA";
document.getElementById("cloudAction").before(btn);
const bar=document.createElement("div");
bar.id="feedBar";bar.className="feed-line warning";
bar.textContent="Partite: copia del calendario SUPERLEGA. Collega il calendario per gli aggiornamenti automatici.";
strip.after(bar);
const modal=document.createElement("div");
modal.id="feedModal";modal.className="auth-screen";
modal.innerHTML='<div class="auth-box" role="dialog" aria-modal="true" aria-labelledby="feedTitle"><div style="display:flex;align-items:center;justify-content:space-between"><b class="kicker">Calendario SUPERLEGA</b><button type="button" id="feedClose" aria-label="Chiudi" style="border:0;border-radius:8px;padding:8px 12px">✕</button></div><h2 id="feedTitle">Sincronizza le partite</h2><p class="feed-guide">L’app legge <b>solo gli eventi contrassegnati come «Match»</b> nel tuo Google Calendar SUPERLEGA. Non inserire altri calendari.</p><p class="feed-guide feed-tip"><b>Per ottenere il collegamento:</b> apri <a href="https://calendar.google.com/calendar/u/0/r/settings" target="_blank" rel="noopener noreferrer">Impostazioni Google Calendar ↗</a>, scegli <b>Superlega</b>, poi <b>Integra calendario</b> e copia <b>Indirizzo segreto in formato iCal</b>.<br><br>Non rendere pubblico il calendario e non incollare l’indirizzo segreto in chat: dà accesso al suo contenuto.</p><form id="feedForm"><label for="feedUrl" style="display:block;font-size:12px;font-weight:750;margin:12px 0 7px">Indirizzo segreto iCal del solo calendario SUPERLEGA</label><input class="feed-form-input" type="password" id="feedUrl" autocomplete="off" placeholder="https://calendar.google.com/calendar/ical/.../private-.../basic.ics" required><div id="feedMsg" class="feed-guide" role="status"></div><div class="feed-controls"><button type="button" class="danger" id="feedUnlink">Scollega</button><button type="button" class="secondary" id="feedCancel">Chiudi</button><button type="submit" id="feedSave">Collega</button></div></form></div>';
document.body.appendChild(modal);
const el=id=>document.getElementById(id);
function status(message,live){bar.textContent=message;bar.classList.toggle("warning",!live);}
function close(){modal.classList.remove("visible");el("feedUrl").value="";}
async function loadConfig(){
 if(!frSession){linked=false;btn.textContent="Collega SUPERLEGA";return}
 try{const r=await frDb("francesca_agenda_events?select=id&id=eq."+ID+"&limit=1");linked=r.length>0;btn.textContent=linked?"SUPERLEGA collegato":"Collega SUPERLEGA";}
 catch(e){linked=false;btn.textContent="Collega SUPERLEGA";}
}
function open(){
 if(!frSession){frOpenAuth();return}
 el("feedUrl").value="";
 el("feedMsg").textContent=linked?"Collegamento già configurato. Puoi sostituire l’indirizzo o scollegarlo.":"Incolla l’indirizzo iCal segreto del calendario SUPERLEGA.";
 modal.classList.add("visible");
}
btn.onclick=open;
el("feedClose").onclick=close;
el("feedCancel").onclick=close;
modal.onclick=e=>{if(e.target===modal)close()};
function isCorrectUrl(value){
 try{
 const u=new URL(value),p=decodeURIComponent(u.pathname);
 const prefix="/calendar/ical/p2au0socbhks3n0sp29i9iumvg@group.calendar.google.com/private-";
 return u.protocol==="https:"&&u.hostname==="calendar.google.com"&&!u.username&&!u.password&&!u.search&&!u.hash&&p.startsWith(prefix)&&/\/private-[a-zA-Z0-9_-]+\/basic\.ics$/.test(p);
 }catch{return false}
}
el("feedForm").onsubmit=async ev=>{
 ev.preventDefault();
 if(!frSession){close();frOpenAuth();return}
 const raw=el("feedUrl").value.trim(),b=el("feedSave");
 if(!isCorrectUrl(raw)){el("feedMsg").textContent="Questo indirizzo non corrisponde al link iCal segreto del tuo calendario SUPERLEGA.";return}
 b.disabled=true;el("feedMsg").textContent="Collegamento in corso…";
 try{
 await frDb("francesca_agenda_events?on_conflict=owner_id,id","POST",{owner_id:frSession.user.id,id:ID,data:{ics_url:raw,source:"SUPERLEGA"}},{"prefer":"resolution=merge-duplicates,return=minimal"});
 linked=true;btn.textContent="SUPERLEGA collegato";close();await frMatches();
 }catch(e){el("feedMsg").textContent="Impossibile salvare: "+e.message}
 finally{b.disabled=false}
};
el("feedUnlink").onclick=async()=>{
 if(!frSession||!linked){close();return}
 if(!confirm("Scollegare SUPERLEGA? Gli eventi personali non saranno modificati."))return;
 try{
 await frDb("francesca_agenda_events?id=eq."+ID,"DELETE");
 linked=false;btn.textContent="Collega SUPERLEGA";matchLive=false;close();
 status("SUPERLEGA scollegato. Partite disponibili nella copia del 19 ottobre 2026.",false);render();
 }catch(e){el("feedMsg").textContent="Impossibile scollegare: "+e.message}
};
const oldSourceNote=sourceNote;
sourceNote=function(){
 const line=matchLive?"<span class=\"source-live\">Partite sincronizzate con Google Calendar SUPERLEGA.</span>":"Elenco iniziale letto dal tuo Google Calendar SUPERLEGA il 19 ottobre 2026. Per ricevere nuove date e modifiche, collega una volta il calendario.";
 return '<p class="footnote">'+line+' Sono inclusi soltanto gli eventi contrassegnati «Match». <a href="https://calendar.google.com/calendar/u/0/r" target="_blank" rel="noopener noreferrer">Apri Google Calendar ↗</a>.</p>';
};
frMatches=async function(){
 if(!frSession?.access_token){status("Partite: elenco SUPERLEGA del 19 ottobre 2026. Accedi per configurare la sincronizzazione.",false);return}
 try{
 if(frSession.expires_at&&Date.now()/1000>frSession.expires_at-90)await frRefresh();
 const res=await fetch(FR_SB+"/functions/v1/francesca-partite",{headers:{"apikey":FR_KEY,"authorization":"Bearer "+frSession.access_token},cache:"no-store"});
 if(!res.ok)throw Error("HTTP "+res.status);
 const data=await res.json();
 if(data.live&&Array.isArray(data.matches)&&data.matches.length){
  GAMES.splice(0,GAMES.length,...data.matches);matchLive=true;
  status("Partite sincronizzate con Google Calendar SUPERLEGA · "+data.matches.length+" gare · aggiornate ora.",true);render();
 }else{
  matchLive=false;
  status("Partite: copia del 19 ottobre 2026. "+(data.reason||"Collega SUPERLEGA per l’aggiornamento automatico."),false);render();
 }
 }catch(e){status("Partite: copia del 19 ottobre 2026. Impossibile verificare le modifiche: "+e.message,false)}
};
const previousLoad=frLoad;
frLoad=async function(){
 const result=await previousLoad();
 await loadConfig();
 await frMatches();
 return result;
};
loadConfig().then(frMatches).catch(()=>{});
setInterval(()=>{if(frSession&&document.visibilityState==="visible")frMatches()},15*60*1000);
})();