(function noticeModule(){
'use strict';
const URL_NOTICE='https://rwfxsbxxykocdcqphfvb.supabase.co/functions/v1/match-story-opening-notice';
const isPlayers=new URLSearchParams(location.search).get('view')==='players';
const imagePattern=/^https:\/\/rwfxsbxxykocdcqphfvb\.supabase\.co\/storage\/v1\/object\/public\/match-story-presenta-images\/[a-z0-9\/_.-]+$/i;
async function request(action,data,raw=false){
 const headers={};
 if(data!==undefined)headers['Content-Type']=raw?'image/jpeg':'application/json';
 const res=await fetch(URL_NOTICE+'?action='+action,{method:data===undefined?'GET':'POST',headers,body:data===undefined?undefined:raw?data:JSON.stringify(data),cache:'no-store'});
 const out=await res.json();if(!res.ok)throw Error(out.error||'Collegamento non disponibile');return out;
}
function popup(data,force=false){
 if(!force&&!data?.enabled)return;
 if(!force&&!String(data?.message||'').trim())return;
 document.getElementById('msn-overlay')?.remove();
 const o=document.createElement('div');o.id='msn-overlay';o.className='msn-overlay';
 o.innerHTML='<section class="msn-dialog" role="dialog" aria-modal="true" aria-labelledby="msnPopTitle"><div class="msn-banner"><small>SIR MATCH STORY · COMUNICAZIONE</small><button class="msn-x" type="button" aria-label="Chiudi avviso">×</button></div><div class="msn-picture" hidden><img alt="Immagine della comunicazione"></div><div class="msn-pop-content"><h2 id="msnPopTitle"></h2><p class="msn-message"></p><button type="button" class="msn-dismiss">HO LETTO · ENTRA</button></div></section>';
 o.querySelector('h2').textContent=String(data.title||'').trim()||'COMUNICAZIONE';
 o.querySelector('.msn-message').textContent=String(data.message||'').trim()||'Anteprima della comunicazione';
 if(imagePattern.test(String(data.imageUrl||''))){o.querySelector('.msn-picture').hidden=false;o.querySelector('.msn-picture img').src=data.imageUrl}
 const previous=document.activeElement;
 function onKey(e){if(e.key==='Escape'){e.preventDefault();close()}}
 function close(){document.removeEventListener('keydown',onKey);o.remove();previous?.focus?.()}
 o.querySelector('.msn-x').onclick=close;o.querySelector('.msn-dismiss').onclick=close;
 o.addEventListener('click',e=>{if(e.target===o)close()});
 document.addEventListener('keydown',onKey);document.body.append(o);o.querySelector('.msn-x').focus({preventScroll:true});
}
let lastNoticeCheck=0,hiddenAt=0;
async function loadNotice(){
 if(document.querySelector('.msn-editor-overlay:not([hidden])')||document.getElementById('msn-overlay'))return;
 lastNoticeCheck=Date.now();try{const notice=await request('public');popup(notice)}catch(e){console.warn('Avviso non disponibile',e)}}
function editor(){
 if(isPlayers)return;
 const target=document.querySelector('#staffRoom .archive-top .line')||document.querySelector('#staffRoom .archive-top');if(!target)return;
 const trigger=document.createElement('button');trigger.type='button';trigger.className='staff-btn msn-manage-btn';trigger.textContent='✉ COMUNICAZIONE DI APERTURA';target.append(trigger);
 const modal=document.createElement('div');modal.className='msn-editor-overlay';modal.hidden=true;
 modal.innerHTML=`
 <section class="msn-editor" role="dialog" aria-modal="true" aria-labelledby="msnEditorTitle">
  <div class="msn-editor-head"><div><small>GESTIONE STAFF</small><h2 id="msnEditorTitle">COMUNICAZIONE DI APERTURA</h2><p>Personalizza il messaggio mostrato all’avvio dell’applicazione.</p></div><button type="button" class="msn-editor-close" aria-label="Chiudi">×</button></div>
  <div class="msn-fields">
   <label class="msn-checkbox"><input type="checkbox" id="msnEnabled"><span><b>ATTIVA IL POPUP ALL’APERTURA</b><small>Quando non è selezionato, nessun messaggio comparirà.</small></span></label>
   <label for="msnTitle">Titolo (facoltativo)</label><input type="text" id="msnTitle" maxlength="110" placeholder="Es. Una cosa importante">
   <label for="msnMessage">Testo della comunicazione</label><textarea id="msnMessage" maxlength="3600" rows="7" placeholder="Scrivi qui la tua comunicazione…"></textarea>
   <label>Immagine facoltativa · puoi usare anche solo il testo</label><div class="msn-upload-row"><label for="msnFile" class="msn-upload-picker"><span>+ SCEGLI IMMAGINE DAL DISPOSITIVO</span><input type="file" id="msnFile" accept="image/*,.heic,.heif" aria-label="Scegli immagine o foto dall’iPad"></label><button type="button" id="msnRemove">RIMUOVI IMMAGINE</button></div><div class="msn-preview-image" id="msnImageBox">Nessuna immagine selezionata</div>
   <p id="msnFeedback" class="msn-feedback" aria-live="polite"></p>
   <div class="msn-actions"><button type="button" id="msnTest">ANTEPRIMA</button><button type="button" id="msnSave">SALVA COMUNICAZIONE</button></div>
  </div>
 </section>`;
 document.body.append(modal);
 const q=s=>modal.querySelector(s);
 let picture=null,active=false;
 const feedback=(s,bad=false)=>{q('#msnFeedback').textContent=s;q('#msnFeedback').classList.toggle('msn-error',bad)};
 function imagePreview(){
  const box=q('#msnImageBox');box.replaceChildren();
  if(!picture){box.textContent='Nessuna immagine selezionata';return}
  const img=document.createElement('img');img.src=picture;img.alt='Immagine scelta';box.append(img);
 }
 function values(){return{enabled:q('#msnEnabled').checked,title:q('#msnTitle').value.trim(),message:q('#msnMessage').value.trim(),imageUrl:picture}}
 async function reload(){
  try{
   const p=await request('edit');
   q('#msnEnabled').checked=!!p.enabled;
   q('#msnTitle').value=p.title||'';
   q('#msnMessage').value=p.message||'';
   picture=p.imageUrl||null;imagePreview();
   feedback('Impostazioni caricate. Il popup è '+(p.enabled?'ATTIVO.':'DISATTIVATO.'));
  }catch(e){feedback('Impossibile caricare le impostazioni: '+e.message,true)}
 }
 trigger.onclick=async()=>{if(active)return;active=true;modal.hidden=false;feedback('Caricamento delle impostazioni…');await reload();q('.msn-editor-close').focus()};
 function close(){modal.hidden=true;active=false}
 q('.msn-editor-close').onclick=close;
 modal.addEventListener('click',e=>{if(e.target===modal)close()});
 q('#msnTest').onclick=()=>popup({...values(),enabled:true},true);
 q('#msnSave').onclick=async()=>{
  if(q('#msnEnabled').checked&&!q('#msnMessage').value.trim())return feedback('Inserisci il testo prima di attivare il popup.',true);
  const btn=q('#msnSave');btn.disabled=true;feedback('Salvataggio e verifica in corso…');
  try{
   const form=values();await request('save',form);
   const saved=await request('edit');
   if(!!saved.enabled!==form.enabled||saved.message!==form.message||saved.title!==form.title||saved.imageUrl!==form.imageUrl)
    throw Error('Il server non ha confermato i dati salvati. Riprova.');
   const publicState=await request('public');
   if(!!publicState.enabled!==form.enabled)
    throw Error('La comunicazione non risulta ancora nello stato selezionato. Riprova.');
   feedback(form.enabled?'✓ Comunicazione salvata e ATTIVA. La vedrai a ogni apertura, finché non disattivi il flag.':'✓ Comunicazione salvata e DISATTIVATA. Nessun popup all’apertura.');
   if(form.enabled)popup({...saved,enabled:true},true);
  }catch(e){feedback('Salvataggio non riuscito: '+e.message,true)}
  finally{btn.disabled=false}
 };
 q('#msnRemove').onclick=()=>{picture=null;imagePreview();feedback('Immagine rimossa dalla bozza. Premi SALVA per confermare.')};
 q('#msnFile').onchange=async()=>{
  const file=q('#msnFile').files?.[0];q('#msnFile').value='';if(!file)return;
  const btn=q('.msn-upload-picker');btn.classList.add('msn-busy');feedback('Preparo la foto…');
  try{
   if(!file.type.startsWith('image/')&&!/\.(heic|heif)$/i.test(file.name)||file.size>26*1024*1024)throw Error('Scegli una foto fino a 26 MB.');
   const url=URL.createObjectURL(file);
   let image;
   try{image=await new Promise((resolve,reject)=>{const i=new Image();i.onload=()=>resolve(i);i.onerror=()=>reject(Error('Formato non leggibile: usa JPG o PNG.'));i.src=url})}
   catch(e){URL.revokeObjectURL(url);throw e}
   const k=Math.min(1,1700/Math.max(image.naturalWidth,image.naturalHeight));
   const canvas=document.createElement('canvas');canvas.width=Math.max(1,Math.round(image.naturalWidth*k));canvas.height=Math.max(1,Math.round(image.naturalHeight*k));
   const ctx=canvas.getContext('2d');if(!ctx){URL.revokeObjectURL(url);throw Error('Impossibile preparare la foto sul dispositivo.')}
   ctx.fillStyle='#fff';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,0,0,canvas.width,canvas.height);URL.revokeObjectURL(url);
   const jpeg=quality=>new Promise((resolve,reject)=>canvas.toBlob(x=>x?resolve(x):reject(Error('Conversione non riuscita.')),'image/jpeg',quality));
   let blob=await jpeg(.82);if(blob.size>4_800_000)blob=await jpeg(.62);
   if(blob.size>4_900_000)throw Error('Foto troppo grande. Riprova con una più piccola.');
   feedback('Caricamento immagine…');
   const result=await request('upload',blob,true);
   picture=result.imageUrl;imagePreview();feedback('✓ Immagine caricata. Premi SALVA per confermare.');
  }catch(e){feedback('Impossibile inserire l’immagine: '+e.message,true)}finally{btn.classList.remove('msn-busy')}
 };
}
function init(){editor();loadNotice();
 window.addEventListener('pageshow',e=>{if(e.persisted)loadNotice()});
 document.addEventListener('visibilitychange',()=>{
  if(document.hidden){hiddenAt=Date.now();return}
  if(hiddenAt&&Date.now()-hiddenAt>2500&&Date.now()-lastNoticeCheck>2500)loadNotice()
 });
}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init,{once:true});else init();
})();