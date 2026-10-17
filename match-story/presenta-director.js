/* PRESENTA REGIA — curated report stage, editable independently of source chapters. */
(function(){
'use strict';
const DB_API='https://rwfxsbxxykocdcqphfvb.supabase.co/functions/v1/match-story-director';
const layouts=['cover','message','split','photo','summary'];
const $=(root,selector)=>root.querySelector(selector);
const all=(root,selector)=>[...root.querySelectorAll(selector)];
const safe=s=>String(s??'').replace(/[\u0000-\u001f]/g,' ').replace(/\s+/g,' ').trim();
const separate=s=>String(s??'').replace(/\u00a0/g,' ').replace(/[ \t]+\n/g,'\n').replace(/\n[ \t]+/g,'\n').replace(/\n{3,}/g,'\n\n').trim();
const shrink=(s,n)=>{const v=separate(s);if(v.length<=n)return v;const cut=v.slice(0,n),last=Math.max(cut.lastIndexOf('. '),cut.lastIndexOf('; '),cut.lastIndexOf(', '));return cut.slice(0,last>n*.62?last+1:n).trim()};
const isHigh=ch=>String(ch.owner).toLowerCase()==='max'&&/^\s*high\s*balls?\s*$/i.test(ch.title||'');
const included=ch=>ch.owner==='Angelo'||isHigh(ch);
const validImage=src=>/^(https?:\/\/|data:image\/(?:jpeg|jpg|png|gif|webp);base64,)/i.test(String(src||''));
const validVideo=href=>/^https:\/\//i.test(String(href||''))&&/(youtube\.com|youtu\.be|vimeo\.com)/i.test(String(href||''));
const nodeText=node=>{
 if(!node)return'';
 const copy=node.cloneNode(true);
 all(copy,'script,style,svg,img,iframe,object,embed,button,.tabbed-card-nav,.native-image-controls').forEach(e=>e.remove());
 all(copy,'br').forEach(el=>el.replaceWith('\n'));
 all(copy,'p,li,h1,h2,h3,h4,blockquote,div.editor-highlight').forEach(el=>el.append('\n'));
 return separate(copy.textContent||'');
};
function chunks(text,max=300,limit=5){
 const bits=separate(text).split(/\n+/).map(s=>s.trim()).filter(s=>s&&s.toLowerCase()!=='scrivi qui…');
 const result=[];let pending='';
 for(const line of bits){
  if(line.length>max){
   if(pending){result.push(pending);pending=''}
   const clauses=line.match(/[^.!?;]+[.!?;]?\s*/g)||[line];
   for(const clause of clauses){
    if(pending.length+clause.length>max&&pending){result.push(pending);pending=''}
    if(clause.length>max){for(let i=0;i<clause.length;i+=max){result.push(clause.slice(i,i+max).trim())}}
    else pending+=(pending?' ':'')+clause.trim();
   }
  }else if(pending.length+line.length+1>max){result.push(pending);pending=line}
  else pending+=(pending?'\n':'')+line;
  if(result.length>=limit)break;
 }
 if(pending&&result.length<limit)result.push(pending);
 return result.filter(Boolean).slice(0,limit);
}
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const ownerTheme=o=>o==='Max'?'sd-max':'';
function source(ch){
 const holder=document.createElement('div');
 holder.innerHTML=String(ch.content||'');
 all(holder,'script,style,iframe,object,embed,form,input,textarea,svg').forEach(el=>el.remove());
 const assets=[];
 all(holder,'img').forEach((img,i)=>{
  const src=img.getAttribute('src')||'';
  if(!validImage(src))return;
  const card=img.closest('.editor-content-card,.tabbed-card-panel');
  const hint=card?nodeText(card).split('\n')[0]:ch.title;
  assets.push({id:ch.section_id+':'+i,src,alt:safe(img.getAttribute('alt'))||'Fotogramma '+(i+1),hint:shrink(hint,95)});
 });
 const cards=all(holder,'.editor-content-card').filter(el=>!el.parentElement?.closest('.editor-content-card'));
 const videos=all(holder,'a[href]').filter(a=>validVideo(a.href)).map(a=>({title:safe(a.textContent)||'Clip video',url:a.href})).slice(0,18);
 const intro=holder.cloneNode(true);
 all(intro,'.editor-content-card,.tabbed-card-nav,img,.tabbed-card-group').forEach(el=>el.remove());
 return{holder,cards,assets,videos,intro:nodeText(intro)};
}
function key(ch,suffix){return String(ch.section_id||ch.title)+'/'+suffix}
function make(ch,title,body,layout='message',imageId=null,suffix='0'){
 return{id:key(ch,suffix),chapterId:String(ch.section_id||ch.title),chapterTitle:String(ch.title||''),owner:ch.owner,title:shrink(title,170),body:shrink(body,1200),layout:layouts.includes(layout)?layout:'message',imageId};
}
function story(ch,sourceData){
 const out=[],assets=sourceData.assets,byCard=(card)=>assets.filter(a=>all(card,'img').some(img=>(img.getAttribute('src')||'')===a.src));
 const introParts=chunks(sourceData.intro,300,4);
 out.push(make(ch,(ch.title||'').toUpperCase(),introParts[0]||'', 'cover',null,'cover'));
 if(/^\s*SIDE OUT PERUGIA\s*$/i.test(ch.title||'')&&sourceData.intro.includes('FERAL')&&sourceData.intro.includes('ALTEZZA')){
  out.push(make(ch,'IL LORO BREAK','Battuta con pochi errori e numerose variazioni.\nCentrali e posti 2 efficaci a muro.','summary',null,'side-break'));
  out.push(make(ch,'GHIGLIOTTINA SU FERAL','TENERE VIVA LA PALLA.','message',null,'side-ghigliottina'));
  out.push(make(ch,'HIGH BALL CORTE','Lavorare sul terzo uomo di muro.\nContro il muro alto di M2 e M3: Tik-tok o mani out.','message',null,'side-high'));
  out.push(make(ch,'VARIAZIONI CORTE','Spin corte e float corte, soprattutto sui centrali.','message',null,'side-corte'));
  out.push(make(ch,'ALTEZZA MEZZO SIDE OUT','Accettiamo di essere esclusi dall’attacco per mettere l’alzatore e gli altri tre attaccanti nelle condizioni migliori.','summary',null,'side-altezza'));
  out.push(make(ch,'COMUNICARE','Libero: comunicazione verbale.\nPrima linea: gesti per aiutare la seconda linea sulle battute corte.','summary',null,'side-comunicare'));
  return out;
 }
 const maxIntro=ch.title==='INFO GENERALI'?2:ch.title==='LA BUSSOLA'?1:1;
 for(let i=1;i<Math.min(introParts.length,maxIntro+1);i++){
   let main=introParts[i].split(/[.!?]\s/)[0];
   if(main.length>77)main=ch.title;
   out.push(make(ch,main.toUpperCase(),introParts[i],'message',null,'intro-'+i));
 }
 sourceData.cards.forEach((card,i)=>{
   const text=nodeText(card);
   const parts=chunks(text,270,3);
   const firstLine=separate(text).split('\n')[0]||ch.title;
   const name=shrink(firstLine,67)||ch.title;
   const pics=byCard(card);
   if(pics.length){
     pics.forEach((img,j)=>out.push(make(ch,name.toUpperCase(),(parts[j]||parts[0]||'').slice(0,350),j===0?'split':'photo',img.id,'card-'+i+'-img-'+j)));
   }else{
     (parts.length?parts:['']).slice(0,2).forEach((paragraph,j)=>out.push(make(ch,name.toUpperCase(),paragraph,'message',null,'card-'+i+'-text-'+j)));
   }
 });
 const used=new Set(out.map(s=>s.imageId).filter(Boolean));
 for(const pic of assets.filter(x=>!used.has(x.id)).slice(0,12)){
   out.push(make(ch,pic.hint||ch.title,'', 'photo',pic.id,'extra-image-'+pic.id.split(':').pop()));
 }
 if(out.length===1&&introParts.length>1){
  introParts.slice(1,5).forEach((chunk,i)=>out.push(make(ch,shrink(chunk.split(/[.!?]\s/)[0],75)||ch.title,chunk,'message',null,'extra-'+i)));
 }
 return out.slice(0,isHigh(ch)?22:14);
}
function createDeck(chapters){const pages=[];for(const ch of chapters.filter(included))pages.push(...story(ch,source(ch)));return pages}
function signature(chapters){return chapters.map(ch=>[ch.section_id,ch.updated_at,ch.title,ch.position].join(':')).join('|').slice(0,4000)}
function preview(ch,report){
 const win=window.open('','_blank');
 if(!win){alert('Consenti i popup per aprire Anteprima.');return}
 const doc=win.document;
 doc.open();doc.write('<!doctype html><html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Anteprima capitolo</title><style>body{background:#edf1f6;color:#15283e;font:16px/1.5 system-ui;margin:0;padding:20px 14px}main{max-width:740px;margin:auto}header{background:#102940;color:white;padding:18px 22px;border-radius:12px}header h1{font-size:25px;margin:5px 0}header small{letter-spacing:.08em;color:#f4c2cf;font-weight:800}.sd-preview-content{background:#fffdf6;border:1px solid #cad4df;border-radius:12px;padding:23px;margin:16px auto;overflow-wrap:anywhere;line-height:1.6}.sd-preview-content img{display:block;max-width:100%;height:auto;margin:13px auto}.sd-preview-content .editor-content-card,.sd-preview-content .tabbed-card-panel{background:#f6f9ff;border:1px solid #c9d6e5;border-radius:10px;padding:15px;margin:13px 0}.sd-preview-content .tabbed-card-panel[hidden]{display:none}.sd-preview-content .tabbed-card-nav{display:flex;flex-wrap:wrap;gap:5px}.sd-preview-content .tabbed-card-nav button{background:#f1f5f9;border:1px solid #9faeba;border-radius:6px;padding:9px;cursor:pointer}.sd-preview-content .tabbed-card-nav button[aria-selected=true]{background:#173a52;color:#fff}.sd-preview-content .editor-highlight{border-left:5px solid #d83567;background:#f7f1f4;padding:9px 12px;margin:8px 0}.sd-preview-content a{color:#0c5aa8}.sd-preview-content *{max-width:100%}body>a{color:#fff}</style></head><body><main><header><small></small><h1></h1><p></p></header><article class="sd-preview-content" id="content"></article></main></body></html>');doc.close();
 $('header small',doc)?.replaceChildren(doc.createTextNode('ANTEPRIMA · '+ch.owner));
 $('header h1',doc)?.replaceChildren(doc.createTextNode(ch.title));
 $('header p',doc)?.replaceChildren(doc.createTextNode(report.title||''));
 const holder=doc.getElementById('content');
 holder.innerHTML=String(ch.content||'');
 all(holder,'script,iframe,object,embed,form,input,textarea,style').forEach(el=>el.remove());
 all(holder,'*').forEach(el=>{for(const at of [...el.attributes]){const k=at.name.toLowerCase(),v=at.value;if(k.startsWith('on')||k==='srcdoc'||(k==='href'&&!/^https?:\/\//i.test(v)&&!v.startsWith('#'))||(k==='src'&&!validImage(v)))el.removeAttribute(at.name)}});
 holder.addEventListener('click',ev=>{
  const b=ev.target.closest('.tabbed-card-nav button');
  if(!b)return;const group=b.closest('.tabbed-card-group'),items=all(group,'.tabbed-card-nav button'),at=items.indexOf(b);
  items.forEach((btn,i)=>btn.setAttribute('aria-selected',String(i===at)));
  all(group,'.tabbed-card-panel').forEach((p,i)=>p.hidden=i!==at);
 });
}
async function open({report,chapters,editMode=false}){
 if(document.querySelector('.sir-dir'))return;
 const chosen=chapters.filter(included);
 const excluded=chapters.filter(ch=>ch.owner==='Max'&&!isHigh(ch));
 if(!chosen.length){alert('Nessun capitolo di Angelo o HIGH BALLS di Max disponibile.');return}
 const records=new Map(chapters.map(ch=>[String(ch.section_id||ch.title),ch]));
 const images=new Map;chapters.filter(included).forEach(ch=>source(ch).assets.forEach(asset=>images.set(asset.id,asset)));
 const srcSig=signature(chapters);
 let pages=createDeck(chapters),at=0,beat=0,editing=false,dirty=false,timer=null,saveInFlight=null,queuedRevision=0,closing=false,stale=false;
 const backupKey='sir-match-story-director:'+report.id;
 let local=null;
 try{local=JSON.parse(localStorage.getItem(backupKey)||'null')}catch(e){}
 const shell=document.createElement('section');shell.className='sir-dir';shell.setAttribute('role','dialog');shell.setAttribute('aria-modal','true');shell.setAttribute('aria-label','PRESENTA · Regia tecnica');
 shell.innerHTML='<header class="sd-top"><strong class="sd-brand">SIR MATCH STORY <span>/ PRESENTA</span></strong><span class="sd-match"></span><div class="sd-tools"><button class="sd-tool" data-sd="index">INDICE</button><button class="sd-tool" data-sd="edit">MODIFICA</button><button class="sd-tool" data-sd="full">SCHERMO INTERO</button><button class="sd-tool" data-sd="close">CHIUDI</button></div></header><main class="sd-main"><div class="sd-stage"><div class="sd-surface" id="sdSurface"></div></div><aside class="sd-sidebar" id="sdEditor" hidden><div class="sd-side-top"><strong>REGIA · MODIFICA SLIDE</strong><p>Il report e le card originali rimangono invariati. Qui modifichi soltanto il racconto proiettato.</p></div><div class="sd-field"><label for="sdEditTitle">Titolo</label><textarea id="sdEditTitle" maxlength="170" rows="2"></textarea></div><div class="sd-field"><label for="sdEditBody">Messaggio / spiegazione</label><textarea id="sdEditBody" rows="5" maxlength="1200"></textarea><small>Se il testo è molto lungo, puoi dividerlo in più slide.</small></div><div class="sd-field"><label for="sdEditLayout">Tipo di schermata</label><select id="sdEditLayout"><option value="cover">Apertura capitolo</option><option value="message">Messaggio forte</option><option value="split">Immagine + testo</option><option value="photo">Immagine protagonista</option><option value="summary">Sintesi</option></select></div><div class="sd-field"><label for="sdEditPhoto">Immagine dal capitolo originale</label><select id="sdEditPhoto"></select></div><div class="sd-assets-preview" id="sdImagePreview">Nessuna immagine selezionata</div><div class="sd-editor-actions"><button class="sd-editbtn" data-sd="up">↑ Sposta</button><button class="sd-editbtn" data-sd="down">↓ Sposta</button><button class="sd-editbtn" data-sd="add">+ Slide</button><button class="sd-editbtn sd-danger" data-sd="delete">Elimina slide</button></div><div class="sd-side-divider"></div><div class="sd-editor-actions"><button class="sd-editbtn sd-save" data-sd="save">Salva adesso</button><button class="sd-editbtn" data-sd="reset">Rigenera dalle card</button></div><div class="sd-save-state" id="sdSaveState">Le modifiche si salvano automaticamente.</div></aside><div class="sd-nav-shade" id="sdNavigator" hidden><div class="sd-nav-panel"><header class="sd-nav-h"><div><small>INDICE DELLA PRESENTAZIONE</small><strong>I capitoli del racconto</strong></div><button class="sd-tool" data-sd="index-close">CHIUDI</button></header><div class="sd-nav-section">PRESENTA · CAPITOLI DI ANGELO E HIGH BALLS</div><div class="sd-nav-grid" id="sdChapters"></div><div class="sd-nav-other"><div class="sd-nav-section">MAX · SOLO ANTEPRIMA ORIGINALE</div><p class="sd-nav-msg">Questi capitoli non vengono trasformati in slide. Si aprono esattamente come materiale di consultazione.</p><div class="sd-nav-grid" id="sdExtras"></div></div></div></div><div class="sd-toast" id="sdToast" hidden></div></main><footer class="sd-footer-bar"><button class="sd-arrow" data-sd="previous" aria-label="Indietro">←</button><button class="sd-arrow sd-forward" data-sd="next" aria-label="Avanti">→</button><span class="sd-count" id="sdCount">01 / 01</span><div class="sd-progress" id="sdProgress"></div><span class="sd-btm-label" id="sdDetailHint">→ mostrerà il dettaglio successivo</span></footer>';
 document.body.append(shell);
 const previousFocus=document.activeElement,oldOverflow=document.body.style.overflow;
 document.body.style.overflow='hidden';
 const stage=$(shell,'.sd-stage'),surface=$(shell,'#sdSurface'),editor=$(shell,'#sdEditor');
 const navigator=$(shell,'#sdNavigator'),progress=$(shell,'#sdProgress'),chaptersEl=$(shell,'#sdChapters'),extraEl=$(shell,'#sdExtras');
 const titleInput=$(shell,'#sdEditTitle'),bodyInput=$(shell,'#sdEditBody'),layoutInput=$(shell,'#sdEditLayout'),photoInput=$(shell,'#sdEditPhoto');
 const previewEl=$(shell,'#sdImagePreview'),status=$(shell,'#sdSaveState'),toaster=$(shell,'#sdToast');
 $(shell,'.sd-match').textContent=report.title||'MATCH REPORT';
 let ro=null;
 const showToast=msg=>{toaster.textContent=msg;toaster.hidden=false;clearTimeout(showToast.handle);showToast.handle=setTimeout(()=>toaster.hidden=true,2600)};
 const imageRef=page=>images.get(page.imageId||'');
 function resize(){if(!shell.isConnected)return;const r=stage.getBoundingClientRect(),scale=Math.min((r.width-22)/1280,(r.height-22)/720);surface.style.transform='scale('+Math.max(.07,scale)+')'}
 ro=new ResizeObserver(resize);ro.observe(stage);
 function note(){status.textContent=dirty?'Modifiche in attesa di salvataggio…':stale?'Materiale originale cambiato: per aggiornare la sequenza usa «Rigenera dalle card» (le modifiche al racconto verranno sostituite).':'✓ Regia salvata. Le card del report restano intatte.'}
 function backup(){try{localStorage.setItem(backupKey,JSON.stringify({pages,signature:srcSig,savedAt:Date.now()}))}catch(e){status.textContent='Spazio locale pieno: salva online prima di uscire.'}}
 function mark(){dirty=true;queuedRevision++;backup();status.textContent='Modifiche in attesa di salvataggio automatico…';clearTimeout(timer);timer=setTimeout(()=>save(true).catch(()=>{}),1200)}
 async function save(automatic=false){
   clearTimeout(timer);
   if(saveInFlight){const okay=await saveInFlight;if(dirty)return save(automatic);return okay}
   if(!dirty)return true;
   const rev=queuedRevision,data=pages.map(p=>({...p}));
   status.textContent='Salvataggio online…';
   const operation=(async()=>{
     try{
       const response=await fetch(DB_API+'?action=save&id='+encodeURIComponent(report.id),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pages:data,source_signature:srcSig}),cache:'no-store'});
       const result=await response.json();
       if(!response.ok)throw Error(result.error||'Errore di salvataggio');
       if(rev===queuedRevision){dirty=false;note();try{localStorage.removeItem(backupKey)}catch(e){}}
       else{dirty=true;backup();clearTimeout(timer);timer=setTimeout(()=>save(true),900)}
       return true;
     }catch(e){backup();status.textContent='⚠ Non salvato online. Copia temporanea nel browser: '+e.message;if(!automatic)alert(status.textContent);return false}
     finally{saveInFlight=null}
   })();saveInFlight=operation;return operation;
 }
 function chooseImage(){
   const p=pages[at],asset=imageRef(p);
   previewEl.replaceChildren();
   if(asset){const img=document.createElement('img');img.src=asset.src;img.alt=asset.alt;previewEl.append(img)}
   else previewEl.textContent='Nessuna immagine selezionata';
 }
 function editForm(){
   if(!editing)return;
   const p=pages[at];if(!p)return;
   titleInput.value=p.title;bodyInput.value=p.body;layoutInput.value=p.layout;
   photoInput.replaceChildren();
   const empty=new Option('Senza immagine','');photoInput.append(empty);
   const origin=records.get(p.chapterId);if(origin){
     source(origin).assets.forEach((asset,i)=>{photoInput.add(new Option((i+1)+'. '+shrink(asset.alt||asset.hint,60),asset.id))});
   }
   photoInput.value=p.imageId||'';
   if(photoInput.selectedIndex<0)photoInput.value='';
   chooseImage();
 }
 function dynamicFont(el,preferred,min,allowedHeight){
   let v=preferred;
   el.style.fontSize=v+'px';
   while(v>min&&(el.scrollHeight>allowedHeight||el.scrollWidth>el.clientWidth+4)){v-=3;el.style.fontSize=v+'px'}
 }
 function show(){
   if(!pages.length)return;
   at=Math.min(Math.max(0,at),pages.length-1);
   const p=pages[at],asset=imageRef(p),hasDetail=!!(p.body?.trim()||asset),revealed=beat>0||!hasDetail;
   surface.replaceChildren();
   const scene=document.createElement('article');scene.className='sd-slide '+(p.owner==='Max'?'sd-max':'')+' sd-'+p.layout+(p.layout==='split'||p.layout==='summary'?' sd-light':'');
   const header=document.createElement('div');header.className='sd-eyebrow';
   const tag=document.createElement('span');tag.textContent='SIR MATCH STORY  /  '+(p.chapterTitle||'');
   const by=document.createElement('span');by.className='sd-owner';by.textContent=p.owner==='Max'?'MAX':'ANGELO';
   header.append(tag,by);scene.append(header);
   const main=document.createElement('div');main.className='sd-body';
   const head=document.createElement('h2');head.className='sd-h';head.textContent=p.title||p.chapterTitle||'';
   const desc=document.createElement('div');desc.className='sd-content';desc.textContent=p.body||'';desc.hidden=!revealed||!p.body;
   let media=null;
   if(asset&&p.layout!=='message'&&p.layout!=='summary'&&p.layout!=='cover'){
     media=document.createElement('div');media.className='sd-media';media.hidden=!revealed;
     const img=document.createElement('img');img.src=asset.src;img.alt=asset.alt;img.loading='eager';media.append(img);
   }
   if(p.layout==='split'||p.layout==='photo'){
     const txt=document.createElement('div');txt.className='sd-textcol';const bar=document.createElement('span');bar.className='sd-underline';
     txt.append(head,bar,desc);
     if(p.layout==='photo'){if(media)main.append(media);main.append(txt)}
     else{main.append(txt);if(media)main.append(media)}
   }else{const bar=document.createElement('span');bar.className='sd-underline';main.append(head,bar,desc)}
   scene.append(main);
   const foot=document.createElement('div');foot.className='sd-foot';const left=document.createElement('small');left.textContent='LA LETTURA · IL MESSAGGIO · IL GIOCO';const right=document.createElement('small');right.textContent=String(at+1).padStart(2,'0')+' / '+String(pages.length).padStart(2,'0');foot.append(left,right);scene.append(foot);
   surface.append(scene);
   // Fixed 1280x720 projection geometry: text is reduced only when it cannot fit its assigned column.
   requestAnimationFrame(()=>{
     if(!scene.isConnected)return;
     const maxTitle=p.layout==='cover'?130:p.layout==='message'?107:p.layout==='summary'?84:p.layout==='split'?68:64;
     const titleSpace=(p.layout==='photo'||p.layout==='split')?360:480;
     dynamicFont(head,maxTitle,34,titleSpace);
     if(!desc.hidden){const bodyMax=p.layout==='cover'?32:p.layout==='message'?37:p.layout==='summary'?40:p.layout==='split'?31:28;dynamicFont(desc,bodyMax,22,p.layout==='photo'?320:p.layout==='split'?360:355)}
   });
   $(shell,'#sdCount').textContent=String(at+1).padStart(2,'0')+' / '+String(pages.length).padStart(2,'0');
   $(shell,'#sdDetailHint').textContent=hasDetail&&!revealed?'→ mostra spiegazione / immagine':'→ prossima schermata';
   $(shell,'[data-sd="previous"]').disabled=at===0&&beat===0;
   $(shell,'[data-sd="next"]').disabled=at===pages.length-1&&(beat>0||!hasDetail);
   progress.replaceChildren();
   pages.forEach((page,i)=>{
     const b=document.createElement('button');b.type='button';b.className='sd-tick'+(i===at?' is-active':i<at?' is-past':'');
     b.title=String(i+1)+' · '+page.title;b.setAttribute('aria-label','Vai alla slide '+(i+1));
     b.onclick=()=>{at=i;beat=0;show()};progress.append(b)
   });
   editForm();resize();
 }
 function next(){const p=pages[at];if(!p)return;if(beat===0&&(p.body?.trim()||imageRef(p))){beat=1;show();return}if(at<pages.length-1){at++;beat=0;show()}}
 function prev(){if(beat===1){beat=0;show();return}if(at>0){at--;beat=1;show()}}
 function updateScene(prop,val){if(!pages[at])return;pages[at][prop]=val;if(prop==='imageId'&&val&&['cover','message','summary'].includes(pages[at].layout))pages[at].layout='split';mark();show()}
 function rebuildIndex(){
   chaptersEl.replaceChildren();extraEl.replaceChildren();
   for(const ch of chosen){
     const first=pages.findIndex(p=>p.chapterId===String(ch.section_id||ch.title));
     if(first<0)continue;
     const b=document.createElement('button');b.className='sd-chapter';b.type='button';
     const n=document.createElement('small');n.textContent=String(first+1).padStart(2,'0');
     const label=document.createElement('span');label.textContent=ch.title+' · '+ch.owner;
     b.append(n,label);b.onclick=()=>{navigator.hidden=true;at=first;beat=0;show()};chaptersEl.append(b)
   }
   if(!excluded.length){const p=document.createElement('p');p.className='sd-nav-msg';p.textContent='Nessun altro capitolo Max.';extraEl.append(p)}
   excluded.forEach(ch=>{const b=document.createElement('button');b.type='button';b.className='sd-preview-link';b.textContent='↗ '+ch.title;b.onclick=()=>preview(ch,report);extraEl.append(b)})
 }
 const delegated=async ev=>{
   const btn=ev.target.closest('[data-sd]');if(!btn)return;
   const action=btn.dataset.sd;
   if(action==='next')next();
   if(action==='previous')prev();
   if(action==='index'){rebuildIndex();navigator.hidden=false}
   if(action==='index-close')navigator.hidden=true;
   if(action==='full'){
     try{if(document.fullscreenElement===shell)await document.exitFullscreen();else await shell.requestFullscreen()}catch(e){showToast('Schermo intero non disponibile')}
   }
   if(action==='edit'){editing=!editing;editor.hidden=!editing;btn.textContent=editing?'FINE MODIFICA':'MODIFICA';editForm();resize()}
   if(action==='up'&&at>0){pages.splice(at-1,0,pages.splice(at,1)[0]);at--;mark();show()}
   if(action==='down'&&at<pages.length-1){pages.splice(at+1,0,pages.splice(at,1)[0]);at++;mark();show()}
   if(action==='add'){
     const origin=pages[at];
     const newPage={...origin,id:origin.chapterId+'/custom-'+crypto.randomUUID(),title:'NUOVA SLIDE',body:'',imageId:null,layout:'message'};
     pages.splice(at+1,0,newPage);at++;beat=0;mark();show();
   }
   if(action==='delete'&&pages.length>1&&confirm('Eliminare questa slide dalla sola presentazione? Il report originale non verrà toccato.')){
     pages.splice(at,1);at=Math.min(at,pages.length-1);beat=0;mark();show()
   }
   if(action==='save')await save(false);
   if(action==='reset'&&confirm('Rigenerare la regia dalle card originali? Le modifiche alle slide saranno sostituite. I capitoli originali restano invariati.')){
     pages=createDeck(chapters);at=0;beat=0;stale=false;mark();show();showToast('Regia rigenerata dalle card')
   }
   if(action==='close')await close();
 };
 shell.addEventListener('click',delegated);
 titleInput.addEventListener('input',()=>{pages[at].title=titleInput.value;mark();show()});
 bodyInput.addEventListener('input',()=>{pages[at].body=bodyInput.value;mark();show()});
 layoutInput.addEventListener('change',()=>updateScene('layout',layoutInput.value));
 photoInput.addEventListener('change',()=>updateScene('imageId',photoInput.value||null));
 const keydown=e=>{
   if(!shell.isConnected)return;
   if(e.target.matches('input,textarea,select,[contenteditable=true]'))return;
   if(e.key==='Escape'){
     if(!navigator.hidden){navigator.hidden=true;e.preventDefault();return}
     if(editing){editing=false;editor.hidden=true;$(shell,'[data-sd="edit"]').textContent='MODIFICA';resize();e.preventDefault();return}
     if(document.fullscreenElement===shell){document.exitFullscreen().catch(()=>{});return}
     close();return
   }
   if(!navigator.hidden)return;
   if(e.ctrlKey||e.metaKey||e.altKey)return;
   if(e.key==='ArrowRight'||e.key===' '||e.key==='PageDown'){e.preventDefault();next()}
   else if(e.key==='ArrowLeft'||e.key==='PageUp'){e.preventDefault();prev()}
   else if(e.key.toLowerCase()==='i'){e.preventDefault();rebuildIndex();navigator.hidden=false}
   else if(e.key.toLowerCase()==='e'){e.preventDefault();$(shell,'[data-sd="edit"]').click()}
   else if(e.key.toLowerCase()==='f'){e.preventDefault();$(shell,'[data-sd="full"]').click()}
 };
 async function close(){
   if(closing)return;closing=true;
   if(dirty){const okay=await save(true);if(!okay&&!confirm('Non riesco a salvare online. La copia locale resta in questo browser. Vuoi chiudere?')){closing=false;return}}
   if(ro)ro.disconnect();
   document.removeEventListener('keydown',keydown,true);
   if(document.fullscreenElement===shell)await document.exitFullscreen().catch(()=>{});
   shell.remove();document.body.style.overflow=oldOverflow;
   previousFocus?.focus();
 }
 document.addEventListener('keydown',keydown,true);
 try{
   const resp=await fetch(DB_API+'?action=load&id='+encodeURIComponent(report.id),{cache:'no-store'});
   if(!resp.ok)throw Error('lettura regia non disponibile');
   const result=await resp.json();
   if(Array.isArray(result.pages)&&result.pages.length&&result.pages.every(p=>p&&layouts.includes(p.layout)&&['Angelo','Max'].includes(p.owner)&&typeof p.title==='string'&&typeof p.body==='string')){
     pages=result.pages.filter(p=>chosen.some(c=>String(c.section_id||c.title)===p.chapterId));
     if(!pages.length)pages=createDeck(chapters);
     stale=!!result.source_signature&&result.source_signature!==srcSig;
   }
   if(local?.pages?.length&&Array.isArray(local.pages)&&Number(local.savedAt)>Date.parse(result.updated_at||0)){
     pages=local.pages.filter(p=>chosen.some(c=>String(c.section_id||c.title)===p.chapterId));
     if(!pages.length)pages=createDeck(chapters);
     dirty=true;queuedRevision++;backup();status.textContent='Bozza locale recuperata. Salvo online…';clearTimeout(timer);timer=setTimeout(()=>save(true),1300)
   }else note();
 }catch(e){
   if(local?.pages?.length){pages=local.pages.filter(p=>chosen.some(c=>String(c.section_id||c.title)===p.chapterId));dirty=true;queuedRevision++;status.textContent='Offline: recuperata la bozza locale.'}
   else status.textContent='Modalità locale: il collegamento al salvataggio online non è disponibile.'
 }
 if(!pages.length)pages=createDeck(chapters);
 at=0;beat=0;editing=!!editMode;editor.hidden=!editing;$(shell,'[data-sd="edit"]').textContent=editing?'FINE MODIFICA':'MODIFICA';
 rebuildIndex();show();resize();
}
window.SirPresentaDirector={open};
})();