/* PRESENTA REGIA — curated report stage, editable independently of source chapters. */
(function(){
'use strict';
const DB_API='https://rwfxsbxxykocdcqphfvb.supabase.co/functions/v1/match-story-director';
const layouts=['cover','message','split','photo','summary'];
const rich=window.SirPresentaRichText;
const backgrounds=window.SirPresentaBackgrounds;
if(!rich)throw Error('Editor testo non caricato: aggiorna la pagina.');

const $=(root,selector)=>root.querySelector(selector);
const all=(root,selector)=>[...root.querySelectorAll(selector)];
const safe=s=>String(s??'').replace(/[\u0000-\u001f]/g,' ').replace(/\s+/g,' ').trim();
const separate=s=>String(s??'').replace(/\u00a0/g,' ').replace(/[ \t]+\n/g,'\n').replace(/\n[ \t]+/g,'\n').replace(/\n{3,}/g,'\n\n').trim();
const shrink=(s,n)=>{const v=separate(s);if(v.length<=n)return v;const cut=v.slice(0,n),last=Math.max(cut.lastIndexOf('. '),cut.lastIndexOf('; '),cut.lastIndexOf(', '));return cut.slice(0,last>n*.62?last+1:n).trim()};
const isHigh=ch=>String(ch.owner).toLowerCase()==='max'&&/^\s*high\s*balls?\s*$/i.test(ch.title||'');
const included=ch=>(ch.owner==='Angelo'&&!/^\s*LA\s+BUSSOLA\s*$/i.test(ch.title||''))||isHigh(ch);
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
 if(introParts[0])out.push(make(ch,(ch.title||'').toUpperCase(),introParts[0], 'summary',null,'intro-0'));
 if(/^\s*SIDE OUT PERUGIA\s*$/i.test(ch.title||'')&&sourceData.intro.includes('FERAL')&&sourceData.intro.includes('ALTEZZA')){
  out.push(make(ch,'IL LORO BREAK','Battuta con pochi errori e numerose variazioni.\nCentrali e posti 2 efficaci a muro.','summary',null,'side-break'));
  out.push(make(ch,'GHIGLIOTTINA SU FERAL','TENERE VIVA LA PALLA.\nLa nostra priorità immediata, fin dal primo scambio.','summary',null,'side-ghigliottina'));
  out.push(make(ch,'HIGH BALL CORTE','Lavorare sul terzo uomo di muro, il meno imponente dei tre.\nContro il muro alto di M2 e M3: Tik-tok o mani out.','summary',null,'side-high'));
  out.push(make(ch,'VARIAZIONI CORTE','Spin corte e float corte, soprattutto sui centrali.\nTre principi: altezza mezzo side out, disponibilità a essere esclusi dall’attacco e comunicazione.','summary',null,'side-corte'));
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
 if(sourceData.videos.length){
  sourceData.videos.forEach((video,i)=>{
    const scene=make(ch,video.title.length>3?video.title.toUpperCase():'VIDEO · '+(i+1),'Guarda il video e poi torniamo alla nostra lettura del gioco.','message',null,'video-'+i);
    scene.videoUrl=video.url;out.push(scene);
  });
 }
 if(out.length===1&&introParts.length>1){
  introParts.slice(1,5).forEach((chunk,i)=>out.push(make(ch,shrink(chunk.split(/[.!?]\s/)[0],75)||ch.title,chunk,'message',null,'extra-'+i)));
 }
 return out.slice(0,isHigh(ch)?22:14);
}
function createDeck(chapters){const pages=[];for(const ch of chapters.filter(included))pages.push(...story(ch,source(ch)));return pages}
function signature(chapters){return chapters.map(ch=>[ch.section_id,ch.updated_at,ch.title,ch.position].join(':')).join('|').slice(0,4000)}
function preview(ch,report){
 const chapters=Array.isArray(ch)?ch:[ch];
 // Opens the complete original chapter contents from the already loaded report,
 // including unpublished chapters. No dependence on PLAYERS publish state.
 const win=window.open('about:blank','_blank');
 if(!win){alert('Il browser ha bloccato la finestra. Consenti i popup per Match Story.');return}
 const doc=win.document;doc.open();doc.write('<!doctype html><html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Anteprima Match Story</title><style>body{background:#edf1f6;color:#15283e;font:16px/1.5 system-ui;margin:0;padding:18px 14px}main{max-width:840px;margin:auto}header{background:#102940;color:white;padding:17px 22px;border-radius:12px;margin-bottom:15px}header h1{font-size:24px;margin:5px 0}header small{letter-spacing:.08em;color:#f4c2cf;font-weight:800}nav{display:flex;flex-wrap:wrap;gap:7px;margin:12px 0 18px}nav a{background:#17344f;color:#fff;text-decoration:none;border-radius:6px;padding:8px 10px;font-size:12px;font-weight:700}section.chapter{background:#fffdf6;border:1px solid #cad4df;border-radius:12px;padding:22px;margin:16px auto;overflow-wrap:anywhere}section.chapter>h2{margin:0 0 14px;border-bottom:2px solid #df4079;padding-bottom:12px;font-size:23px}.sd-preview-content img{display:block;max-width:100%;height:auto;margin:13px auto}.sd-preview-content .editor-content-card,.sd-preview-content .tabbed-card-panel{background:#f6f9ff;border:1px solid #c9d6e5;border-radius:10px;padding:15px;margin:13px 0}.sd-preview-content .tabbed-card-panel[hidden]{display:none}.sd-preview-content .tabbed-card-nav{display:flex;flex-wrap:wrap;gap:5px}.sd-preview-content .tabbed-card-nav button{background:#f1f5f9;border:1px solid #9faeba;border-radius:6px;padding:9px;cursor:pointer}.sd-preview-content .tabbed-card-nav button[aria-selected=true]{background:#173a52;color:#fff}.sd-preview-content .editor-highlight{border-left:5px solid #d83567;background:#f7f1f4;padding:9px 12px;margin:8px 0}.sd-preview-content a{color:#0c5aa8}.sd-preview-content *{max-width:100%}.sd-owner-label{font-size:11px;color:#8a4261;font-weight:900;letter-spacing:.12em;text-transform:uppercase}a.return-top{display:block;margin-top:18px;color:#105481;font-size:13px}</style></head><body><main><header><small>ANTEPRIMA · MATERIALE ORIGINALE</small><h1 id="storyTitle"></h1><p id="reportTitle"></p></header><nav id="chapterNav" aria-label="Capitoli del rapportino"></nav><div id="chapterContents"></div></main></body></html>');doc.close();
 doc.getElementById('storyTitle').textContent=chapters.length===1?chapters[0].title:'RAPPORTINO COMPLETO';
 doc.getElementById('reportTitle').textContent=report.title||'';
 const content=doc.getElementById('chapterContents'),nav=doc.getElementById('chapterNav');
 chapters.forEach((item,i)=>{
  const section=doc.createElement('section');section.className='chapter';section.id='capitolo-'+(i+1);
  const owner=doc.createElement('div');owner.className='sd-owner-label';owner.textContent=item.owner||'';
  const title=doc.createElement('h2');title.textContent=item.title;section.append(owner,title);
  const holder=doc.createElement('div');holder.className='sd-preview-content';holder.innerHTML=String(item.content||'');
  all(holder,'script,iframe,object,embed,form,input,textarea,style').forEach(el=>el.remove());
  all(holder,'*').forEach(el=>{for(const at of [...el.attributes]){const k=at.name.toLowerCase(),v=at.value;if(k.startsWith('on')||k==='srcdoc'||(k==='href'&&!/^https?:\/\//i.test(v)&&!v.startsWith('#'))||(k==='src'&&!validImage(v)))el.removeAttribute(at.name)}});
  section.append(holder);
  if(chapters.length>1){
   const jump=doc.createElement('a');jump.className='return-top';jump.href='#chapterNav';jump.textContent='↑ Torna all’indice';section.append(jump);
   const link=doc.createElement('a');link.href='#'+section.id;link.textContent=item.title+' · '+item.owner;nav.append(link);
  }
  content.append(section);
 });
 doc.addEventListener('click',ev=>{
  const b=ev.target.closest('.tabbed-card-nav button');
  if(!b)return;const group=b.closest('.tabbed-card-group');if(!group)return;
  const items=all(group,'.tabbed-card-nav button'),at=items.indexOf(b);
  items.forEach((btn,i)=>btn.setAttribute('aria-selected',String(i===at)));
  all(group,'.tabbed-card-panel').forEach((p,i)=>p.hidden=i!==at);
 });
 win.focus();
}
async function open({report,chapters,editMode=false}){
 if(document.querySelector('.sir-dir'))return;
 const chosen=chapters.filter(included);
 const excluded=chapters.filter(ch=>!included(ch));
 if(!chosen.length){alert('Nessun capitolo di Angelo o HIGH BALLS di Max disponibile.');return}
 const records=new Map(chapters.map(ch=>[String(ch.section_id||ch.title),ch]));
 const images=new Map;chapters.filter(included).forEach(ch=>source(ch).assets.forEach(asset=>images.set(asset.id,asset)));
 const srcSig=signature(chapters);
 const chapterSources=new Map(chapters.filter(included).map(ch=>[String(ch.section_id||ch.title),source(ch)]));
 let pages=createDeck(chapters),at=0,beat=0,editing=false,dirty=false,timer=null,saveInFlight=null,queuedRevision=0,closing=false,stale=false;
 const backupKey='sir-match-story-director:'+report.id;
 let local=null;
 try{local=JSON.parse(localStorage.getItem(backupKey)||'null')}catch(e){}
 const shell=document.createElement('section');shell.className='sir-dir';shell.setAttribute('role','dialog');shell.setAttribute('aria-modal','true');shell.setAttribute('aria-label','PRESENTA · Regia tecnica');
 shell.innerHTML='<header class="sd-top"><strong class="sd-brand">SIR MATCH STORY <span>/ PRESENTA</span></strong><span class="sd-match"></span><div class="sd-tools"><button class="sd-tool sd-primary" data-sd="report">RAPPORTINO COMPLETO ↗</button><button class="sd-tool" data-sd="edit">MODIFICA</button><button class="sd-tool" data-sd="full">SCHERMO INTERO</button><button class="sd-tool" data-sd="close">CHIUDI</button></div></header><main class="sd-main"><div class="sd-stage"><div class="sd-surface" id="sdSurface"></div></div><aside class="sd-sidebar" id="sdEditor" hidden><div class="sd-side-top"><button type="button" class="sd-editbtn" data-sd="close-editor" style="float:right;background:#17344f;color:white;border-color:#17344f;padding:12px 18px;font-weight:900">← TORNA ALLA SLIDE</button><strong>REGIA · MODIFICA SLIDE · iPAD 2</strong><p>Il report e le card originali rimangono invariati. Qui modifichi soltanto il racconto proiettato.</p></div><div class="sd-rich-toolbar" id="sdRichToolbar" role="toolbar" aria-label="Formattazione selettiva del testo"><strong>FORMATTARE PAROLE E FRASI</strong><small id="sdRichHint">Su iPad fai doppio tap sulla parola da modificare, poi scegli e applica il formato.</small><div class="sd-rich-line"><button type="button" data-sd-rich="bold" class="sd-rbutton" title="Grassetto"><b>G</b></button><button type="button" data-sd-rich="italic" class="sd-rbutton" title="Corsivo"><i>C</i></button><button type="button" data-sd-rich="underline" class="sd-rbutton" title="Sottolineato"><u>S</u></button><button type="button" data-sd-rich="list" class="sd-rbutton">• Elenco</button><button type="button" data-sd-rich="clear" class="sd-rbutton">Cancella formato</button></div><div class="sd-rich-line"><label for="sdWordSize">Dimensione parola</label><input type="number" id="sdWordSize" min="12" max="140" step="1" value="40" inputmode="numeric" aria-label="Dimensione in pixel della selezione"><button type="button" data-sd-rich="size" class="sd-rbutton sd-apply">APPLICA px</button><label for="sdWordColor">Colore</label><input type="color" id="sdWordColor" value="#d82c63" aria-label="Colore selezione"><button type="button" data-sd-rich="color" class="sd-rbutton sd-apply">APPLICA</button></div><div class="sd-rich-line"><label for="sdWordFont">Font parola</label><select id="sdWordFont"><option value="Arial">Arial</option><option value="Georgia">Georgia</option><option value="Verdana">Verdana</option><option value="Trebuchet MS">Trebuchet</option><option value="Times New Roman">Times New Roman</option><option value="Inter">Inter</option></select><button type="button" data-sd-rich="word-font" class="sd-rbutton sd-apply">APPLICA</button><span class="sd-rich-spacer"></span><button type="button" data-sd-rich="align-left" class="sd-rbutton" title="Allinea a sinistra">≡</button><button type="button" data-sd-rich="align-center" class="sd-rbutton" title="Centra">☷</button><button type="button" data-sd-rich="align-right" class="sd-rbutton" title="Allinea a destra">≣</button></div></div><div class="sd-field"><label for="sdEditTitle">TITOLO · puoi andare a capo e formattare parole</label><div id="sdEditTitle" class="sd-rich-input sd-rich-title" contenteditable="true" role="textbox" aria-multiline="true" spellcheck="true" data-placeholder="Scrivi il titolo della slide…"></div></div><div class="sd-field"><label for="sdEditBody">TESTO · scrivi liberamente, premi Invio per andare a capo</label><div id="sdEditBody" class="sd-rich-input sd-rich-body" contenteditable="true" role="textbox" aria-multiline="true" spellcheck="true" data-placeholder="Scrivi il testo. Seleziona una parola per ingrandirla…"></div><small>Puoi usare più righe, grassetto, corsivo e dimensioni diverse anche nella stessa frase. Se serve spazio, dividi il contenuto in due slide.</small></div><div class="sd-field"><label for="sdTitleSize">Dimensione titolo <output id="sdTitleSizeValue">Auto</output></label><input id="sdTitleSize" type="range" min="30" max="135" step="1" value="89"><small>Regola le dimensioni direttamente mentre guardi la slide.</small></div><div class="sd-field"><label for="sdBodySize">Dimensione testo <output id="sdBodySizeValue">Auto</output></label><input id="sdBodySize" type="range" min="18" max="52" step="1" value="34"></div><div class="sd-field"><label for="sdTitleFont">Font del titolo</label><select id="sdTitleFont"><option value="inter">Moderno · Inter / sistema</option><option value="arial">Arial · essenziale</option><option value="georgia">Georgia · editoriale</option><option value="verdana">Verdana · leggibile</option><option value="trebuchet">Trebuchet · dinamico</option><option value="times">Times · classico</option></select></div><div class="sd-field"><label for="sdBodyFont">Font del testo</label><select id="sdBodyFont"><option value="inter">Moderno · Inter / sistema</option><option value="arial">Arial · essenziale</option><option value="georgia">Georgia · editoriale</option><option value="verdana">Verdana · leggibile</option><option value="trebuchet">Trebuchet · dinamico</option><option value="times">Times · classico</option></select></div><p class="sd-font-help">Le dimensioni sono applicate alla slide. Se il testo non entra, verrà ridotto soltanto quanto necessario, senza tagliarlo.</p><div class="sd-field"><label for="sdEditLayout">Tipo di schermata</label><select id="sdEditLayout"><option value="cover">Apertura capitolo</option><option value="message">Messaggio forte</option><option value="split">Immagine + testo</option><option value="photo">Immagine protagonista</option><option value="summary">Sintesi</option></select></div><div class="sd-field"><label for="sdEditPhoto">Immagine della slide</label><select id="sdEditPhoto"></select><small>Scegli dal rapportino oppure carica una nuova immagine, solo per PRESENTA.</small></div><div class="sd-upload-area"><button type="button" class="sd-editbtn sd-upload-btn" data-sd="upload-image">+ CARICA IMMAGINE DAL DISPOSITIVO</button><input id="sdUploadImage" type="file" accept="image/*,.heic,.heif" aria-label="Seleziona foto o immagine" style="position:absolute;width:1px;height:1px;opacity:0;pointer-events:none"><p id="sdUploadState" role="status" aria-live="polite">Puoi scegliere una foto dal dispositivo. Sarà salvata con questa presentazione.</p></div><div class="sd-assets-preview" id="sdImagePreview">Nessuna immagine selezionata</div><div class="sd-editor-actions"><button class="sd-editbtn" data-sd="up">↑ Sposta</button><button class="sd-editbtn" data-sd="down">↓ Sposta</button><button class="sd-editbtn" data-sd="add">+ Slide</button><button class="sd-editbtn sd-danger" data-sd="delete">Elimina slide</button></div><div class="sd-side-divider"></div><div class="sd-editor-actions"><button class="sd-editbtn sd-save" data-sd="save">Salva adesso</button><button class="sd-editbtn" data-sd="reset">Rigenera dalle card</button></div><div class="sd-save-state" id="sdSaveState">Le modifiche si salvano automaticamente.</div></aside><div class="sd-nav-shade" id="sdNavigator" hidden><div class="sd-nav-panel"><header class="sd-nav-h"><div><small>INDICE DELLA PRESENTAZIONE</small><strong>I capitoli del racconto</strong></div><button class="sd-tool" data-sd="index-close">CHIUDI</button></header><div class="sd-nav-section">PRESENTA · ANGELO (ESCLUSA LA BUSSOLA) E HIGH BALLS</div><div class="sd-nav-grid" id="sdChapters"></div><div class="sd-nav-other"><div class="sd-nav-section">ALTRI CAPITOLI · ANTEPRIMA ORIGINALE</div><p class="sd-nav-msg">Gli altri capitoli, compresa LA BUSSOLA, restano consultabili dal rapportino completo e in Anteprima.</p><div class="sd-nav-grid" id="sdExtras"></div></div></div></div><div class="sd-toast" id="sdToast" hidden></div></main><footer class="sd-footer-bar"><button class="sd-arrow" data-sd="previous" aria-label="Indietro">←</button><button class="sd-arrow sd-forward" data-sd="next" aria-label="Avanti">→</button><span class="sd-count" id="sdCount">01 / 01</span><div class="sd-progress" id="sdProgress"></div><span class="sd-btm-label" id="sdDetailHint">→ mostrerà il dettaglio successivo</span></footer>';
 document.body.append(shell);
 const previousFocus=document.activeElement,oldOverflow=document.body.style.overflow;
 document.body.style.overflow='hidden';
 const stage=$(shell,'.sd-stage'),surface=$(shell,'#sdSurface'),editor=$(shell,'#sdEditor');
 // REGIA lives directly under the full-screen root, never inside the flex slide area.
 // This avoids iPad Safari clipping / off-screen fixed descendants.
 shell.appendChild(editor);
 editor.style.cssText='position:fixed!important;inset:0!important;z-index:2147480010!important;width:100vw!important;height:100dvh!important;max-width:none!important;max-height:100dvh!important;min-width:0!important;min-height:0!important;overflow-y:auto!important;overflow-x:hidden!important;overscroll-behavior:contain!important;-webkit-overflow-scrolling:touch!important;touch-action:pan-y!important;padding:0 0 110px!important;background:#f7f8f9!important;color:#172a3b!important;box-sizing:border-box!important';
 const navigator=$(shell,'#sdNavigator'),progress=$(shell,'#sdProgress'),chaptersEl=$(shell,'#sdChapters'),extraEl=$(shell,'#sdExtras');
 const titleInput=$(shell,'#sdEditTitle'),bodyInput=$(shell,'#sdEditBody'),layoutInput=$(shell,'#sdEditLayout'),photoInput=$(shell,'#sdEditPhoto');
 const titleSizeInput=$(shell,'#sdTitleSize'),bodySizeInput=$(shell,'#sdBodySize'),titleFontInput=$(shell,'#sdTitleFont'),bodyFontInput=$(shell,'#sdBodyFont');
 const wordSizeInput=$(shell,'#sdWordSize'),wordColorInput=$(shell,'#sdWordColor'),wordFontInput=$(shell,'#sdWordFont'),richHint=$(shell,'#sdRichHint');
 const previewEl=$(shell,'#sdImagePreview'),status=$(shell,'#sdSaveState'),toaster=$(shell,'#sdToast');
 const uploadInput=$(shell,'#sdUploadImage'),uploadState=$(shell,'#sdUploadState');
 $(shell,'.sd-match').textContent=report.title||'MATCH REPORT';
 let ro=null;
 const showToast=msg=>{toaster.textContent=msg;toaster.hidden=false;clearTimeout(showToast.handle);showToast.handle=setTimeout(()=>toaster.hidden=true,2600)};
 const imageRef=page=>page.customImageUrl&&validImage(page.customImageUrl) ?{id:'direct-upload',src:page.customImageUrl,alt:page.customImageName||'Immagine caricata in REGIA'}:images.get(page.imageId||'');
 const backgroundPanel=backgrounds.build(editor,{reportId:report.id,getPage:()=>pages[at],getPages:()=>pages,getImage:imageRef,changed:mark,draw:()=>show(false),save:()=>save(false)});
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
   titleInput.innerHTML=rich.sanitize(p.titleHtml||rich.fromPlain(p.title));
   bodyInput.innerHTML=rich.sanitize(p.bodyHtml||rich.fromPlain(p.body));
   titleInput.style.textAlign=p.titleAlign||'left';bodyInput.style.textAlign=p.bodyAlign||'left';
   layoutInput.value=p.layout;
   photoInput.replaceChildren();
   const empty=new Option('Senza immagine','');photoInput.append(empty);
   if(p.customImageUrl&&validImage(p.customImageUrl))photoInput.add(new Option('CARICATA · '+(p.customImageName||'Immagine').slice(0,55),'__uploaded__'));
   const origin=records.get(p.chapterId);if(origin){
     (chapterSources.get(p.chapterId)?.assets||[]).forEach((asset,i)=>{photoInput.add(new Option((i+1)+'. '+shrink(asset.alt||asset.hint,60),asset.id))});
   }
   photoInput.value=p.customImageUrl?'__uploaded__':(p.imageId||'');
   if(photoInput.selectedIndex<0)photoInput.value='';
   chooseImage();
   backgroundPanel.refresh();
   titleSizeInput.value=p.titleSize||89;bodySizeInput.value=p.bodySize||34;titleFontInput.value=p.titleFont||'inter';bodyFontInput.value=p.bodyFont||'inter';
   $(shell,'#sdTitleSizeValue').textContent=p.titleSize?p.titleSize+' px':'Auto';
   $(shell,'#sdBodySizeValue').textContent=p.bodySize?p.bodySize+' px':'Auto';
 }
 const fontFamilies={inter:'Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Arial,sans-serif',arial:'Arial,Helvetica,sans-serif',georgia:'Georgia,Times New Roman,serif',verdana:'Verdana,Geneva,sans-serif',trebuchet:'Trebuchet MS,Arial,sans-serif',times:'Times New Roman,Times,serif'};
 function dynamicFont(el,preferred,min,allowedHeight){
   let v=preferred;
   el.style.fontSize=v+'px';
   while(v>min&&(el.scrollHeight>allowedHeight||el.scrollWidth>el.clientWidth+4)){v-=3;el.style.fontSize=v+'px'}
 }
 function show(syncForm=true){
   if(!pages.length)return;
   at=Math.min(Math.max(0,at),pages.length-1);
   const p=pages[at],asset=imageRef(p),hasDetail=!!(p.body?.trim()||asset),revealed=true;
   surface.replaceChildren();
   const scene=document.createElement('article');scene.className='sd-slide '+(p.owner==='Max'?'sd-max':'')+' sd-'+p.layout+(p.layout==='split'||p.layout==='summary'?' sd-light':'');
   backgrounds.apply(scene,p);
   const header=document.createElement('div');header.className='sd-eyebrow';
   const tag=document.createElement('span');tag.textContent='SIR MATCH STORY  /  '+(p.chapterTitle||'');
   const by=document.createElement('span');by.className='sd-owner';by.textContent=p.owner==='Max'?'MAX':'ANGELO';
   header.append(tag,by);scene.append(header);
   const main=document.createElement('div');main.className='sd-body';
   const head=document.createElement('h2');head.className='sd-h';
   if(p.titleHtml)head.innerHTML=rich.sanitize(p.titleHtml);
   else head.textContent=p.title||p.chapterTitle||'';
   head.style.fontFamily=fontFamilies[p.titleFont]||fontFamilies.inter;
   head.style.textAlign=p.titleAlign||'left';
   const desc=document.createElement('div');desc.className='sd-content';
   if(p.bodyHtml)desc.innerHTML=rich.sanitize(p.bodyHtml);
   else desc.textContent=p.body||'';
   desc.hidden=!revealed||(!p.body&&!p.bodyHtml&&!editing);
   desc.style.fontFamily=fontFamilies[p.bodyFont]||fontFamilies.inter;
   desc.style.textAlign=p.bodyAlign||'left';
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
   const foot=document.createElement('div');foot.className='sd-foot';const left=document.createElement('small');left.textContent='LA LETTURA · IL MESSAGGIO · IL GIOCO';const right=document.createElement('small');right.textContent=String(at+1).padStart(2,'0')+' / '+String(pages.length).padStart(2,'0');
   foot.append(left);
   if(validVideo(p.videoUrl)){const link=document.createElement('button');link.type='button';link.className='sd-clip-link';link.textContent='▶ APRI VIDEO';link.onclick=()=>window.open(p.videoUrl,'_blank','noopener');foot.append(link)}
   foot.append(right);scene.append(foot);
   surface.append(scene);
   if(editing){
     for(const [el,field,limit] of [[head,'title',170],[desc,'body',1200]]){
       el.contentEditable='true';el.spellcheck=true;el.setAttribute('role','textbox');el.setAttribute('aria-label',field==='title'?'Modifica titolo della slide':'Modifica testo della slide');
       el.dataset.sdEditable=field;el.dataset.placeholder=field==='title'?'Scrivi il titolo…':'Clicca e scrivi il testo…';
       el.addEventListener('input',()=>{
         const value=rich.plainFromEditor(el).slice(0,limit);
         pages[at][field]=value;
         pages[at][field+'Html']=rich.sanitize(el.innerHTML);
         mark();
       });
       el.addEventListener('blur',()=>{if(editing){const value=rich.plainFromEditor(el).slice(0,limit);if(pages[at][field]!==value){pages[at][field]=value;pages[at][field+'Html']=rich.sanitize(el.innerHTML);mark()}}});
       el.addEventListener('keydown',e=>e.stopPropagation());
     }
   }
   // Fixed 1280x720 projection geometry: text is reduced only when it cannot fit its assigned column.
   requestAnimationFrame(()=>{
     if(!scene.isConnected)return;
     const maxTitle=p.layout==='cover'?130:p.layout==='message'?107:p.layout==='summary'?84:p.layout==='split'?68:64;
     const titleSpace=(p.layout==='photo'||p.layout==='split')?360:480;
     dynamicFont(head,p.titleSize?Number(p.titleSize):maxTitle,24,titleSpace);
     if(!desc.hidden){const bodyMax=p.layout==='cover'?32:p.layout==='message'?37:p.layout==='summary'?40:p.layout==='split'?31:28;dynamicFont(desc,p.bodySize?Number(p.bodySize):bodyMax,16,p.layout==='photo'?320:p.layout==='split'?360:355)}
   });
   $(shell,'#sdCount').textContent=String(at+1).padStart(2,'0')+' / '+String(pages.length).padStart(2,'0');
   $(shell,'#sdDetailHint').textContent='TUTTI GLI ELEMENTI DELLA SLIDE SONO VISIBILI';
   $(shell,'[data-sd="previous"]').disabled=at===0;
   $(shell,'[data-sd="next"]').disabled=at===pages.length-1;
   progress.replaceChildren();
   pages.forEach((page,i)=>{
     const b=document.createElement('button');b.type='button';b.className='sd-tick'+(i===at?' is-active':i<at?' is-past':'');
     b.title=String(i+1)+' · '+page.title;b.setAttribute('aria-label','Vai alla slide '+(i+1));
     b.onclick=()=>{at=i;beat=0;show()};progress.append(b)
   });
   if(syncForm)editForm();resize();
 }
 function next(){if(at<pages.length-1){at++;show()}}
 function prev(){if(at>0){at--;show()}}
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
   if(!excluded.length){const p=document.createElement('p');p.className='sd-nav-msg';p.textContent='Tutti i capitoli sono già in PRESENTA.';extraEl.append(p)}
   excluded.forEach(ch=>{const b=document.createElement('button');b.type='button';b.className='sd-preview-link';b.textContent='↗ '+ch.title;b.onclick=()=>preview(ch,report);extraEl.append(b)})
 }
 const delegated=async ev=>{
   const btn=ev.target.closest('[data-sd]');if(!btn)return;
   const action=btn.dataset.sd;
   if(action==='report')preview(chapters,report);
   if(action==='next')next();
   if(action==='previous')prev();
   if(action==='index-close')navigator.hidden=true;
   if(action==='full'){
     try{if(document.fullscreenElement===shell)await document.exitFullscreen();else await shell.requestFullscreen()}catch(e){showToast('Schermo intero non disponibile')}
   }
   if(action==='edit'){
     editing=!editing;editor.hidden=!editing;
     if(editing)editor.scrollTop=0;
     btn.textContent=editing?'FINE MODIFICA':'MODIFICA';
     show();resize()
   }
   if(action==='close-editor'){$(shell,'[data-sd="edit"]').click()}
   if(action==='upload-image')uploadInput.click();
   if(action==='up'&&at>0){pages.splice(at-1,0,pages.splice(at,1)[0]);at--;mark();show()}
   if(action==='down'&&at<pages.length-1){pages.splice(at+1,0,pages.splice(at,1)[0]);at++;mark();show()}
   if(action==='add'){
     const origin=pages[at];
     const newPage={...origin,id:origin.chapterId+'/custom-'+crypto.randomUUID(),title:'NUOVA SLIDE',body:'',imageId:null,customImageUrl:null,customImageName:null,layout:'message'};
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

 let currentRich=bodyInput,lastRichRange=null,richPreviewTimer=null;
 const richEditors=[titleInput,bodyInput];
 function captureRichRange(){
   const sel=window.getSelection();
   if(!sel||!sel.rangeCount)return;
   const range=sel.getRangeAt(0);
   const found=richEditors.find(ed=>ed.contains(range.commonAncestorContainer));
   if(found){currentRich=found;lastRichRange=range.cloneRange();richHint.textContent=range.collapsed?'Scrivi normalmente. Per formattare una sola parola, prima selezionala.':'Selezione pronta: scegli un formato qui sopra.'}
 }
 document.addEventListener('selectionchange',captureRichRange);
 const updateRich=ed=>{
   const isTitle=ed===titleInput,field=isTitle?'title':'body',limit=isTitle?170:1200;
   const text=rich.plainFromEditor(ed);
   if(text.length>limit){
     richHint.textContent='Testo troppo lungo: limite '+limit+' caratteri. Riduci oppure crea un’altra slide.';
     return;
   }
   pages[at][field]=text;
   pages[at][field+'Html']=rich.sanitize(ed.innerHTML);
   mark();clearTimeout(richPreviewTimer);richPreviewTimer=setTimeout(()=>{if(shell.isConnected)show(false)},150);
 };
 richEditors.forEach(ed=>{
   ed.addEventListener('focus',()=>{currentRich=ed;captureRichRange()});
   ed.addEventListener('keyup',captureRichRange);
   ed.addEventListener('mouseup',captureRichRange);
   ed.addEventListener('touchend',()=>setTimeout(captureRichRange,60),{passive:true});
   ed.addEventListener('input',()=>updateRich(ed));
   ed.addEventListener('paste',ev=>{
     ev.preventDefault();
     const str=ev.clipboardData?.getData('text/plain')||'';
     document.execCommand('insertText',false,str.slice(0,2500));
   });
 });
 function applyRich(action){
   const ed=currentRich||bodyInput,field=ed===titleInput?'title':'body';
   if(action.startsWith('align-')){
     const alignment=action.slice(6);
     pages[at][field+'Align']=alignment;
     ed.style.textAlign=alignment;
     mark();show(false);return;
   }
   if(!lastRichRange||lastRichRange.collapsed||!ed.contains(lastRichRange.commonAncestorContainer)){
     richHint.textContent='Prima seleziona una parola o una frase nel titolo o nel testo.';
     return;
   }
   let ok=false;
   if(action==='size'){
     const size=Math.max(12,Math.min(140,Number(wordSizeInput.value)||36));
     ok=rich.applySpan(ed,lastRichRange,'fontSize',size+'px');
   }else if(action==='color'){
     ok=rich.applySpan(ed,lastRichRange,'color',wordColorInput.value);
   }else if(action==='word-font'){
     ok=rich.applySpan(ed,lastRichRange,'fontFamily',wordFontInput.value);
   }else{
     const command={bold:'bold',italic:'italic',underline:'underline',list:'insertUnorderedList',clear:'removeFormat'}[action];
     if(command)ok=rich.formatCommand(ed,lastRichRange,command);
   }
   if(ok){
     richHint.textContent='Formato applicato. Puoi selezionare altre parole.';
     const sel=window.getSelection();lastRichRange=sel?.rangeCount?sel.getRangeAt(0).cloneRange():null;
   }else richHint.textContent='Seleziona una sola parola o una frase e riprova.';
 }
 shell.addEventListener('click',ev=>{
   const button=ev.target.closest('[data-sd-rich]');
   if(!button)return;
   ev.preventDefault();
   applyRich(button.dataset.sdRich);
 });

 layoutInput.addEventListener('change',()=>updateScene('layout',layoutInput.value));
 photoInput.addEventListener('change',()=>{const p=pages[at],value=photoInput.value;if(value==='__uploaded__')return;p.customImageUrl=null;p.customImageName=null;p.imageId=value||null;mark();show()});

 let uploadBusy=false;
 uploadInput.addEventListener('change',async()=>{
   const file=uploadInput.files?.[0];
   uploadInput.value='';
   if(!file||uploadBusy)return;
   const originalSlide=pages[at];
   if(!originalSlide)return;
   uploadBusy=true;
   const button=$(shell,'[data-sd="upload-image"]');
   button.disabled=true;
   try{
     if(!window.SirPresentaPhotos)throw Error('Caricamento immagini non disponibile. Aggiorna la pagina.');
     const asset=await window.SirPresentaPhotos.upload(report.id,file,message=>{uploadState.textContent=message});
     if(!pages.includes(originalSlide))throw Error('La slide è stata eliminata durante il caricamento.');
     originalSlide.customImageUrl=asset.src;
     originalSlide.customImageName=asset.name;
     originalSlide.imageId=null;
     // A slide needs a media-capable layout to display the uploaded photo.
     if(!['split','photo'].includes(originalSlide.layout))originalSlide.layout='split';
     mark();
     if(originalSlide===pages[at])show();
     const ok=await save(false);
     uploadState.textContent=ok?'✓ Immagine aggiunta e salvata nella slide.':'⚠ Immagine caricata ma la slide non è stata salvata online. Premi Salva adesso.';
     if(ok)showToast('Immagine inserita nella slide');
   }catch(e){
     uploadState.textContent='⚠ '+(e.message||'Caricamento non riuscito.');
     showToast('Non è stato possibile inserire la foto');
   }finally{
     uploadBusy=false;
     button.disabled=false;
   }
 });

 titleSizeInput.addEventListener('input',()=>{
   pages[at].titleSize=Number(titleSizeInput.value);$(shell,'#sdTitleSizeValue').textContent=titleSizeInput.value+' px';mark();show(false);
 });
 bodySizeInput.addEventListener('input',()=>{
   pages[at].bodySize=Number(bodySizeInput.value);$(shell,'#sdBodySizeValue').textContent=bodySizeInput.value+' px';mark();show(false);
 });
 titleFontInput.addEventListener('change',()=>{pages[at].titleFont=titleFontInput.value;mark();show(false)});
 bodyFontInput.addEventListener('change',()=>{pages[at].bodyFont=bodyFontInput.value;mark();show(false)});
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
   else if(e.key.toLowerCase()==='e'){e.preventDefault();$(shell,'[data-sd="edit"]').click()}
   else if(e.key.toLowerCase()==='f'){e.preventDefault();$(shell,'[data-sd="full"]').click()}
 };
 async function close(){
   if(closing)return;closing=true;
   if(dirty){const okay=await save(true);if(!okay&&!confirm('Non riesco a salvare online. La copia locale resta in questo browser. Vuoi chiudere?')){closing=false;return}}
   if(ro)ro.disconnect();
   clearTimeout(richPreviewTimer);
   document.removeEventListener('selectionchange',captureRichRange);
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