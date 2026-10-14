/* Editor visuale Presenta, separato dai capitoli originali. */
(function(){
"use strict";
const forbidden='script,style,iframe,object,embed,form,input,select,textarea,button,svg,math,link,meta,base';
function clean(html){
 const tpl=document.createElement('template');tpl.innerHTML=String(html||'');
 tpl.content.querySelectorAll(forbidden).forEach(e=>e.remove());
 tpl.content.querySelectorAll('*').forEach(el=>{
  for(const a of [...el.attributes]){
   const k=a.name.toLowerCase(),v=a.value;
   if(k.startsWith('on')||['srcdoc','contenteditable','autofocus','formaction'].includes(k)){el.removeAttribute(k);continue}
   if(k==='style'&&/(url\s*\(|expression\s*\(|@import)/i.test(v)){el.removeAttribute(k);continue}
   if(k==='src'||k==='href'){
    try{const u=new URL(v,location.href);if(!['http:','https:'].includes(u.protocol)&&!(k==='src'&&/^data:image\/(png|jpeg|gif|webp);base64,/i.test(v)))el.removeAttribute(k)}catch{el.removeAttribute(k)}
   }
  }
  el.classList.remove('sir-ed-selected');
 });
 return tpl.innerHTML;
}
function restore(saved,canvas,pages){
 for(const item of saved){
  if(!item||typeof item.html!=='string'||!String(item.className).startsWith('sir-presentation-slide'))throw Error('Presentazione salvata non valida');
  const slide=document.createElement('article');
  slide.className=String(item.className).split(/\s+/).filter(c=>/^sir-presentation-[\w-]+$/.test(c)).join(' ');
  slide.innerHTML=clean(item.html);
  if(!slide.querySelector('.sir-presentation-heading')||!slide.querySelector('.sir-presentation-column')||!slide.querySelector('.sir-presentation-footer'))throw Error('Pagina salvata incompleta');
  slide.hidden=true;canvas.append(slide);pages.push(slide);
 }
}
function serialize(pages){
 return pages.map(slide=>{
  const clone=slide.cloneNode(true);
  clone.querySelectorAll('.sir-ed-selected').forEach(e=>e.classList.remove('sir-ed-selected'));
  clone.querySelectorAll('[contenteditable]').forEach(e=>e.removeAttribute('contenteditable'));
  return {className:[...slide.classList].filter(c=>c.startsWith('sir-presentation-')).join(' '),html:clean(clone.innerHTML)};
 });
}
function install(o){
 const {overlay,canvas,pages,report,api,render,getIndex,setIndex,reset}=o;
 let selected=null,dragMode=false,dragState=null,dirty=false,saving=false;
 let revision=0,saveTimer=null,backupTimer=null,retryTimer=null,saveInFlight=null,paused=false;
 const backup=window.SirMatchAutosave?.presentation;
 function preserve(){if(dirty&&!paused)backup?.backup(report.id,serialize(pages))}
 function schedule(delay=2800){clearTimeout(saveTimer);if(!paused)saveTimer=setTimeout(()=>{save(true).catch(()=>{})},delay)}
 function onVisibility(){if(document.hidden&&dirty){preserve();save(true).catch(()=>{})}}
 function onPageHide(){if(dirty)preserve()}
 function onBeforeUnload(e){if(dirty||saving){preserve();e.preventDefault();e.returnValue=''}}
 document.addEventListener('visibilitychange',onVisibility);
 window.addEventListener('pagehide',onPageHide);
 window.addEventListener('beforeunload',onBeforeUnload);

 const toolbar=document.createElement('section');toolbar.className='sir-editor-toolbar';
 toolbar.innerHTML="<div class=\"sir-ed-top\"><strong>MODIFICA PRESENTA</strong><button data-ed=\"save\">Salva modifiche</button><button data-ed=\"reset\">Rigenera dalle card</button><span data-ed-status>Modifiche solo a questa partita</span></div><div class=\"sir-ed-controls\"><span>Seleziona testo, card o immagine.</span><button data-ed=\"drag\">Sposta: OFF</button><button data-ed=\"text\">Modifica testo</button><button data-ed=\"font-down\">A−</button><button data-ed=\"font-up\">A+</button><button data-ed=\"left\">Sinistra</button><button data-ed=\"center\">Centro</button><button data-ed=\"right\">Destra</button><button data-ed=\"justify\">Giustifica</button><button data-ed=\"leading-down\">Interlinea −</button><button data-ed=\"leading-up\">Interlinea +</button><button data-ed=\"spacing-down\">Spazio −</button><button data-ed=\"spacing-up\">Spazio +</button><button data-ed=\"size-down\">− Dimensione</button><button data-ed=\"size-up\">+ Dimensione</button><button data-ed=\"make-text\">+ Testo</button><button data-ed=\"split\">Sposta in nuova pagina</button><button data-ed=\"remove\">Elimina elemento</button><button data-ed=\"previous\">Pagina ←</button><button data-ed=\"following\">Pagina →</button><button data-ed=\"add\">+ Pagina</button><button data-ed=\"delete\">Elimina pagina</button></div>";
 const style=document.createElement('style');style.textContent="\n.sir-presentation-editor-open .sir-editor-toolbar{display:flex;flex-direction:column;gap:6px;padding:9px 11px;max-height:188px;overflow:auto;background:#0b233a;border-bottom:1px solid #47607e;color:#fff;z-index:12}\n.sir-presentation .sir-ed-top,.sir-presentation .sir-ed-controls{display:flex;align-items:center;gap:5px;flex-wrap:wrap}\n.sir-presentation .sir-ed-top strong{color:#ff90ac;font-size:13px;letter-spacing:1px;margin-right:10px}\n.sir-presentation .sir-ed-top [data-ed-status]{font-size:12px;color:#c6d7ec;margin-left:8px}\n.sir-presentation .sir-editor-toolbar button{padding:6px 9px;font-size:12px;border-radius:6px}\n.sir-presentation .sir-editor-toolbar [data-ed=save]{background:#f72569;color:#fff;border-color:#f72569}\n.sir-presentation .sir-editor-toolbar [data-ed=drag][aria-pressed=true]{background:#eff6ff;color:#18334b}\n.sir-presentation-editor-open .sir-presentation-block,.sir-presentation-editor-open .sir-presentation-heading h2{cursor:pointer}\n.sir-presentation-editor-open .sir-ed-selected{outline:4px dashed #fa276d!important;outline-offset:4px;position:relative;z-index:6}\n.sir-presentation-editor-open [contenteditable=true]{outline:4px solid #1389cf!important;cursor:text!important}\n@media(max-width:700px){.sir-presentation-editor-open .sir-editor-toolbar{max-height:200px}.sir-presentation .sir-ed-controls>span{display:none}}\n";
 overlay.prepend(style);overlay.querySelector('.sir-presentation-bar').after(toolbar);
 overlay.classList.add('sir-presentation-editor-open');
 const status=toolbar.querySelector('[data-ed-status]');
 function message(t){status.textContent=t}
 function mark(){if(paused)return;dirty=true;revision++;message('Modifiche in attesa di salvataggio…');clearTimeout(backupTimer);backupTimer=setTimeout(preserve,350);schedule()}
 function page(){return pages[getIndex()]}
 function select(el){
  if(selected){selected.classList.remove('sir-ed-selected');selected.removeAttribute('contenteditable')}
  selected=el;if(el)el.classList.add('sir-ed-selected');
 }
 function fitPages(){
  canvas.replaceChildren(...pages);
  pages.forEach((p,i)=>{const f=p.querySelector('.sir-presentation-footer span:last-child');if(f)f.textContent=(i+1)+' / '+pages.length});
  render();
 }
 function blank(){
  const current=page(),slide=document.createElement('article');
  slide.className='sir-presentation-slide sir-presentation-text-page sir-presentation-text-single';
  slide.innerHTML='<div class="sir-presentation-heading"><small></small><h2></h2></div><div class="sir-presentation-content"><div class="sir-presentation-column"></div><div class="sir-presentation-column"></div></div><div class="sir-presentation-footer"><span></span><span></span></div>';
  slide.querySelector('small').textContent=current?.querySelector('.sir-presentation-heading small')?.textContent||'SIR MATCH STORY';
  slide.querySelector('h2').textContent=current?.querySelector('.sir-presentation-heading h2')?.textContent||report.title;
  slide.querySelector('.sir-presentation-footer span:first-child').textContent=report.title;
  return slide;
 }
 function font(delta){
  if(!selected||selected.tagName==='IMG')return;
  const sz=parseFloat(getComputedStyle(selected).fontSize)||25;
  selected.style.setProperty('font-size',Math.max(12,Math.min(64,sz+delta))+'px','important');
  if(selected.classList.contains('sir-presentation-block'))selected.querySelectorAll('h1,h2,h3,h4').forEach(h=>h.style.setProperty('font-size',Math.max(12,Math.min(64,sz+delta+3))+'px','important'));
  mark();
 }
 function size(d){
  if(!selected)return;
  if(selected.tagName==='IMG'){
   const width=selected.getBoundingClientRect().width,container=selected.closest('.sir-presentation-column')?.getBoundingClientRect().width||600;
   const val=Math.max(10,Math.min(100,100*width/container+d*10));
   selected.style.setProperty('width',val+'%','important');selected.style.setProperty('height','auto','important');selected.style.setProperty('max-height','none','important');
  }else{
   const current=parseFloat(selected.style.width)||(100*selected.getBoundingClientRect().width/(selected.parentElement?.getBoundingClientRect().width||1));
   selected.style.width=Math.max(20,Math.min(100,current+d*10))+'%';
  }
  mark();
 }
 function shift(el,dx,dy){
  const x=Math.max(-1100,Math.min(1100,(Number(el.dataset.sirX)||0)+dx));
  const y=Math.max(-500,Math.min(500,(Number(el.dataset.sirY)||0)+dy));
  el.dataset.sirX=x;el.dataset.sirY=y;el.style.translate=x+'px '+y+'px';
 }
 async function save(automatic=false){
  clearTimeout(saveTimer);saveTimer=null;
  if(paused)return false;
  if(saveInFlight){const result=await saveInFlight;if(!result)return false;if(dirty)return save(automatic);return true}
  if(automatic&&!dirty)return true;
  saving=true;const btn=toolbar.querySelector('[data-ed=save]');btn.disabled=true;
  const version=revision,pagesToSave=serialize(pages);
  message('Salvataggio automatico online…');
  const operation=(async()=>{
   try{
    const response=await fetch(api+'?action=presentation-save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:report.id,pages:pagesToSave})});
    const result=await response.json();if(!response.ok)throw Error(result.error||'Errore di salvataggio');
    if(revision===version){dirty=false;await backup?.clear(report.id);message('✓ Salvato online · '+new Date().toLocaleTimeString('it-IT',{hour:'2-digit',minute:'2-digit'}));}
    else{dirty=true;preserve();schedule(900)}
    clearTimeout(retryTimer);
    return true;
   }catch(e){
    message('⚠ Salvataggio online non riuscito. Bozza locale mantenuta.');
    preserve();
    if(!automatic)alert('Presentazione non salvata: '+e.message);
    clearTimeout(retryTimer);retryTimer=setTimeout(()=>save(true).catch(()=>{}),20000);
    return false;
   }finally{btn.disabled=false;saving=false;saveInFlight=null}
  })();
  saveInFlight=operation;return operation;
 }
 toolbar.addEventListener('click',event=>{
  const b=event.target.closest('[data-ed]');if(!b)return;const a=b.dataset.ed;
  if(a==='save'){save(false);return}
  if(a==='reset'){if(confirm('Rigenerare tutte le pagine dalle card? Perderai le modifiche manuali salvate.'))reset();return}
  if(a==='drag'){dragMode=!dragMode;b.textContent=dragMode?'Sposta: ON':'Sposta: OFF';b.setAttribute('aria-pressed',String(dragMode));return}
  if(a==='text'){if(!selected||selected.tagName==='IMG'){message('Seleziona prima un testo.');return}selected.contentEditable='true';selected.focus();message('Modifica il testo nella pagina e poi salva.');return}
  if(a==='font-down'||a==='font-up'){font(a==='font-up'?2:-2);return}
  if(['left','center','right','justify'].includes(a)){if(selected){selected.style.setProperty('text-align',a,'important');mark()}return}
  if(a==='leading-down'||a==='leading-up'){
   if(!selected||selected.tagName==='IMG')return;
   const line=parseFloat(getComputedStyle(selected).lineHeight)/Math.max(parseFloat(getComputedStyle(selected).fontSize)||1,1)||1.3;
   const val=Math.max(0.9,Math.min(2.6,line+(a==='leading-up'?.1:-.1)));
   selected.style.setProperty('line-height',String(val),'important');mark();return
  }
  if(a==='spacing-down'||a==='spacing-up'){
   if(!selected)return;
   const old=parseFloat(selected.style.marginBottom)||0;
   selected.style.setProperty('margin-bottom',Math.max(-30,Math.min(160,old+(a==='spacing-up'?8:-8)))+'px','important');mark();return
  }
  if(a==='size-down'||a==='size-up'){size(a==='size-up'?1:-1);return}
  if(a==='make-text'){
   const column=page()?.querySelector('.sir-presentation-column');if(!column)return;
   const block=document.createElement('div');block.className='sir-presentation-block';block.innerHTML='<p>Scrivi qui il nuovo testo.</p>';
   column.append(block);select(block);block.contentEditable='true';block.focus();mark();return
  }
  if(a==='split'){
   if(!selected){message('Seleziona il testo, la card o la foto da spostare.');return}
   const block=selected.closest('.sir-presentation-block');if(!block){message('Seleziona un elemento della pagina.');return}
   const newPage=blank();newPage.querySelector('.sir-presentation-column').append(block);
   const i=getIndex()+1;pages.splice(i,0,newPage);setIndex(i);select(block);mark();fitPages();return
  }
  if(a==='remove'){
   if(!selected)return;if(!confirm('Eliminare questo elemento? I capitoli originali restano invariati.'))return;
   selected.remove();select(null);mark();return
  }
  if(a==='previous'||a==='following'){
   const from=getIndex(),to=from+(a==='previous'?-1:1);if(to<0||to>=pages.length)return;
   pages.splice(to,0,pages.splice(from,1)[0]);setIndex(to);select(null);mark();fitPages();return
  }
  if(a==='add'){
   const at=getIndex()+1;pages.splice(at,0,blank());setIndex(at);select(null);mark();fitPages();return
  }
  if(a==='delete'){
   if(pages.length<2||!confirm('Eliminare questa pagina?'))return;
   pages.splice(getIndex(),1)[0].remove();setIndex(Math.min(getIndex(),pages.length-1));select(null);mark();fitPages();
  }
 });
 canvas.addEventListener('click',event=>{
  const current=event.target.closest('.sir-presentation-slide');if(!pages.includes(current))return;
  const item=event.target.closest('.sir-presentation-block img,.sir-presentation-block,.sir-presentation-heading h2');
  if(item&&!event.target.closest('[contenteditable=true]')){event.preventDefault();select(item)}
 });
 canvas.addEventListener('input',event=>{if(event.target.closest('[contenteditable=true]'))mark()});
 canvas.addEventListener('pointerdown',event=>{
  if(!dragMode||event.button!==0)return;
  const item=event.target.closest('.sir-presentation-block img,.sir-presentation-block,.sir-presentation-heading h2');
  if(!item||!pages.includes(item.closest('.sir-presentation-slide')))return;
  select(item);
  const scale=canvas.getBoundingClientRect().width/1280||1;
  dragState={item,id:event.pointerId,x:event.clientX,y:event.clientY,scale,wasMoved:false};
  item.setPointerCapture?.(event.pointerId);event.preventDefault();
 });
 canvas.addEventListener('pointermove',event=>{
  if(!dragState||dragState.id!==event.pointerId)return;
  const dx=(event.clientX-dragState.x)/dragState.scale,dy=(event.clientY-dragState.y)/dragState.scale;
  if(Math.abs(dx)+Math.abs(dy)<0.5)return;
  shift(dragState.item,dx,dy);dragState.wasMoved=true;dragState.x=event.clientX;dragState.y=event.clientY;
 });
 function pointerStop(e){if(dragState?.id===e.pointerId){if(dragState.wasMoved)mark();dragState=null}}
 canvas.addEventListener('pointerup',pointerStop);canvas.addEventListener('pointercancel',pointerStop);
 return {
  isDirty:()=>dirty||saving,
  flush:()=>save(true),
  whenIdle:async()=>{if(saveInFlight)await saveInFlight},
  suspend:()=>{paused=true;clearTimeout(saveTimer);clearTimeout(backupTimer);clearTimeout(retryTimer)},
  resume:()=>{paused=false;if(dirty){preserve();schedule()}},
  markRecovered:()=>{mark();message('Bozza recuperata dal dispositivo · salvataggio in corso…')},
  cleanup:()=>{paused=true;clearTimeout(saveTimer);clearTimeout(backupTimer);clearTimeout(retryTimer);document.removeEventListener('visibilitychange',onVisibility);window.removeEventListener('pagehide',onPageHide);window.removeEventListener('beforeunload',onBeforeUnload);select(null);toolbar.remove();style.remove();overlay.classList.remove('sir-presentation-editor-open')}
 };
}
window.SirPresentationEditor={clean,restore,serialize,install};
})();