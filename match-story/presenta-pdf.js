/* SIR PRESENTA · PDF export. Slides are captured at native 1280x720,
   one by one, so iPad never needs a huge canvas covering all slides. */
(function(){
'use strict';
const html2canvasSources=[
 'https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js',
 'https://unpkg.com/html2canvas@1.4.1/dist/html2canvas.min.js'
];
const jsPdfSources=[
 'https://cdn.jsdelivr.net/npm/jspdf@2.5.2/dist/jspdf.umd.min.js',
 'https://unpkg.com/jspdf@2.5.2/dist/jspdf.umd.min.js'
];
const sleepFrame=()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
let librariesPromise=null;
function loadScript(url){
 return new Promise((resolve,reject)=>{
  const node=document.createElement('script');node.src=url;node.async=true;
  const timeout=setTimeout(()=>{node.remove();reject(Error('Caricamento libreria PDF non disponibile.'))},20000);
  node.onload=()=>{clearTimeout(timeout);resolve()};
  node.onerror=()=>{clearTimeout(timeout);node.remove();reject(Error('Impossibile caricare '+url.split('/')[2]))};
  document.head.append(node);
 });
}
async function useSources(list,isReady){
 if(isReady())return;
 let failure;
 for(const url of list){
  try{await loadScript(url);if(isReady())return}
  catch(e){failure=e}
 }
 throw failure||Error('Libreria PDF non disponibile: controlla la connessione.');
}
function libraries(){
 if(!librariesPromise)librariesPromise=(async()=>{
  await useSources(html2canvasSources,()=>typeof window.html2canvas==='function');
  await useSources(jsPdfSources,()=>typeof window.jspdf?.jsPDF==='function');
 })().catch(e=>{librariesPromise=null;throw e});
 return librariesPromise;
}
function safeName(title){
 const base=String(title||'SIR-Match-Story').normalize('NFKD')
  .replace(/[\u0300-\u036f]/g,'').replace(/[^a-zA-Z0-9 _.-]+/g,'')
  .trim().replace(/\s+/g,'-').slice(0,90);
 return (base||'SIR-Match-Story')+'-PRESENTA.pdf';
}
function overlay(total,onCancel){
 const blocker=document.createElement('div');
 blocker.className='sd-pdf-overlay';
 blocker.setAttribute('role','dialog');
 blocker.setAttribute('aria-modal','true');
 blocker.setAttribute('aria-label','Esportazione PDF della presentazione');
 blocker.innerHTML='<div class="sd-pdf-dialog"><h2>ESPORTAZIONE PDF</h2><p id="sdPdfStatus">Preparo le slide…</p><progress id="sdPdfProgress" max="1" value="0"></progress><p class="sd-pdf-detail">Il PDF mantiene le proporzioni 16:9, con una pagina per slide.</p><div class="sd-pdf-actions"><button type="button" id="sdPdfCancel">ANNULLA</button><a id="sdPdfDownload" hidden download>SCARICA IL PDF</a><button type="button" id="sdPdfClose" hidden>CHIUDI</button></div></div>';
 const label=blocker.querySelector('#sdPdfStatus'),bar=blocker.querySelector('#sdPdfProgress');
 bar.max=total;
 blocker.querySelector('#sdPdfCancel').onclick=onCancel;
 const close=blocker.querySelector('#sdPdfClose');
 close.onclick=()=>{const link=blocker.querySelector('#sdPdfDownload');if(link.href?.startsWith('blob:'))URL.revokeObjectURL(link.href);blocker.remove()};
 document.body.append(blocker);
 return{
  status(msg){label.textContent=msg},
  update(i,msg){bar.value=i;label.textContent=msg},
  complete(pdfBlob,filename){
   const link=blocker.querySelector('#sdPdfDownload');
   const url=URL.createObjectURL(pdfBlob);link.href=url;link.download=filename;link.hidden=false;
   blocker.querySelector('#sdPdfCancel').hidden=true;
   close.hidden=false;bar.value=total;
   label.textContent='PDF pronto: '+total+' slide esportate.';
   const advice=blocker.querySelector('.sd-pdf-detail');
   advice.textContent='Su iPad tocca SCARICA IL PDF e scegli dove salvarlo. Il file include tutte le slide.';
   // Auto-download on desktop, but retain a direct user-gesture link for iPad Safari.
   if(!/iPad|iPhone|iPod|Macintosh.*Mobile/i.test(navigator.userAgent))link.click();
  },
  fail(message){
   label.textContent='Impossibile completare il PDF: '+message;
   blocker.querySelector('#sdPdfCancel').hidden=true;close.hidden=false;
   blocker.querySelector('.sd-pdf-detail').textContent='La presentazione originale e le slide salvate non sono state modificate.';
  },
  dismiss(){blocker.remove()}
 }
}
function exportStage(){
 const host=document.createElement('div');
 host.className='sir-dir sd-pdf-capture';
 host.setAttribute('aria-hidden','true');
 host.style.cssText='position:fixed!important;top:0!important;left:0!important;right:auto!important;bottom:auto!important;inset:auto!important;width:1280px!important;height:720px!important;max-width:none!important;min-height:0!important;display:block!important;z-index:2147482000!important;overflow:hidden!important;pointer-events:none!important;transform:none!important;opacity:1!important;';
 const surface=document.createElement('div');
 surface.className='sd-surface';
 surface.style.cssText='width:1280px!important;height:720px!important;transform:none!important;position:relative!important;overflow:hidden!important;box-shadow:none!important;border:0!important;';
 host.append(surface);
 document.body.append(host);
 return{host,surface};
}
function waitForImages(root){
 const images=Array.from(root.querySelectorAll('img'));
 return Promise.all(images.map(img=>{
  img.loading='eager';img.decoding='sync';
  if(img.complete&&img.naturalWidth)return Promise.resolve();
  return new Promise((resolve,reject)=>{
   const timeout=setTimeout(()=>{img.removeEventListener('load',ok);img.removeEventListener('error',bad);reject(Error('Una immagine non è stata caricata. Riprova con una connessione stabile.'))},14000);
   function ok(){clearTimeout(timeout);img.removeEventListener('error',bad);resolve()}
   function bad(){clearTimeout(timeout);img.removeEventListener('load',ok);reject(Error('Immagine non disponibile nella slide.'))}
   img.addEventListener('load',ok,{once:true});img.addEventListener('error',bad,{once:true});
  })
 }));
}
async function create({count,getIndex,render,scene,select,title}){
 if(!count||count>180)throw Error('Numero di slide non valido.');
 let cancelled=false;
 const ui=overlay(count,()=>{cancelled=true;ui.status('Annullamento in corso…')});
 const original=getIndex();
 let capture=null,doc=null;
 try{
  ui.status('Carico i componenti per creare il PDF…');
  await libraries();
  if(cancelled){ui.dismiss();return}
  if(document.fonts?.ready)await document.fonts.ready;
  capture=exportStage();
  const {jsPDF}=window.jspdf;
  doc=new jsPDF({orientation:'landscape',unit:'px',format:[1280,720],hotfixes:['px_scaling'],compress:true});
  doc.setProperties({title:String(title||'SIR PRESENTA'),subject:'Presentazione SIR Match Story'});
  for(let i=0;i<count;i++){
   if(cancelled){ui.dismiss();return}
   select(i);
   render();
   await sleepFrame();
   const originalScene=scene();
   if(!originalScene)throw Error('La slide '+(i+1)+' non è stata renderizzata.');
   const slide=originalScene.cloneNode(true);
   slide.querySelectorAll('[contenteditable]').forEach(el=>{el.removeAttribute('contenteditable');el.removeAttribute('role');el.removeAttribute('data-sd-editable');el.removeAttribute('data-placeholder')});
   slide.querySelectorAll('button').forEach(el=>{el.blur?.()});
   capture.surface.replaceChildren(slide);
   await waitForImages(slide);
   await sleepFrame();
   if(cancelled){ui.dismiss();return}
   ui.update(i,'Esporto la slide '+(i+1)+' di '+count+'…');
   const canvas=await window.html2canvas(capture.surface,{
    backgroundColor:'#ffffff',scale:1,useCORS:true,allowTaint:false,
    imageTimeout:14000,logging:false,scrollX:0,scrollY:0,
    windowWidth:1280,windowHeight:720,
    onclone:cloned=>{
      const node=cloned.querySelector('.sd-pdf-capture');
      if(node)node.style.setProperty('transform','none','important');
    }
   });
   if(canvas.width<1200||canvas.height<690)throw Error('Dimensioni PDF non corrette nella slide '+(i+1)+'.');
   if(i)doc.addPage([1280,720],'landscape');
   doc.addImage(canvas.toDataURL('image/jpeg',.85),'JPEG',0,0,1280,720,undefined,'FAST');
   canvas.width=1;canvas.height=1;
   ui.update(i+1,'Slide '+(i+1)+' di '+count+' esportate.');
   // Give Safari a chance to paint the status and release the previous canvas.
   await new Promise(resolve=>setTimeout(resolve,0));
  }
  const pdf=doc.output('blob');
  if(!pdf||pdf.size<500)throw Error('Il file PDF è vuoto.');
  ui.complete(pdf,safeName(title));
 }catch(e){
  ui.fail(e?.message||'Errore inatteso');
 }finally{
  if(capture)capture.host.remove();
  select(original);render();
 }
}
window.SirPresentaPdf={create};
})();