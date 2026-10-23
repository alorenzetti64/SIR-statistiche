(function marginsModule(){
'use strict';
const fields=['slideHeaderLeft','slideHeaderRight','slideFooterLeft','slideFooterRight','slideFooterPageNumber','slideHeaderSize','slideFooterSize'];
const has=(o,k)=>Object.prototype.hasOwnProperty.call(o,k);
const value=(p,k,def)=>has(p,k)?String(p[k]??''):def;
const defaults=p=>({
 slideHeaderLeft:'SIR MATCH STORY  /  '+(p.chapterTitle||''),
 slideHeaderRight:p.owner==='Max'?'MAX':'ANGELO',
 slideFooterLeft:'LA LETTURA · IL MESSAGGIO · IL GIOCO',
 slideFooterRight:'',
 slideFooterPageNumber:true,
 slideHeaderSize:15,
 slideFooterSize:13
});
const clamp=(v,min,max,df)=>Number.isFinite(Number(v))?Math.min(max,Math.max(min,Number(v))):df;
function apply(page,parts,position,total){
 const base=defaults(page);
 parts.headerLeft.textContent=value(page,'slideHeaderLeft',base.slideHeaderLeft);
 parts.headerRight.textContent=value(page,'slideHeaderRight',base.slideHeaderRight);
 parts.footerLeft.textContent=value(page,'slideFooterLeft',base.slideFooterLeft);
 const auto=page.slideFooterPageNumber!==false;
 parts.footerRight.textContent=auto?String(position+1).padStart(2,'0')+' / '+String(total).padStart(2,'0'):value(page,'slideFooterRight','');
 parts.scene.style.setProperty('--sd-label-header-size',clamp(page.slideHeaderSize,10,35,15)+'px');
 parts.scene.style.setProperty('--sd-label-footer-size',clamp(page.slideFooterSize,9,30,13)+'px');
 parts.footerRight.hidden=!auto&&parts.footerRight.textContent==='';
 parts.headerLeft.hidden=parts.headerLeft.textContent==='';
 parts.headerRight.hidden=parts.headerRight.textContent==='';
 parts.footerLeft.hidden=parts.footerLeft.textContent==='';
}
function build(editor,cfg){
 const anchor=editor.querySelector('#sdEditLayout')?.closest('.sd-field');
 if(!anchor)return {refresh(){}};
 const root=document.createElement('section');root.className='sd-margins';root.setAttribute('aria-label','Modifica intestazioni e piè di pagina');
 root.innerHTML=`
 <div class="sd-margins-head"><strong>INTESTAZIONI E PIÈ DI PAGINA</strong><p>Personalizza le quattro scritte della slide. Lascia un campo vuoto per nasconderlo.</p></div>
 <div class="sd-margins-grid">
  <div class="sd-margins-group"><strong>IN ALTO</strong>
   <label for="sdTopLeft">Intestazione a sinistra</label><input id="sdTopLeft" type="text" maxlength="150" placeholder="Testo in alto a sinistra">
   <label for="sdTopRight">Intestazione a destra</label><input id="sdTopRight" type="text" maxlength="100" placeholder="Testo in alto a destra">
   <label for="sdHeaderSize">Dimensione intestazioni <output id="sdHeaderPx">15 px</output></label><input id="sdHeaderSize" type="range" min="10" max="35" step="1" value="15">
  </div>
  <div class="sd-margins-group"><strong>IN BASSO</strong>
   <label for="sdBottomLeft">Piè di pagina a sinistra</label><input id="sdBottomLeft" type="text" maxlength="170" placeholder="Testo in basso a sinistra">
   <label class="sd-margins-checkbox" for="sdNumberEnabled"><input id="sdNumberEnabled" type="checkbox" checked><span>Mostra il numero automatico della slide</span></label>
   <label for="sdBottomRight">Piè di pagina a destra (in alternativa al numero)</label><input id="sdBottomRight" type="text" maxlength="100" placeholder="Testo in basso a destra" disabled>
   <label for="sdFooterSize">Dimensione piè di pagina <output id="sdFooterPx">13 px</output></label><input id="sdFooterSize" type="range" min="9" max="30" step="1" value="13">
  </div>
 </div>
 <div class="sd-margins-actions"><p>Di norma, le modifiche riguardano soltanto questa slide.</p>
  <button type="button" id="sdMarginsRestore" class="sd-editbtn">RIPRISTINA SCRITTE ORIGINALI</button>
  <button type="button" id="sdMarginsCopy" class="sd-editbtn sd-margins-copy">APPLICA A TUTTE LE SLIDE</button>
 </div>`;
 anchor.before(root);
 const q=id=>root.querySelector('#'+id);
 const inputs={slideHeaderLeft:q('sdTopLeft'),slideHeaderRight:q('sdTopRight'),slideFooterLeft:q('sdBottomLeft'),slideFooterRight:q('sdBottomRight'),slideHeaderSize:q('sdHeaderSize'),slideFooterSize:q('sdFooterSize'),slideFooterPageNumber:q('sdNumberEnabled')};
 let refreshing=false;
 function reflect(){
  const useNumber=inputs.slideFooterPageNumber.checked;
  inputs.slideFooterRight.disabled=useNumber;
  inputs.slideFooterRight.setAttribute('aria-disabled',String(useNumber));
  q('sdHeaderPx').textContent=inputs.slideHeaderSize.value+' px';
  q('sdFooterPx').textContent=inputs.slideFooterSize.value+' px';
 }
 function refresh(){
  const p=cfg.getPage();if(!p)return;
  refreshing=true;
  const base=defaults(p);
  for(const [key,input] of Object.entries(inputs)){
   if(key==='slideFooterPageNumber'){input.checked=p.slideFooterPageNumber!==false;continue}
   if(key==='slideHeaderSize'){input.value=clamp(p.slideHeaderSize,10,35,15);continue}
   if(key==='slideFooterSize'){input.value=clamp(p.slideFooterSize,9,30,13);continue}
   input.value=value(p,key,base[key]);
  }
  reflect();refreshing=false;
 }
 function change(key){
  if(refreshing)return;
  const p=cfg.getPage();if(!p)return;
  const input=inputs[key];
  p[key]=key==='slideFooterPageNumber'?input.checked:key==='slideHeaderSize'||key==='slideFooterSize'?Number(input.value):input.value;
  reflect();cfg.changed();cfg.draw();
 }
 for(const [key,input] of Object.entries(inputs)){
  input.addEventListener((key==='slideFooterPageNumber'?'change':key==='slideHeaderSize'||key==='slideFooterSize'?'input':'input'),()=>change(key));
 }
 q('sdMarginsRestore').addEventListener('click',()=>{
  const p=cfg.getPage();if(!p)return;
  for(const name of fields)delete p[name];
  refresh();cfg.changed();cfg.draw();
 });
 q('sdMarginsCopy').addEventListener('click',()=>{
  const p=cfg.getPage(),pages=cfg.getPages();if(!p||!pages.length)return;
  if(!confirm('Copiare le stesse intestazioni e piè di pagina su tutte le '+pages.length+' slide? I numeri resteranno automatici se attivati.'))return;
  const base=defaults(p);
  const data=Object.fromEntries(fields.map(key=>[key,has(p,key)?p[key]:base[key]]));
  for(const slide of pages)Object.assign(slide,data);
  cfg.changed();cfg.draw();
 });
 refresh();
 return{refresh};
}
window.SirPresentaMargins={build,apply};
})();