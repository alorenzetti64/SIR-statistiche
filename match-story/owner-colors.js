/* Colori personalizzabili per capitoli e card, con valori validati. */
(function(){
 'use strict';
 const defaults={Angelo:{page:'#fffbe8',card:'#fff7d5'},Max:{page:'#edf5ff',card:'#e2efff'}};
 const valid=value=>/^#[0-9a-f]{6}$/i.test(String(value||''))?String(value).toLowerCase():null;
 function owner(name){return /^(Angelo|Max)$/.test(String(name))?name:'Max'}
 function initial(name,type){return defaults[owner(name)][type]}
 function apply(root,name){
  if(!root)return;
  const actual=owner(name);root.dataset.sirOwner=actual;
  const meta=root.querySelector('[data-sir-chapter-color]');
  const page=valid(meta?.dataset.sirChapterColor);
  if(page)root.style.setProperty('--sir-page-color',page);
  else root.style.removeProperty('--sir-page-color');
  root.querySelectorAll('[data-sir-card-color]').forEach(card=>{
   const color=valid(card.dataset.sirCardColor);
   if(color)card.style.setProperty('--sir-custom-card-color',color);
   else card.style.removeProperty('--sir-custom-card-color');
  });
 }
 function setCard(card,value){
  if(!card)return;
  const c=valid(value);
  if(c){card.dataset.sirCardColor=c;card.style.setProperty('--sir-custom-card-color',c)}
  else{card.removeAttribute('data-sir-card-color');card.style.removeProperty('--sir-custom-card-color')}
 }
 function setChapter(editor,value){
  if(!editor)return;
  let marker=editor.querySelector('[data-sir-chapter-color]');
  const color=valid(value);
  if(color){
   if(!marker){marker=editor.ownerDocument.createElement('span');marker.hidden=true;marker.dataset.sirChapterColor=color;editor.prepend(marker)}
   marker.dataset.sirChapterColor=color;editor.style.setProperty('--sir-page-color',color);
  }else{
   marker?.remove();editor.style.removeProperty('--sir-page-color');
  }
 }
 window.SirOwnerColors={defaults,valid,owner,initial,apply,setCard,setChapter};
})();
