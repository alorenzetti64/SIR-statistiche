/* PRESENTA rich editor — formatting is scoped to selected text; source chapters stay untouched. */
(function(){
'use strict';
const allowed=new Set(['DIV','P','BR','SPAN','B','STRONG','I','EM','U','UL','OL','LI']);
const forbidden=new Set(['SCRIPT','STYLE','SVG','IFRAME','OBJECT','EMBED','FORM','INPUT','TEXTAREA','BUTTON','IMG','VIDEO','AUDIO']);
const safeFonts=new Set(['Arial','Georgia','Verdana','Trebuchet MS','Times New Roman','Inter','Helvetica','Segoe UI']);
function sanitize(html){
 const source=document.createElement('template');source.innerHTML=String(html||'').slice(0,35000);
 const out=document.createElement('div');
 function walk(node,host){
  if(node.nodeType===Node.TEXT_NODE){host.append(document.createTextNode(node.textContent||''));return}
  if(node.nodeType!==Node.ELEMENT_NODE)return;
  if(forbidden.has(node.tagName))return;
  const tag=allowed.has(node.tagName)?node.tagName.toLowerCase():'span';
  const el=document.createElement(tag);
  if(tag==='span'||tag==='div'||tag==='p'){
   const st=node.style;
   const size=Number.parseFloat(st.fontSize||'');
   if(/^[0-9.]+px$/i.test(st.fontSize||'')&&Number.isFinite(size)&&size>=12&&size<=140)el.style.fontSize=size+'px';
   const color=st.color||'';
   if(color&&(/^#[0-9a-f]{3,8}$/i.test(color)||/^rgb(a)?\([\d\s.,%]+\)$/i.test(color)))el.style.color=color;
   const fm=(st.fontFamily||'').replace(/["']/g,'').split(',')[0].trim();
   if(safeFonts.has(fm))el.style.fontFamily='"'+fm+'"';
   if(['left','center','right','justify'].includes(st.textAlign))el.style.textAlign=st.textAlign;
   if(['bold','700','800','900'].includes(st.fontWeight))el.style.fontWeight='bold';
   if(st.fontStyle==='italic')el.style.fontStyle='italic';
   if(st.textDecorationLine==='underline')el.style.textDecoration='underline';
  }
  if(tag!=='br')for(const ch of node.childNodes)walk(ch,el);
  host.append(el);
 }
 for(const ch of source.content.childNodes)walk(ch,out);
 return out.innerHTML;
}
function plainFromEditor(editor){
 const v=editor.innerText;
 return String(v||'').replace(/\r/g,'').replace(/\u00a0/g,' ').replace(/\n{4,}/g,'\n\n\n').trimEnd().slice(0,2000);
}
function fromPlain(value){
 const root=document.createElement('div');root.textContent=String(value||'');return root.innerHTML.replace(/\n/g,'<br>');
}
function restoreSelection(editor,range){
 if(!range)return false;
 try{
  if(!editor.contains(range.commonAncestorContainer))return false;
  const sel=window.getSelection();sel.removeAllRanges();sel.addRange(range);editor.focus({preventScroll:true});return true;
 }catch(e){return false}
}
function applySpan(editor,range,property,value){
 if(!range||range.collapsed||!editor.contains(range.commonAncestorContainer))return false;
 const span=document.createElement('span');span.style[property]=value;
 try{
  const fragment=range.extractContents();span.append(fragment);range.insertNode(span);
  const selection=window.getSelection(),reselect=document.createRange();reselect.selectNodeContents(span);
  selection.removeAllRanges();selection.addRange(reselect);
  editor.dispatchEvent(new Event('input',{bubbles:true}));return true;
 }catch(e){return false}
}
function formatCommand(editor,range,cmd){
 if(!restoreSelection(editor,range))return false;
 try{
  const ok=document.execCommand(cmd,false,null);
  if(ok)editor.dispatchEvent(new Event('input',{bubbles:true}));
  return !!ok;
 }catch(e){return false}
}
window.SirPresentaRichText={sanitize,plainFromEditor,fromPlain,restoreSelection,applySpan,formatCommand};
})();