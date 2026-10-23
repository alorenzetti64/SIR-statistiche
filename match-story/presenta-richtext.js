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

function insideList(editor,range){
 if(!range||!editor.contains(range.commonAncestorContainer))return null;
 let node=range.startContainer;
 if(node.nodeType===Node.TEXT_NODE)node=node.parentElement;
 const li=node?.closest?.('li');
 if(!li||!editor.contains(li))return null;
 const list=li.parentElement;
 return list&&(list.tagName==='UL'||list.tagName==='OL')&&editor.contains(list)?li:null;
}
function cursorIn(editor){
 const selection=window.getSelection();
 if(!selection||!selection.rangeCount)return null;
 const range=selection.getRangeAt(0);
 return editor.contains(range.commonAncestorContainer)?range.cloneRange():null;
}
function setCaretAtStart(node){
 const range=document.createRange(),selection=window.getSelection();
 range.selectNodeContents(node);range.collapse(true);
 selection.removeAllRanges();selection.addRange(range);
}
// Remove one bullet without destroying its text or the surrounding list.
// The rows before/after the selected row remain independent list items.
function removeBullet(editor,range){
 const li=insideList(editor,range);
 if(!li)return false;
 const list=li.parentElement,parent=list.parentNode;
 if(!parent)return false;
 const paragraph=document.createElement('div');
 while(li.firstChild)paragraph.append(li.firstChild);
 if(!paragraph.textContent.trim()&&!paragraph.querySelector('br'))paragraph.append(document.createElement('br'));
 const afterList=list.cloneNode(false);
 while(li.nextSibling)afterList.append(li.nextSibling);
 parent.insertBefore(paragraph,list.nextSibling);
 if(afterList.childNodes.length)parent.insertBefore(afterList,paragraph.nextSibling);
 li.remove();
 if(!list.querySelector('li'))list.remove();
 editor.focus({preventScroll:true});
 setCaretAtStart(paragraph);
 editor.dispatchEvent(new Event('input',{bubbles:true}));
 return true;
}
// The list tool works with a *cursor*, not only highlighted words.
// On a bulleted line it removes that bullet; on normal text it starts a list.
function toggleBulletList(editor,range){
 const selected=range&&editor.contains(range.commonAncestorContainer)?range:cursorIn(editor);
 if(!selected)return false;
 if(selected.collapsed&&insideList(editor,selected))return removeBullet(editor,selected);
 return formatCommand(editor,selected,'insertUnorderedList');
}
// Enter or Backspace on an empty bullet returns to normal text.
// This also works with iPad's on-screen keyboard through beforeinput.
function escapeEmptyBullet(editor,range){
 if(!range||!range.collapsed)return false;
 const li=insideList(editor,range);
 if(!li)return false;
 const text=(li.innerText||li.textContent||'').replace(/\u200b/g,'').replace(/\u00a0/g,' ').trim();
 return !text&&removeBullet(editor,range);
}

window.SirPresentaRichText={sanitize,plainFromEditor,fromPlain,restoreSelection,applySpan,formatCommand,toggleBulletList,removeBullet,escapeEmptyBullet,insideList};
})();