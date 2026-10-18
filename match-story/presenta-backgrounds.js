(function backgroundModule(){
'use strict';
const colors=['#ffffff','#000000','#102a41','#f6efe0','#fff8c4','#dce9f6'];
const isHex=s=>/^#[0-9a-f]{6}$/i.test(String(s||''));
const hex=(s,f)=>isHex(s)?s:f;
const clamp=(s,a,b,f)=>s!==null&&s!==undefined&&Number.isFinite(Number(s))?Math.max(a,Math.min(b,Number(s))):f;
const modes=['standard','color','gradient','image'],tones=['dark','light','none'],fits=['cover','contain','auto'];
const imageAllowed=s=>/^https:\/\/rwfxsbxxykocdcqphfvb\.supabase\.co\/storage\/v1\/object\/public\/match-story-presenta-images\/[a-z0-9/_\-.]+$/i.test(String(s||''));
const fields=['bgMode','bgColor','bgColor2','bgImageUrl','bgImageName','bgOpacity','bgOverlay','bgOverlayAlpha','bgFit'];
const kind=p=>modes.includes(p?.bgMode)?p.bgMode:'standard';
function brightness(c){
const x=hex(c,'#102a41').slice(1);
const v=[0,2,4].map(i=>parseInt(x.slice(i,i+2),16)/255).map(n=>n<=.04045?n/12.92:Math.pow((n+.055)/1.055,2.4));
return .2126*v[0]+.7152*v[1]+.0722*v[2];
}
function apply(scene,p){
const type=kind(p);if(type==='standard')return;
scene.classList.add('sd-custom-bg');
const c1=hex(p.bgColor,'#102a41'),c2=hex(p.bgColor2,'#274b6e');
let light=false,bg='#102a41',image=null,overlay=null;
if(type==='color'){bg=c1;light=brightness(c1)>.36}
else if(type==='gradient'){bg='linear-gradient(135deg,'+c1+','+c2+')';light=brightness(c1)>.40&&brightness(c2)>.40}
else if(type==='image'){
const tone=tones.includes(p.bgOverlay)?p.bgOverlay:'dark';light=tone==='light';
if(imageAllowed(p.bgImageUrl)){
image=document.createElement('div');image.className='sd-bg-image';image.style.backgroundImage='url("'+p.bgImageUrl+'")';
image.style.backgroundSize=fits.includes(p.bgFit)?p.bgFit:'cover';image.style.opacity=String(clamp(p.bgOpacity,0,100,100)/100);
}
if(tone!=='none'){
overlay=document.createElement('div');overlay.className='sd-bg-overlay';
overlay.style.background=tone==='light'?'#fff':'#04101c';
overlay.style.opacity=String(clamp(p.bgOverlayAlpha,0,85,40)/100);
}
}
scene.style.background=bg;scene.classList.toggle('sd-light',light);
if(overlay)scene.prepend(overlay);
if(image)scene.prepend(image);
}
function build(editor,cfg){
const anchor=editor.querySelector('#sdEditLayout')?.closest('.sd-field');
if(!anchor)return{refresh(){}};
const section=document.createElement('section');section.className='sd-bg-controls';
section.innerHTML=`
<header class="sd-bg-head"><strong>SFONDO DELLA SLIDE</strong><p>Personalizza la pagina senza modificare il rapportino originale.</p></header>
<div class="sd-bg-row"><label for="sdBgMode">Tipo di sfondo</label><select id="sdBgMode"><option value="standard">Standard · come adesso</option><option value="color">Colore pieno</option><option value="gradient">Sfumatura</option><option value="image">Immagine</option></select></div>
<div id="sdBgColors" hidden>
 <div class="sd-bg-row"><label for="sdBgColor">Colore principale</label><input id="sdBgColor" type="color" value="#102a41"></div>
 <div class="sd-bg-presets" id="sdBgPresets"></div>
 <div class="sd-bg-row" id="sdBgGradient" hidden><label for="sdBgColor2">Secondo colore</label><input id="sdBgColor2" type="color" value="#274b6e"></div>
</div>
<div id="sdBgImageGroup" hidden>
 <div class="sd-bg-actions"><button type="button" class="sd-editbtn sd-upload-btn" id="sdBgUpload">+ CARICA IMMAGINE DI SFONDO</button><input type="file" id="sdBgFile" accept="image/*,.heic,.heif" style="position:absolute;width:1px;height:1px;opacity:0;pointer-events:none" aria-label="Scegli sfondo"><button type="button" class="sd-editbtn" id="sdBgReuse">Usa immagine della slide</button></div>
 <p id="sdBgStatus" role="status" aria-live="polite">Scegli una foto dall'iPad o dal computer.</p>
 <div class="sd-bg-preview" id="sdBgPreview"></div>
 <div class="sd-bg-row"><label for="sdBgFit">Adatta immagine</label><select id="sdBgFit"><option value="cover">Copri la pagina</option><option value="contain">Mostra tutta</option><option value="auto">Centrata, dimensione originale</option></select></div>
 <div class="sd-bg-row"><label for="sdBgOpacity">Opacità immagine <output id="sdBgOpacityValue">100%</output></label><input type="range" min="0" max="100" step="5" id="sdBgOpacity"></div>
 <div class="sd-bg-row"><label for="sdBgOverlay">Velo per il testo</label><select id="sdBgOverlay"><option value="dark">Scuro</option><option value="light">Chiaro</option><option value="none">Nessuno</option></select></div>
 <div class="sd-bg-row" id="sdBgOverlayGroup"><label for="sdBgOverlayAlpha">Intensità velo <output id="sdBgOverlayValue">40%</output></label><input type="range" min="0" max="85" step="5" id="sdBgOverlayAlpha"></div>
</div>
<div class="sd-bg-all"><p><b>Solo questa slide:</b> le regolazioni sopra valgono per la pagina corrente.</p><button type="button" class="sd-editbtn sd-bg-copy" id="sdBgAll">APPLICA LO SFONDO A TUTTE LE SLIDE</button></div>`;
anchor.before(section);
const q=id=>section.querySelector('#'+id),inputs={bgMode:q('sdBgMode'),bgColor:q('sdBgColor'),bgColor2:q('sdBgColor2'),bgFit:q('sdBgFit'),bgOpacity:q('sdBgOpacity'),bgOverlay:q('sdBgOverlay'),bgOverlayAlpha:q('sdBgOverlayAlpha')},status=q('sdBgStatus');
function reflect(){
const mode=inputs.bgMode.value;
q('sdBgColors').hidden=mode!=='color'&&mode!=='gradient';
q('sdBgGradient').hidden=mode!=='gradient';
q('sdBgImageGroup').hidden=mode!=='image';
q('sdBgOverlayGroup').hidden=inputs.bgOverlay.value==='none';
q('sdBgOpacityValue').textContent=inputs.bgOpacity.value+'%';
q('sdBgOverlayValue').textContent=inputs.bgOverlayAlpha.value+'%';
const p=cfg.getPage(),url=imageAllowed(p?.bgImageUrl)?p.bgImageUrl:null,prev=q('sdBgPreview');
prev.style.backgroundImage=url?'url("'+url+'")':'none';prev.style.backgroundSize=inputs.bgFit.value;
prev.textContent=url?'':'Nessuna foto di sfondo. Caricane una qui sopra.';
}
function refresh(){
const p=cfg.getPage();if(!p)return;
inputs.bgMode.value=kind(p);
inputs.bgColor.value=hex(p.bgColor,'#102a41');inputs.bgColor2.value=hex(p.bgColor2,'#274b6e');
inputs.bgFit.value=fits.includes(p.bgFit)?p.bgFit:'cover';
inputs.bgOpacity.value=clamp(p.bgOpacity,0,100,100);
inputs.bgOverlay.value=tones.includes(p.bgOverlay)?p.bgOverlay:'dark';
inputs.bgOverlayAlpha.value=clamp(p.bgOverlayAlpha,0,85,40);
reflect();
}
function update(field,value){const p=cfg.getPage();if(!p)return;p[field]=value;cfg.changed();reflect();cfg.draw()}
for(const [field,input] of Object.entries(inputs)){
const event=['bgColor','bgColor2','bgOpacity','bgOverlayAlpha'].includes(field)?'input':'change';
input.addEventListener(event,()=>update(field,['bgOpacity','bgOverlayAlpha'].includes(field)?Number(input.value):input.value));
}
const presets=q('sdBgPresets');
for(const [idx,c] of colors.entries()){
const btn=document.createElement('button');btn.type='button';btn.title=['Bianco','Nero','Blu scuro','Beige','Giallo chiaro','Celeste'][idx];btn.setAttribute('aria-label',btn.title);btn.style.background=c;
btn.addEventListener('click',()=>{const p=cfg.getPage();if(!p)return;p.bgMode='color';p.bgColor=c;cfg.changed();refresh();cfg.draw()});
presets.append(btn);
}
q('sdBgAll').addEventListener('click',()=>{
const p=cfg.getPage(),pages=cfg.getPages();if(!p||!pages?.length)return;
if(!confirm('Applicare lo sfondo a tutte le '+pages.length+' slide? I testi e le immagini rimarranno invariati.'))return;
const data=Object.fromEntries(fields.map(f=>[f,p[f]??null]));
pages.forEach(page=>Object.assign(page,data));
cfg.changed();cfg.draw();status.textContent='Sfondo applicato a tutte le slide.';
});
const file=q('sdBgFile'),button=q('sdBgUpload');
button.addEventListener('click',()=>file.click());
file.addEventListener('change',async()=>{
const image=file.files?.[0];file.value='';if(!image)return;
const dest=cfg.getPage();if(!dest)return;
button.disabled=true;
try{
const uploader=window.SirPresentaPhotos;if(!uploader)throw Error('Caricamento immagini non disponibile.');
const uploaded=await uploader.upload(cfg.reportId,image,msg=>{status.textContent=msg});
if(!cfg.getPages().includes(dest))throw Error('Questa slide è stata eliminata.');
dest.bgImageUrl=uploaded.src;dest.bgImageName=uploaded.name;dest.bgMode='image';
if(dest.bgOpacity==null)dest.bgOpacity=100;
if(dest.bgOverlay==null)dest.bgOverlay='dark';
if(dest.bgOverlayAlpha==null)dest.bgOverlayAlpha=40;
cfg.changed();if(dest===cfg.getPage()){refresh();cfg.draw()}
const ok=await cfg.save();
status.textContent=ok?'✓ Immagine caricata e salvata.':'Immagine caricata; riprova a salvare la slide.';
}catch(e){status.textContent='⚠ '+(e.message||'Caricamento non riuscito.')}
finally{button.disabled=false}
});
q('sdBgReuse').addEventListener('click',()=>{
const p=cfg.getPage();if(!p)return;const img=cfg.getImage(p);
if(!img||!imageAllowed(img.src)){status.textContent='Per usare questa foto come sfondo, caricala con il pulsante qui sopra.';return}
p.bgImageUrl=img.src;p.bgImageName='Immagine della slide';p.bgMode='image';
cfg.changed();refresh();cfg.draw();
});
refresh();
return{refresh};
}
window.SirPresentaBackgrounds={build,apply};
})();