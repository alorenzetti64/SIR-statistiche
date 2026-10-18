/* PRESENTA — image upload from iPad and computer.
 * Store durable image URL in the slide; do not change the original chapter. */
(function(){
'use strict';
const api='https://rwfxsbxxykocdcqphfvb.supabase.co/functions/v1/match-story-director';
const error=s=>new Error(s);
async function prepare(file,progress=()=>{}){
 if(!file||!(/^image\//i.test(file.type)||/\.(heic|heif)$/i.test(file.name||'')))
  throw error('Scegli una foto o un file immagine.');
 if(file.size>35*1024*1024)throw error('La foto supera 35 MB. Usane una più piccola.');
 const link=URL.createObjectURL(file);
 let picture;
 try{
  picture=await new Promise((resolve,reject)=>{
   const img=new Image();
   img.onload=()=>resolve(img);
   img.onerror=()=>reject(error('Non posso aprire questo formato su questo dispositivo. Prova con JPG o PNG.'));
   img.src=link;
  });
  const w=picture.naturalWidth||picture.width,h=picture.naturalHeight||picture.height;
  if(!w||!h)throw error('Questa immagine non è leggibile.');
  const canvas=document.createElement('canvas');
  const ratio=Math.min(1,2200/Math.max(w,h));
  let scale=ratio,quality=.83;
  for(let attempt=0;attempt<3;attempt++){
   canvas.width=Math.max(1,Math.round(w*scale));
   canvas.height=Math.max(1,Math.round(h*scale));
   const ctx=canvas.getContext('2d');
   if(!ctx)throw error('Il dispositivo non può elaborare questa immagine.');
   ctx.fillStyle='#ffffff';
   ctx.fillRect(0,0,canvas.width,canvas.height);
   ctx.drawImage(picture,0,0,canvas.width,canvas.height);
   const result=await new Promise((resolve,reject)=>{
    canvas.toBlob(blob=>blob?resolve(blob):reject(error('Conversione immagine non riuscita.')),'image/jpeg',quality);
   });
   if(result.size<=4_800_000)return result;
   progress('Ottimizzo una foto molto grande…');
   scale*=.78;quality-=.09;
  }
  throw error('Foto troppo grande anche dopo la riduzione. Prova con una più piccola.');
 }finally{URL.revokeObjectURL(link)}
}
async function upload(reportId,file,progress=()=>{}){
 progress('Preparo la foto per la slide…');
 const blob=await prepare(file,progress);
 progress('Carico e salvo la foto…');
 const response=await fetch(api+'?action=upload-image&id='+encodeURIComponent(reportId),{
  method:'POST',headers:{'Content-Type':'image/jpeg'},body:blob,cache:'no-store'
 });
 let reply;
 try{reply=await response.json()}catch(e){throw error('Il server non ha restituito una risposta valida.')}
 if(!response.ok||!reply?.ok||!/^https:\/\/rwfxsbxxykocdcqphfvb\.supabase\.co\/storage\/v1\/object\/public\/match-story-presenta-images\//.test(String(reply?.url||'')))
  throw error(reply?.error||'Non è stato possibile salvare la foto.');
 return{src:reply.url,name:String(file.name||'Immagine').slice(0,100)};
}
window.SirPresentaPhotos={upload};
})();