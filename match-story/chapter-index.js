/* Indice condiviso da PLAYERS, Anteprima e Presenta. */
function sirMakeChapterIndex({doc,host,chapters,onChoose,targets=null,lang='it'}) {
  if(!chapters.length)return null;
  const labels={
    it:{button:'INDICE',heading:'INDICE DEI CAPITOLI',close:'Chiudi indice',missing:'Nessuna slide per questo capitolo'},
    en:{button:'CONTENTS',heading:'CHAPTER INDEX',close:'Close index',missing:'No slides for this chapter'},
    fa:{button:'فهرست',heading:'فهرست فصل‌ها',close:'بستن فهرست',missing:'اسلایدی برای این فصل وجود ندارد'}
  }[lang]||{button:'INDICE',heading:'INDICE DEI CAPITOLI',close:'Chiudi indice',missing:'Nessuna slide per questo capitolo'};
  const menu=doc.createElement('nav');
  menu.className='sir-chapter-index';
  menu.setAttribute('aria-label',labels.heading);
  const trigger=doc.createElement('button');
  trigger.type='button';trigger.className='sir-index-trigger';trigger.textContent='☷  '+labels.button;
  trigger.setAttribute('aria-haspopup','true');trigger.setAttribute('aria-expanded','false');
  const panel=doc.createElement('div');
  panel.className='sir-index-panel';panel.hidden=true;
  const uid='sir-chapter-index-'+Math.random().toString(36).slice(2);
  panel.id=uid;trigger.setAttribute('aria-controls',uid);
  const header=doc.createElement('div');
  header.className='sir-index-heading';
  const heading=doc.createElement('span');heading.textContent=labels.heading;header.append(heading);
  const closeButton=doc.createElement('button');closeButton.type='button';
  closeButton.textContent='×';closeButton.setAttribute('aria-label',labels.close);
  header.append(closeButton);panel.append(header);
  const list=doc.createElement('div');list.className='sir-index-list';
  const buttons=chapters.map((chapter,i)=>{
    const button=doc.createElement('button');button.type='button';
    const number=doc.createElement('span');number.className='sir-index-number';number.textContent=String(i+1);
    const title=doc.createElement('span');title.className='sir-index-text';title.textContent=chapter.title||String(i+1);
    const check=doc.createElement('span');check.className='sir-index-check';check.textContent='✓';check.setAttribute('aria-hidden','true');
    button.append(number,title,check);button.disabled=!!chapter.disabled;
    if(button.disabled)button.title=labels.missing;
    button.addEventListener('click',()=>{
      if(onChoose(i)===false)return;
      setActive(i);close();trigger.focus({preventScroll:true});
    });
    list.append(button);return button;
  });
  panel.append(list);menu.append(trigger,panel);host.append(menu);
  let activeIndex=0,scrollListener=null,rafPending=false;
  function setActive(i) {
    if(i<0||i>=buttons.length)return;
    activeIndex=i;
    buttons.forEach((button,n)=>{
      if(n===i)button.setAttribute('aria-current','location');
      else button.removeAttribute('aria-current');
    });
  }
  function close(){panel.hidden=true;trigger.setAttribute('aria-expanded','false')}
  function open(){panel.hidden=false;trigger.setAttribute('aria-expanded','true');(buttons[activeIndex]?.disabled?buttons.find(b=>!b.disabled):buttons[activeIndex])?.focus({preventScroll:true})}
  trigger.addEventListener('click',()=>panel.hidden?open():close());
  closeButton.addEventListener('click',()=>{close();trigger.focus({preventScroll:true})});
  const outside=event=>{if(!menu.contains(event.target))close()};
  const keydown=event=>{if(event.key==='Escape'&&!panel.hidden){event.preventDefault();close();trigger.focus({preventScroll:true})}};
  doc.addEventListener('click',outside);doc.addEventListener('keydown',keydown);
  const view=doc.defaultView;
  if(targets?.length&&view){
    const update=()=>{
      rafPending=false;
      const threshold=Math.min(175,Math.max(95,view.innerHeight*.24));
      let visible=0;
      targets.forEach((target,i)=>{if(target.getBoundingClientRect().top<=threshold)visible=i});
      setActive(visible);
    };
    scrollListener=()=>{if(!rafPending){rafPending=true;view.requestAnimationFrame(update)}};
    view.addEventListener('scroll',scrollListener,{passive:true});
    update();
  }else setActive(0);
  return {root:menu,setActive,close,isOpen:()=>!panel.hidden,destroy:()=>{
    if(scrollListener)view?.removeEventListener('scroll',scrollListener);
    doc.removeEventListener('click',outside);doc.removeEventListener('keydown',keydown);menu.remove();
  }};
}
