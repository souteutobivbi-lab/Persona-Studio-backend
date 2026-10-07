content = open('static/index.html', encoding='utf-8').read()

old = """function useQuickPortraitForVideo(){
  if(!window._qpLastImg) return;
  const {url, personaName} = window._qpLastImg;
  const p = S.personas.find(x=>x.name===personaName);
  if(p) selectPersona(p);
  S.portraitUrl = url;
  S.portraitPath = window._qpLastImg.path;
  fetch(url).then(r=>r.blob()).then(blob=>{S.portraitFile=new File([blob],'portrait.png',{type:'image/png'});});
  showPanel('create', null);
  goStep(4);"""

new = """async function useQuickPortraitForVideo(){
  if(!window._qpLastImg) return;
  const {url, personaName} = window._qpLastImg;
  const p = S.personas.find(x=>x.name===personaName);
  if(p) selectPersona(p);
  S.portraitUrl = url;
  S.portraitPath = window._qpLastImg.path;
  try {
    const blob = await fetch(url).then(r=>r.blob());
    S.portraitFile = new File([blob],'portrait.png',{type:'image/png'});
  } catch(e){ console.warn('portrait blob fetch failed', e); }
  showPanel('create', null);
  goStep(4);"""

print('Found:', old in content)
if old in content:
    content = content.replace(old, new, 1)
    open('static/index.html', 'w', encoding='utf-8').write(content)
    print('Done')
else:
    idx = content.find('useQuickPortraitForVideo')
    print(repr(content[idx:idx+400]))
