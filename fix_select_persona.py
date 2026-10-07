content = open('static/index.html', encoding='utf-8').read()

old = '  if(p) selectPersona(p);\n  S.portraitUrl = url;'
new = '  if(p) { S.persona = p; }\n  S.portraitUrl = url;'

print('Found:', old in content)
if old in content:
    content = content.replace(old, new, 1)
    open('static/index.html', 'w', encoding='utf-8').write(content)
    print('Done')
else:
    idx = content.find('selectPersona')
    print(repr(content[idx-100:idx+100]))
