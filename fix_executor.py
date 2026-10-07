content = open('main.py', encoding='utf-8').read()

old = ('                        import concurrent.futures as _cf\n'
       '                        with _cf.ThreadPoolExecutor(max_workers=1) as _ex:\n'
       '                            _fut = _ex.submit(_fal.subscribe, "fal-ai/instant-character", arguments={\n'
       '                                "prompt": prompt,\n'
       '                                "image_url": ref_url,\n'
       '                                "scale": 0.8,\n'
       '                                "guidance_scale": 3.5,\n'
       '                                "num_inference_steps": 28,\n'
       '                                "image_size": "portrait_4_3",\n'
       '                                "num_images": 1\n'
       '                            })\n'
       '                            result = _fut.result(timeout=45)')

new = ('                        import concurrent.futures as _cf\n'
       '                        _ex = _cf.ThreadPoolExecutor(max_workers=1)\n'
       '                        _fut = _ex.submit(_fal.subscribe, "fal-ai/instant-character", arguments={\n'
       '                            "prompt": prompt,\n'
       '                            "image_url": ref_url,\n'
       '                            "scale": 0.8,\n'
       '                            "guidance_scale": 3.5,\n'
       '                            "num_inference_steps": 28,\n'
       '                            "image_size": "portrait_4_3",\n'
       '                            "num_images": 1\n'
       '                        })\n'
       '                        _ex.shutdown(wait=False)\n'
       '                        result = _fut.result(timeout=45)')

print('Found:', old in content)
if old in content:
    content = content.replace(old, new, 1)
    open('main.py', 'w', encoding='utf-8').write(content)
    print('Done - executor fix applied')
else:
    # Show what's actually around the executor code
    idx = content.find('ThreadPoolExecutor')
    if idx >= 0:
        print('Raw bytes around ThreadPoolExecutor:')
        print(repr(content[idx-50:idx+300]))
    else:
        print('ThreadPoolExecutor not found in main.py at all')
