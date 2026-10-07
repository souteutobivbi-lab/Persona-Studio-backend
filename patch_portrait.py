"""
patch_portrait.py
Converts /generate-portrait from a synchronous blocking call to an async
background-job pattern (fire-and-forget thread + polling), matching how
the video pipeline already works.

Run from the project root:
    python patch_portrait.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).parent

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  PATCH main.py
# ═══════════════════════════════════════════════════════════════════════════════
main_py = ROOT / "main.py"
content = main_py.read_text(encoding="utf-8")

OLD_START = '@app.post("/generate-portrait")'
OLD_END   = '@app.get("/portrait/{jid}/{idx}")'

assert OLD_START in content, "ERROR: could not find @app.post('/generate-portrait') in main.py"
assert OLD_END   in content, "ERROR: could not find @app.get('/portrait/{jid}/{idx}') in main.py"

start = content.index(OLD_START)
end   = content.index(OLD_END)

NEW_HANDLER = '''\
@app.post("/generate-portrait")
async def generate_portrait(
    persona_name: str = Form(...),
    persona_age: str = Form(...),
    appearance: str = Form(...),
    outfit: str = Form("elegant fitted dress"),
    outfit_color: str = Form(""),
    count: int = Form(1),
    persona_id: str = Form(""),
    master_url: str = Form("")
):
    """Generate podcast-style portrait.
    Returns {job_id, status:"running"} immediately.
    Poll /portrait-status/{jid} until status=="complete".
    """
    import fal_client
    os.environ["FAL_KEY"] = FAL_KEY
    jid = str(uuid.uuid4())[:8]

    # Resolve master portrait URL — must be a public https URL for fal
    ref_url = master_url.strip()
    if ref_url and not ref_url.startswith("http"):
        ref_url = ""
    if not ref_url and persona_id:
        entry = MASTER_PORTRAITS.get(persona_id)
        if entry:
            ref_url = entry["url"]
    if not ref_url and persona_id:
        for key, entry in MASTER_PORTRAITS.items():
            if key.lower() == persona_id.lower():
                ref_url = entry["url"]
                break

    prompt = build_portrait_prompt(appearance, persona_age, outfit, outfit_color)
    set_job(jid, {"status": "running", "progress": 0, "stage": "generating"})

    def _run():
        try:
            import fal_client as _fal
            import requests as _req
            _imgs   = []
            _scores = []
            _method = "flux"
            n = min(count, 4)

            for i in range(n):
                img_path = OUTPUT_DIR / f"{jid}_portrait_{i}.png"
                set_job(jid, {
                    "status": "running",
                    "stage": f"generating_{i+1}_of_{n}",
                    "progress": int((i / n) * 80)
                })

                if ref_url:
                    print(f"[{jid}] instant-character {i} master: {ref_url[:60]}...")
                    try:
                        result = _fal.subscribe("fal-ai/instant-character", arguments={
                            "prompt": prompt,
                            "image_url": ref_url,
                            "scale": 0.8,
                            "guidance_scale": 3.5,
                            "num_inference_steps": 28,
                            "image_size": "portrait_4_3",
                            "num_images": 1
                        })
                        img_url = result["images"][0]["url"]
                        _method = "instant-character"
                    except Exception as ic_err:
                        print(f"[{jid}] instant-character failed ({ic_err}), falling back to FLUX Dev")
                        result = _fal.subscribe("fal-ai/flux/dev", arguments={
                            "prompt": prompt,
                            "image_size": "portrait_4_3",
                            "num_inference_steps": 28,
                            "guidance_scale": 3.5,
                            "num_images": 1,
                            "enable_safety_checker": False
                        })
                        img_url = result["images"][0]["url"]
                        _method = "flux-fallback"
                else:
                    print(f"[{jid}] FLUX Dev {i} (no master)")
                    result = _fal.subscribe("fal-ai/flux/dev", arguments={
                        "prompt": prompt,
                        "image_size": "portrait_4_3",
                        "num_inference_steps": 28,
                        "guidance_scale": 3.5,
                        "num_images": 1,
                        "enable_safety_checker": False
                    })
                    img_url = result["images"][0]["url"]
                    _method = "flux"

                r = _req.get(img_url, timeout=60)
                img_path.write_bytes(r.content)

                score = -1.0
                if ref_url:
                    master_local = MASTER_PORTRAITS.get(persona_id, {}).get("local")
                    if master_local and Path(master_local).exists():
                        score = get_face_similarity(str(img_path), master_local)

                _imgs.append(f"/portrait/{jid}/{i}")
                _scores.append(round(score, 3) if score >= 0 else None)

            set_job(jid, {
                "status":           "complete",
                "progress":         100,
                "stage":            "done",
                "images":           _imgs,
                "identity_scores":  _scores,
                "method":           _method,
                "master_used":      bool(ref_url)
            })
            print(f"[{jid}] Portrait generation complete ({_method})")

        except Exception as e:
            import traceback; traceback.print_exc()
            set_job(jid, {"status": "error", "error": str(e), "progress": 0})

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": jid, "status": "running"}


@app.get("/portrait-status/{jid}")
async def portrait_status(jid: str):
    return get_job(jid)


'''

content = content[:start] + NEW_HANDLER + content[end:]
main_py.write_text(content, encoding="utf-8")
print("✓ main.py — generate_portrait converted to async background job")
print(f"  /portrait-status/{{jid}} endpoint added")

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  PATCH index.html
# ═══════════════════════════════════════════════════════════════════════════════
html_path = ROOT / "static" / "index.html"
html = html_path.read_text(encoding="utf-8")

# ── 2a. Add the polling helper (insert before the LAST </script> tag) ─────────
POLL_HELPER = """
// ── Portrait job polling helper ──────────────────────────────────────────────
// POSTs to /generate-portrait, gets job_id, polls /portrait-status/{jid}
// every 3 s until complete. Returns the completed job object (has .images[]).
async function generatePortraitAndPoll(fd) {
  const r = await fetch(API()+'/generate-portrait', {method:'POST', body:fd});
  if(!r.ok) throw new Error('Portrait request failed: '+r.status);
  const init = await r.json();
  if(init.error) throw new Error(init.error);
  const jid = init.job_id;
  // Poll every 3 s
  for(let attempts = 0; attempts < 40; attempts++) {
    await new Promise(res => setTimeout(res, 3000));
    const sr = await fetch(API()+'/portrait-status/'+jid);
    const s  = await sr.json();
    if(s.status === 'complete') return s;
    if(s.status === 'error')   throw new Error(s.error || 'Portrait generation failed');
  }
  throw new Error('Portrait generation timed out after 2 minutes');
}
// ─────────────────────────────────────────────────────────────────────────────
"""

# Insert before last </script>
last_script = html.rfind('</script>')
assert last_script != -1, "ERROR: </script> not found in index.html"
html = html[:last_script] + POLL_HELPER + html[last_script:]
print("✓ index.html — generatePortraitAndPoll() helper added")

# ── 2b. Replace all 7 call-site patterns ─────────────────────────────────────
# We handle three distinct formatting variants found across the file.

replacements = [
    # Variant A — backtick URL, two separate lines, tight spacing (5 occurrences)
    (
        r"const r=await fetch\(`\$\{API\(\)\}/generate-portrait`,\{method:'POST',body:fd\}\);\n(\s*)const d=await r\.json\(\);\n\s*if\(d\.error\)\s*throw new Error\(d\.error\);",
        r"const d=await generatePortraitAndPoll(fd);"
    ),
    # Variant B — backtick URL + json on SAME line, error check on next (2 occurrences)
    (
        r"const r=await fetch\(`\$\{API\(\)\}/generate-portrait`,\{method:'POST',body:fd\}\);const d=await r\.json\(\);\n(\s*)if\(d\.error\)\s*throw new Error\(d\.error\);",
        r"const d=await generatePortraitAndPoll(fd);"
    ),
    # Variant C — API()+ string concat with spaces (1 occurrence — Quick Portrait tab)
    (
        r"const r = await fetch\(API\(\)\+'/generate-portrait',\{method:'POST',body:fd\}\);\n(\s*)const d = await r\.json\(\);\n\s*if\(d\.error\) throw new Error\(d\.error\);",
        r"const d = await generatePortraitAndPoll(fd);"
    ),
]

total = 0
for pattern, repl in replacements:
    new_html, n = re.subn(pattern, repl, html)
    if n:
        print(f"  replaced {n} occurrence(s) of pattern: {pattern[:60]}...")
        total += n
        html = new_html

print(f"✓ index.html — {total} generate-portrait call site(s) converted to poll pattern")
if total < 7:
    print(f"  ⚠  Expected ~7 replacements but got {total}.")
    print("    Check the file manually for any remaining raw fetch('/generate-portrait') calls.")

html_path.write_text(html, encoding="utf-8")

print()
print("All done. Commit and push to deploy:")
print("  git add main.py static/index.html")
print('  git commit -m "fix: portrait generation uses async job polling (fixes Railway upstream timeout)"')
print("  git push")
