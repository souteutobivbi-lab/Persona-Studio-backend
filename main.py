import asyncio, os, random, subprocess, uuid, json
from pathlib import Path
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import httpx

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GROQ_KEY       = os.environ.get("GROQ_KEY", "")
ELEVENLABS_KEY = os.environ.get("ELEVENLABS_KEY", "")
FAL_KEY        = os.environ.get("FAL_KEY", "")

UPLOAD_DIR = Path("/app/uploads")
OUTPUT_DIR = Path("/app/outputs")
JOBS_FILE  = Path("/app/jobs.json")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)





import threading
def _prewarm():
    try:
        from rembg import remove
        from PIL import Image
        import numpy as np
        print("Pre-warming rembg (bria model)...")
        remove(Image.fromarray(np.zeros((10,10,3), dtype=np.uint8)))
        print("rembg ready.")
    except Exception as e:
        print(f"rembg pre-warm error: {e}")
threading.Thread(target=_prewarm, daemon=True).start()

def load_jobs():
    try:
        if JOBS_FILE.exists():
            return json.loads(JOBS_FILE.read_text())
    except: pass
    return {}

def save_jobs(jobs):
    try: JOBS_FILE.write_text(json.dumps(jobs))
    except: pass

def get_job(jid):
    return load_jobs().get(jid, {"status": "not_found"})

def set_job(jid, data):
    jobs = load_jobs()
    jobs[jid] = data
    save_jobs(jobs)

@app.get("/health")
async def health():
    return {"status": "ok", "version": "3.0", "host": "railway"}

@app.get("/")
async def root():
    p = Path(__file__).parent / "static" / "index.html"
    if p.exists():
        return HTMLResponse(p.read_text())
    return {"api": "Persona Studio v3.0"}

from fastapi.staticfiles import StaticFiles
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/assets", StaticFiles(directory=str(static_path)), name="static")

MUSIC_DIR = Path(__file__).parent / "static" / "music"
MUSIC_TRACKS = {
    "village-hearth-harp":  {"file": "village-hearth-harp.mp3",  "label": "Village Hearth — warm harp"},
    "warm-drone-ambience":  {"file": "warm-drone-ambience.mp3",   "label": "Warm Drone — meditation ambience"},
}

@app.get("/music-tracks")
async def list_music_tracks():
    return {"tracks": [{"id": k, "label": v["label"]} for k, v in MUSIC_TRACKS.items()]}

@app.post("/parse-skill")
async def parse_skill(skill_text: str = Form(...)):
    from groq import Groq
    client = Groq(api_key=GROQ_KEY)
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": f"""You are reading a persona skill file for an AI influencer platform.
Extract the following fields from the skill file text below and return them as a JSON object.

Skill file:
{skill_text[:3000]}

Extract these fields (leave empty string if not found):
- name: persona's full name
- age: persona's age as a number string
- role: their role and niche (e.g. "Transformation Authority · Wealth")
- niche: primary niche keyword (e.g. "wealth", "property", "interior design", "health", "relationships", "mindset", "luxury lifestyle")
- appearance: physical description for image generation
- catchphrase: their signature phrase
- anchors: signature fixed visual details (e.g. "pearl earrings, warm smile")
- voice_summary: 2-3 sentence summary of their voice and tone
- hook_examples: 3 example hooks as a JSON array of strings
- cta_examples: 3 example CTAs as a JSON array of strings

Return ONLY valid JSON, nothing else. No markdown, no backticks."""}],
        max_tokens=600, temperature=0.3
    )
    try:
        raw = response.choices[0].message.content.strip()
        raw = raw.replace('```json','').replace('```','').strip()
        parsed = json.loads(raw)
        return {"status": "ok", "data": parsed}
    except Exception as e:
        return JSONResponse({"error": "Parse failed: "+str(e), "raw": response.choices[0].message.content[:200]}, status_code=400)

@app.get("/voices")
async def get_voices():
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get("https://api.elevenlabs.io/v1/voices",
            headers={"xi-api-key": ELEVENLABS_KEY})
    if r.status_code != 200:
        return JSONResponse({"error": "ElevenLabs error"}, status_code=400)
    voices = r.json().get("voices", [])
    return [{"id": v["voice_id"], "name": v["name"],
             "description": ", ".join(v.get("labels", {}).values())} for v in voices]

@app.post("/generate-script")
async def generate_script(
    persona_name: str = Form(...),
    persona_age: str = Form(...),
    niche: str = Form(...),
    topic: str = Form(...),
    skill_context: Optional[str] = Form(None)
):
    from groq import Groq
    client = Groq(api_key=GROQ_KEY)

    # Detect Vivienne-style skill vs generic — look for structure markers
    has_vivienne_structure = False
    script_examples = ""
    structure_rules = ""
    if skill_context:
        sc = skill_context
        has_vivienne_structure = ("SCRIPT STRUCTURE" in sc or "podcast confession" in sc.lower()
                                   or "EXAMPLE SCRIPTS" in sc or "The Claim" in sc)
        # Extract script structure section
        if "SCRIPT STRUCTURE" in sc:
            start = sc.find("SCRIPT STRUCTURE")
            end = sc.find("━━━", start + 20)
            structure_rules = sc[start:end].strip() if end > start else sc[start:start+800]
        # Extract example scripts section
        if "EXAMPLE SCRIPTS" in sc:
            start = sc.find("EXAMPLE SCRIPTS")
            end = sc.find("━━━", start + 20)
            script_examples = sc[start:end].strip() if end > start else sc[start:start+2000]
        # Extract sign-offs
        signoffs = ""
        if "SIGN-OFFS" in sc:
            start = sc.find("SIGN-OFFS")
            end = sc.find("━━━", start + 20)
            signoffs = sc[start:end].strip() if end > start else ""

    if has_vivienne_structure:
        # Full Vivienne-spec prompt: 100-140 words, 5-part structure
        prompt = f"""You are writing 5 scripts for {persona_name}, aged {persona_age}, in the {niche} niche.
Topic: {topic}

{structure_rules}

{script_examples}

CRITICAL RULES — follow exactly:
1. Each script MUST be 100-140 words. Count carefully. Do not submit anything under 90 words.
2. Follow the 5-part structure every time:
   - Opening line: first name introduction ("I'm {persona_name}..." or "My name is {persona_name}...")
   - The Claim: one clear, specific uncomfortable truth (15-25 words)
   - The Unpacking: real mechanism explained with specificity (40-70 words)
   - The Turn: reframe that changes how the viewer sees it (15-25 words)
   - CTA: one of the sign-offs from the skill — earned, not tacked on
3. Podcast confession style — essay-like, not a listicle
4. Warm, direct, British cadence — measured, real, understated firmness
5. No hashtags, no emojis, no stage directions, no filler
6. Each of the 5 must cover a different angle of the topic with a different opening emotion
7. Never shame the viewer. Validate before educating.

{signoffs}

Return ONLY a JSON array of 5 script strings. No explanation, no markdown, no labels.
["Script one here.", "Script two here.", "Script three here.", "Script four here.", "Script five here."]"""
        max_tok = 2000
    else:
        # Generic shorter scripts for other personas
        skill_section = f"\n\nPERSONA SKILL:\n{skill_context[:2000]}\nMatch this persona's exact tone and style." if skill_context else ""
        prompt = f"""You are writing 5 short podcast-style talking head scripts for {persona_name}, aged {persona_age}, British, in the {niche} niche.
Topic: {topic}{skill_section}

STYLE: Podcast confession — emotionally honest, intimate, slightly vulnerable. A real conversation, not a sales pitch.

Rules:
- 45-80 words each (comfortable speaking pace)
- Open with a different emotional hook each time
- Warm, direct, British cadence
- End with a quiet insight or gentle challenge
- No hashtags, no emojis, no stage directions, no filler
- Each must feel like a completely different moment

Return ONLY a JSON array of 5 script strings. No explanation, no markdown, no labels.
["Script one here.", "Script two here.", "Script three here.", "Script four here.", "Script five here."]"""
        max_tok = 1200

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tok, temperature=0.88
    )
    raw = response.choices[0].message.content.strip()
    try:
        # Strip <think>...</think> reasoning blocks (Llama 3.3 / DeepSeek style)
        import re as _re
        raw = _re.sub(r'<think>.*?</think>', '', raw, flags=_re.DOTALL).strip()
        raw = raw.replace('```json','').replace('```','').strip()
        # Find JSON array even if prefixed with prose
        start = raw.find('[')
        end = raw.rfind(']')
        if start != -1 and end != -1:
            raw = raw[start:end+1]
        scripts = json.loads(raw)
        if not isinstance(scripts, list):
            scripts = [scripts]
    except:
        scripts = [raw]
    # Filter out any scripts under 40 words (malformed outputs)
    scripts = [s for s in scripts if len(s.split()) >= 40] or scripts
    return {"scripts": scripts, "script": scripts[0] if scripts else "", "words": len(scripts[0].split()) if scripts else 0}

@app.post("/generate-voice")
async def generate_voice(
    script: str = Form(...),
    voice_id: str = Form("EXAVITQu4vr4xnSDxMaL"),
    job_id: str = Form(None)
):
    jid = job_id or str(uuid.uuid4())[:8]
    out_path = OUTPUT_DIR / f"{jid}_voice.mp3"
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={"xi-api-key": ELEVENLABS_KEY, "Content-Type": "application/json"},
            json={"text": script, "model_id": "eleven_multilingual_v2",
                  "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}
        )
    if r.status_code != 200:
        return JSONResponse({"error": r.text[:200]}, status_code=400)
    out_path.write_bytes(r.content)
    return {"job_id": jid, "audio_path": str(out_path), "status": "done"}

@app.get("/audio/{job_id}")
async def serve_audio(job_id: str):
    path = OUTPUT_DIR / f"{job_id}_voice.mp3"
    if path.exists():
        return FileResponse(str(path), media_type="audio/mpeg")
    return JSONResponse({"error": "not found"}, status_code=404)

def build_portrait_prompt(appearance: str, persona_age: str, outfit: str, outfit_color: str = "") -> str:
    """Build podcast-style seated portrait prompt matching linakodi1_ / mayaa_speaks aesthetic.

    The appearance string from the persona is layered ON TOP of a hardcoded base that locks
    the podcast format, skin, body type and setting. This ensures the aesthetic never drifts
    regardless of what text is in the persona skill file.
    """
    color_hint = f"{outfit_color} " if outfit_color else ""

    # Base identity layer — locks the visual aesthetic to the reference accounts
    base_identity = (
        "deep rich warm amber-brown luminous skin, high-gloss healthy glow on cheekbones and shoulders, "
        "high cheekbones, almond-shaped dark brown eyes, long flared black lashes, "
        "full wide lips with nude-brown gloss, strong defined jawline, broad smooth forehead, "
        "West African facial bone structure, "
        "full-figured curvy body, full rounded bust, soft voluminous arms, thick legs crossed, "
        "hourglass silhouette with natural weight and softness, substantial physical presence, "
        "long voluminous natural black curls falling past shoulders, "
        "natural glam makeup, contoured cheeks, highlighted nose bridge, defined arched brows, "
        "gold medium hoop earrings, thin gold chain necklace, gold wrist bracelet"
    )

    # Persona-specific overrides (age, any unique traits from skill file)
    persona_layer = appearance if appearance else ""

    # Podcast set — fixed every time, never varies
    podcast_set = (
        "seated relaxed in a cream upholstered podcast armchair, legs crossed at knee, "
        "hands resting naturally in lap, upper body and arms fully visible in frame, "
        "large black podcast microphone on boom arm positioned in front slightly to right, "
        "background: soft warm grey textured wall, "
        "large lush tropical areca palm plant with long green fronds behind right shoulder, "
        "warm amber studio lighting, shallow depth of field, bokeh background, "
        "camera angle slightly below eye level, subject looking slightly downward toward lens"
    )

    # Technical photography layer — locks render quality
    photo_tech = (
        "Sony A7R IV 85mm f1.4 portrait lens, "
        "natural skin texture visible pores real human skin, "
        "photorealistic authentic photography not airbrushed not CGI, "
        "sharp crisp face soft blurred background, "
        "professional editorial portrait photography, high resolution"
    )

    return (
        f"photorealistic portrait photograph, {persona_age} year old British-Nigerian woman, "
        f"{base_identity}, "
        f"{persona_layer}, "
        f"wearing a {color_hint}{outfit} that fits her curves, "
        f"{podcast_set}, "
        f"{photo_tech}"
    )

def get_face_similarity(img_path_a: str, img_path_b: str) -> float:
    """Compare two face images using insightface buffalo_l. Returns cosine similarity 0.0–1.0.
    Used only as a post-generation quality check — never as a retry trigger for generation."""
    try:
        import insightface
        import numpy as np
        from PIL import Image
        _app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            root=os.environ.get("INSIGHTFACE_HOME", "/app/insightface_models"),
            providers=["CPUExecutionProvider"]
        )
        _app.prepare(ctx_id=0, det_size=(640, 640))
        def _emb(path):
            img_np = np.array(Image.open(path).convert("RGB"))
            faces = _app.get(img_np)
            if not faces:
                return None
            return sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)[0].normed_embedding
        ea, eb = _emb(img_path_a), _emb(img_path_b)
        if ea is None or eb is None:
            return -1.0  # -1 signals "no face detected" to caller
        sim = float(np.dot(ea, eb))
        print(f"Face similarity: {sim:.3f}")
        return sim
    except Exception as e:
        print(f"Face similarity error (skipping): {e}")
        return -1.0

# Persistent master portrait store — survives Railway restarts
# Saved to /app/master_portraits.json; fal CDN URLs are permanent
PORTRAITS_FILE = Path("/app/master_portraits.json")

def load_master_portraits() -> dict:
    try:
        if PORTRAITS_FILE.exists():
            data = json.loads(PORTRAITS_FILE.read_text())
            print(f"Loaded {len(data)} locked portraits from disk: {list(data.keys())}")
            return data
    except Exception as e:
        print(f"Warning: could not load master portraits: {e}")
    return {}

def save_master_portraits(portraits: dict):
    try:
        PORTRAITS_FILE.write_text(json.dumps(portraits, indent=2))
    except Exception as e:
        print(f"Warning: could not save master portraits: {e}")

MASTER_PORTRAITS: dict = load_master_portraits()  # persona_id → {"local": path, "url": fal_url}

@app.post("/set-master-portrait")
async def set_master_portrait(
    persona_id: str = Form(...),
    portrait: UploadFile = File(...)
):
    """Upload the locked master portrait for a persona.
    Stored locally AND uploaded to fal CDN for persistence across restarts.
    All future PuLID generations use this face as the identity reference."""
    import fal_client
    os.environ["FAL_KEY"] = FAL_KEY
    master_path = OUTPUT_DIR / f"master_{persona_id}.png"
    master_path.write_bytes(await portrait.read())
    # Upload to fal CDN for persistence
    fal_url = fal_client.upload_file(str(master_path))
    MASTER_PORTRAITS[persona_id] = {"local": str(master_path), "url": fal_url}
    save_master_portraits(MASTER_PORTRAITS)
    print(f"[LOCKED] Master portrait saved for '{persona_id}': {fal_url}")
    return {"status": "ok", "persona_id": persona_id, "fal_url": fal_url}

@app.post("/get-master-portrait")
async def get_master_portrait(persona_id: str = Form(...)):
    """Return the stored master portrait URL for a persona."""
    entry = MASTER_PORTRAITS.get(persona_id)
    if not entry:
        return JSONResponse({"error": "No master portrait set for this persona"}, status_code=404)
    return {"persona_id": persona_id, "fal_url": entry["url"], "has_master": True}

@app.get("/locked-personas")
async def list_locked_personas():
    """Return all personas that have a locked master portrait."""
    return {
        "locked": [
            {"persona_id": pid, "fal_url": entry["url"]}
            for pid, entry in MASTER_PORTRAITS.items()
        ],
        "count": len(MASTER_PORTRAITS)
    }

@app.delete("/locked-personas/{persona_id}")
async def delete_locked_persona(persona_id: str):
    """Remove identity lock for a persona."""
    if persona_id not in MASTER_PORTRAITS:
        return JSONResponse({"error": "Not found"}, status_code=404)
    del MASTER_PORTRAITS[persona_id]
    save_master_portraits(MASTER_PORTRAITS)
    print(f"[UNLOCKED] Removed identity lock for '{persona_id}'")
    return {"status": "ok", "persona_id": persona_id}

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
                    print(f"[{jid}] PuLID {i} master: {ref_url[:60]}...")
                    # fal-ai/pulid is the proven face-lock model (instant-character stalls)
                    try:
                        result = _fal.subscribe("fal-ai/pulid", arguments={
                            "prompt": prompt,
                            "face_image_url": ref_url,
                            "num_inference_steps": 20,
                            "style_strength": 20,
                            "num_images": 1,
                        })
                        imgs = result.get("images") or result.get("image") or []
                        if isinstance(imgs, dict):
                            imgs = [imgs]
                        img_url = imgs[0]["url"]
                        _method = "pulid"
                        print(f"[{jid}] PuLID {i} done: {img_url[:60]}")
                    except Exception as pulid_err:
                        print(f"[{jid}] PuLID failed ({pulid_err!r}), falling back to FLUX Dev")
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


@app.get("/portrait/{jid}/{idx}")
async def serve_portrait(jid: str, idx: int):
    path = OUTPUT_DIR / f"{jid}_portrait_{idx}.png"
    if path.exists():
        return FileResponse(str(path), media_type="image/png")
    return JSONResponse({"error": "not found"}, status_code=404)

PODCAST_SCENE_PROMPT = (
    "empty podcast studio interior, no people, warm grey neutral wall, "
    "large lush tropical areca palm plant with long green fronds in background, "
    "beige or cream upholstered podcast chair visible, "
    "black microphone on boom arm stand, warm amber studio lighting, "
    "soft bokeh background, cinematic shallow depth of field, "
    "professional photography, sharp foreground soft background, "
    "warm cozy atmosphere, photorealistic"
)

@app.post("/generate-scene")
async def generate_scene(
    scene_description: str = Form(...),
    width: int = Form(512),
    height: int = Form(768),
    use_podcast_default: bool = Form(False)
):
    import fal_client
    os.environ["FAL_KEY"] = FAL_KEY
    jid = str(uuid.uuid4())[:8]
    if use_podcast_default or scene_description.strip() == "":
        prompt = PODCAST_SCENE_PROMPT
    else:
        prompt = (f"{scene_description}, no people, no text, photorealistic, "
                  f"cinematic lighting, high quality interior photography, sharp focus")
    try:
        result = fal_client.subscribe("fal-ai/flux/schnell", arguments={
            "prompt": prompt,
            "image_size": "portrait_4_3" if height > width else "landscape_4_3",
            "num_inference_steps": 8, "num_images": 1,
            "enable_safety_checker": False
        })
        img_url = result["images"][0]["url"]
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(img_url)
        img_path = OUTPUT_DIR / f"{jid}_scene.png"
        img_path.write_bytes(r.content)
        return {"job_id": jid, "status": "complete", "image_url": f"/scene/{jid}"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/scene/{jid}")
async def serve_scene(jid: str):
    path = OUTPUT_DIR / f"{jid}_scene.png"
    if path.exists():
        return FileResponse(str(path), media_type="image/png")
    return JSONResponse({"error": "not found"}, status_code=404)

@app.post("/generate-video")
async def generate_video(
    portrait: UploadFile = File(...),
    audio: UploadFile = File(...),
    scene: Optional[UploadFile] = File(None),
    job_id: str = Form(None),
    bg_music: str = Form("")   # track id from MUSIC_TRACKS, empty = no music
):
    jid = job_id or str(uuid.uuid4())[:8]
    set_job(jid, {"status": "running", "progress": 0, "stage": "starting"})
    portrait_path = UPLOAD_DIR / f"{jid}_portrait.png"
    portrait_path.write_bytes(await portrait.read())
    audio_path = UPLOAD_DIR / f"{jid}_audio.mp3"
    audio_path.write_bytes(await audio.read())
    scene_path = None
    if scene and scene.filename:
        scene_path = UPLOAD_DIR / f"{jid}_scene.png"
        scene_path.write_bytes(await scene.read())
    music_path = None
    if bg_music and bg_music in MUSIC_TRACKS:
        music_path = str(MUSIC_DIR / MUSIC_TRACKS[bg_music]["file"])
        print(f"[{jid}] Background music: {bg_music}")
    asyncio.create_task(run_video_pipeline(jid, str(portrait_path), str(audio_path), scene_path, music_path))
    return {"job_id": jid, "status": "running"}

async def run_video_pipeline(jid, portrait_path, audio_path, scene_path=None, music_path=None):
    try:
        import fal_client
        from PIL import Image
        from rembg import remove as rembg_remove
        os.environ["FAL_KEY"] = FAL_KEY
        print(f"[{jid}] Pipeline started.")

        # Stage 1: Remove background with rembg (bria model - best quality)
        set_job(jid, {"status": "running", "stage": "removing_background", "progress": 10})
        print(f"[{jid}] Removing background...")
        portrait_rgba = rembg_remove(Image.open(portrait_path))
        print(f"[{jid}] Background removed.")

        # Stage 2: Composite onto scene if provided
        if scene_path and Path(scene_path).exists():
            set_job(jid, {"status": "running", "stage": "compositing", "progress": 20})
            print(f"[{jid}] Compositing...")
            bg = Image.open(scene_path).convert("RGBA")
            bg = bg.resize(portrait_rgba.size, Image.LANCZOS)
            bg.paste(portrait_rgba, (0, 0), portrait_rgba)
            final_portrait_path = str(UPLOAD_DIR / f"{jid}_composite.png")
            bg.convert("RGB").save(final_portrait_path)
            print(f"[{jid}] Composite done.")
        else:
            final_portrait_path = portrait_path

        # Stage 3: Upload files
        set_job(jid, {"status": "running", "stage": "uploading", "progress": 30})
        print(f"[{jid}] Uploading...")
        portrait_url = fal_client.upload_file(final_portrait_path)
        audio_url = fal_client.upload_file(audio_path)
        print(f"[{jid}] Uploaded.")

        # Stage 4: Generate video with flashtalk
        set_job(jid, {"status": "running", "stage": "generating_video", "progress": 40})
        print(f"[{jid}] Starting flashtalk...")
        result = fal_client.subscribe(
            "fal-ai/flashtalk",
            arguments={"image_url": portrait_url, "audio_url": audio_url}
        )
        print(f"[{jid}] Flashtalk done: duration={result.get('duration')}s")
        output_url = result.get("video", {}).get("url")
        if not output_url:
            raise Exception("No video URL from flashtalk: " + str(result))

        # Stage 5: Download
        set_job(jid, {"status": "running", "stage": "downloading", "progress": 80})
        import urllib.request
        raw_path = str(OUTPUT_DIR / f"{jid}_raw.mp4")
        urllib.request.urlretrieve(output_url, raw_path)
        print(f"[{jid}] Downloaded.")

        # Stage 6: Fix aspect ratio + grain filter + optional background music
        set_job(jid, {"status": "running", "stage": "processing", "progress": 90})
        final_path = str(OUTPUT_DIR / f"{jid}_final.mp4")
        # Scale to true 9:16 (720x1280) — flashtalk outputs landscape so we
        # scale to height 1280 then center-crop to 720 wide.
        vf = (
            "scale=-2:1280,"
            "crop=720:1280,"
            "noise=alls=10:allf=t+u,"
            "unsharp=5:5:1.5:5:5:0.0,"
            "eq=contrast=1.05:brightness=-0.02:saturation=0.92"
        )
        if music_path and Path(music_path).exists():
            print(f"[{jid}] Mixing background music: {music_path}")
            ret = subprocess.run([
                "ffmpeg",
                "-i", raw_path,
                "-stream_loop", "-1", "-i", music_path,
                "-vf", vf,
                "-filter_complex",
                "[1:a]volume=0.12,aloop=loop=-1:size=2147483647[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[a]",
                "-map", "0:v", "-map", "[a]",
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart",
                final_path, "-y"
            ], capture_output=True)
        else:
            ret = subprocess.run([
                "ffmpeg", "-i", raw_path,
                "-vf", vf,
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart",
                final_path, "-y"
            ], capture_output=True)
        print(f"[{jid}] ffmpeg done. Return code: {ret.returncode}")
        if ret.returncode != 0:
            print(f"[{jid}] ffmpeg stderr: {ret.stderr.decode()[:500]}")

        if not Path(final_path).exists():
            final_path = raw_path

        # Upload final video to fal.ai storage for permanent URL
        print(f"[{jid}] Uploading final video to fal.ai storage...")
        final_url = fal_client.upload_file(final_path)
        print(f"[{jid}] Final URL: {final_url}")

        set_job(jid, {"status": "complete", "progress": 100, "stage": "done", 
                      "output": final_path, "url": final_url})
        print(f"[{jid}] Pipeline complete!")

    except Exception as e:
        print(f"[{jid}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        set_job(jid, {"status": "error", "error": str(e), "progress": 0})

@app.get("/video-status/{job_id}")
async def video_status(job_id: str):
    return get_job(job_id)

@app.get("/download/{job_id}")
async def download_video(job_id: str):
    job = get_job(job_id)
    if job.get("status") == "complete":
        # Redirect to permanent fal.ai URL
        url = job.get("url")
        if url:
            from fastapi.responses import RedirectResponse
            return RedirectResponse(url)
        # Fallback to local file if still available
        if job.get("output") and Path(job["output"]).exists():
            return FileResponse(job["output"], media_type="video/mp4",
                              filename=f"persona_{job_id}.mp4")
    return JSONResponse({"error": "not ready"}, status_code=404)




# ---------------------------------------------------------------------------
# Outfit swap — preserves face, body & background; replaces only clothing
# Uses SAM2 for clothing segmentation mask + FLUX inpainting
# ---------------------------------------------------------------------------

@app.post("/swap-outfit")
async def swap_outfit(
    persona_id: str = Form(...),
    outfit: str = Form(...),
    outfit_color: str = Form(""),
):
    """
    Inpaint new clothing onto the locked master portrait.
    Face, body shape, skin, hair and background are pixel-perfect unchanged.
    Returns {job_id, status:"running"} immediately.
    Poll /portrait-status/{jid} for completion.
    """
    import fal_client
    os.environ["FAL_KEY"] = FAL_KEY

    entry = MASTER_PORTRAITS.get(persona_id)
    if not entry:
        return JSONResponse({"error": f"No locked portrait for '{persona_id}'"}, status_code=400)

    master_url = entry["url"]
    jid = str(uuid.uuid4())[:8]
    set_job(jid, {"status": "running", "progress": 0, "stage": "segmenting"})

    outfit_prompt = f"{outfit_color} {outfit}".strip() if outfit_color else outfit

    def _run():
        try:
            import fal_client as _fal
            import requests as _req
            import base64

            # ── Step 1: SAM2 — get clothing mask ──────────────────────────
            print(f"[{jid}] SAM2 segmenting clothing on {master_url[:60]}...")
            set_job(jid, {"status": "running", "progress": 15, "stage": "segmenting_clothing"})

            sam_result = _fal.subscribe("fal-ai/sam2", arguments={
                "image_url": master_url,
                "prompts": [{"type": "text", "text": "clothing, outfit, shirt, dress, top, jacket, clothes"}],
                "output_format": "png",
            })

            mask_url = None
            if sam_result.get("masks"):
                mask_url = sam_result["masks"][0].get("url") or sam_result["masks"][0].get("image", {}).get("url")
            if not mask_url and sam_result.get("image"):
                mask_url = sam_result["image"].get("url")

            if not mask_url:
                raise ValueError(f"SAM2 returned no mask: {sam_result}")

            print(f"[{jid}] SAM2 mask: {mask_url[:60]}")
            set_job(jid, {"status": "running", "progress": 40, "stage": "inpainting"})

            # ── Step 2: FLUX inpainting — fill only the clothing region ───
            inpaint_prompt = (
                f"Professional portrait, {outfit_prompt}, "
                f"photorealistic, sharp, studio lighting, "
                f"same person same pose same background"
            )
            print(f"[{jid}] FLUX inpainting: {inpaint_prompt[:80]}...")

            inpaint_result = _fal.subscribe("fal-ai/flux-lora/inpainting", arguments={
                "image_url": master_url,
                "mask_url": mask_url,
                "prompt": inpaint_prompt,
                "num_inference_steps": 28,
                "strength": 0.95,
                "guidance_scale": 3.5,
                "num_images": 1,
                "enable_safety_checker": False,
            })

            imgs = inpaint_result.get("images") or []
            if not imgs:
                raise ValueError(f"Inpainting returned no images: {inpaint_result}")

            out_url = imgs[0]["url"]
            print(f"[{jid}] Inpainted: {out_url[:60]}")

            # ── Save locally ───────────────────────────────────────────────
            set_job(jid, {"status": "running", "progress": 85, "stage": "saving"})
            out_path = OUTPUT_DIR / f"{jid}_portrait_0.png"
            r = _req.get(out_url, timeout=60)
            out_path.write_bytes(r.content)

            set_job(jid, {
                "status": "complete",
                "progress": 100,
                "images": [f"/portrait/{jid}/0"],
                "method": "outfit-swap-inpaint",
                "outfit": outfit_prompt,
            })
            print(f"[{jid}] Outfit swap complete")

        except Exception as e:
            print(f"[{jid}] Outfit swap error: {e}")
            set_job(jid, {"status": "error", "error": str(e)})

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": jid, "status": "running"}


@app.post("/generate-description")
async def generate_description(
    script: str = Form(...),
    persona_name: str = Form(...),
    niche: str = Form(...),
    platform: str = Form("all")   # tiktok | instagram | youtube | all
):
    """Generate social media descriptions/captions from a video script."""
    from groq import Groq
    client = Groq(api_key=GROQ_KEY)

    prompt = f"""You are a social media copywriter for {persona_name}, a creator in the {niche} niche posting short-form vertical video.

The video script is:
\"\"\"
{script}
\"\"\"

Write platform-optimised captions for this video. Return ONLY a JSON object with these keys:
{{
  "tiktok": "TikTok caption — hook line, 2-3 lines max, 3-5 relevant hashtags, ends with a CTA question",
  "instagram": "Instagram caption — 3-5 sentences expanding slightly on the script's core truth, line breaks between thoughts, 5-8 hashtags on a new line at the end",
  "youtube": "YouTube Shorts description — 3 to 5 sentences: open with a strong hook that names the core truth of the video, expand on why it matters, add a line inviting the viewer to follow for more content like this, then 5-8 relevant hashtags on a new line. Rich, complete and worth reading — not a throwaway line."
}}

Rules:
- Keep the voice and tone of the script — warm, direct, specific
- Never use generic hashtags like #motivation #love — make them specific to the content
- The hook line for TikTok should mirror the script opening or amplify the claim
- No emojis except sparingly on Instagram (1-2 max)
- Do not invent facts not in the script
- Hashtags must be lowercase, no spaces

Return only the JSON object, nothing else."""

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # fast model — no thinking phase, responds in ~3s
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.7,
        )
        raw = resp.choices[0].message.content.strip()
        import re as _re
        raw = raw.replace('```json', '').replace('```', '').strip()
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1:
            raw = raw[start:end+1]
        result = json.loads(raw)
        return result
    except Exception as e:
        return {"error": str(e), "tiktok": "", "instagram": "", "youtube": ""}

@app.get("/proxy-image")
async def proxy_image(url: str):
    """Proxy an external image URL through the backend to avoid CORS issues.
    Used by the frontend to fetch fal.ai CDN portraits as blobs."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(url)
            r.raise_for_status()
            content_type = r.headers.get("content-type", "image/jpeg")
            from fastapi.responses import Response
            return Response(content=r.content, media_type=content_type)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)
