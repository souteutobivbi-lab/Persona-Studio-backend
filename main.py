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
    skill_section = ""
    if skill_context:
        skill_section = f"\n\nPERSONA SKILL:\n{skill_context[:2000]}\nMatch this persona's exact tone and style."
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": f"""Write 5 different 15-second talking head scripts for {persona_name}, aged {persona_age}, British, in the {niche} niche.
Topic: {topic}{skill_section}

Rules for each script:
- Maximum 40 words each
- Different hook for each script
- Conversational, warm, authoritative
- End with soft CTA or persona sign-off
- No hashtags, no emojis, no stage directions
- Each script must feel distinct — different angle, different hook, different energy

Return ONLY a JSON array of 5 script strings, nothing else. Example format:
["Script one here.", "Script two here.", "Script three here.", "Script four here.", "Script five here."]"""}],
        max_tokens=600, temperature=0.85
    )
    raw = response.choices[0].message.content.strip()
    try:
        raw = raw.replace('```json','').replace('```','').strip()
        scripts = json.loads(raw)
        if not isinstance(scripts, list):
            scripts = [scripts]
    except:
        scripts = [raw]
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

# In-memory master portrait store  {persona_id: fal_cdn_url}
# Using fal CDN URLs so they survive Railway container restarts
MASTER_PORTRAITS: dict = {}  # persona_id → {"local": path, "url": fal_url}

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
    print(f"Master portrait set for {persona_id}: {fal_url}")
    return {"status": "ok", "persona_id": persona_id, "fal_url": fal_url}

@app.post("/get-master-portrait")
async def get_master_portrait(persona_id: str = Form(...)):
    """Return the stored master portrait URL for a persona."""
    entry = MASTER_PORTRAITS.get(persona_id)
    if not entry:
        return JSONResponse({"error": "No master portrait set for this persona"}, status_code=404)
    return {"persona_id": persona_id, "fal_url": entry["url"], "has_master": True}

@app.post("/generate-portrait")
async def generate_portrait(
    persona_name: str = Form(...),
    persona_age: str = Form(...),
    appearance: str = Form(...),
    outfit: str = Form("elegant fitted dress"),
    outfit_color: str = Form(""),
    count: int = Form(1),
    persona_id: str = Form(""),
    master_url: str = Form("")   # can pass fal CDN URL directly if known
):
    """Generate podcast-style portrait.

    Pipeline:
      A) If master portrait exists → PuLID (identity-locked, one generation, no retries)
         Cost: ~$0.05 per image. Face is 100% consistent with master every time.
      B) No master portrait yet → FLUX Schnell baseline portrait
         Cost: ~$0.02 per image. Face varies. Use this to generate the master.
    """
    import fal_client
    os.environ["FAL_KEY"] = FAL_KEY
    jid = str(uuid.uuid4())[:8]

    # Resolve master portrait URL — must be a public https URL for PuLID
    ref_url = master_url.strip()
    # If it's a relative local path (e.g. /portrait/abc/0), ignore it — not fetchable by fal
    if ref_url and not ref_url.startswith("http"):
        ref_url = ""
    # Fall back to in-memory store (set via /set-master-portrait which uploads to fal CDN)
    if not ref_url and persona_id:
        entry = MASTER_PORTRAITS.get(persona_id)
        if entry:
            ref_url = entry["url"]
    # Also check by persona_name in case persona_id is the name string
    if not ref_url and persona_id:
        for key, entry in MASTER_PORTRAITS.items():
            if key.lower() == persona_id.lower():
                ref_url = entry["url"]
                break

    prompt = build_portrait_prompt(appearance, persona_age, outfit, outfit_color)
    images = []
    identity_scores = []

    try:
        for i in range(min(count, 4)):
            img_path = OUTPUT_DIR / f"{jid}_portrait_{i}.png"

            if ref_url:
                # ── PuLID: inject master face identity into the generated image ──
                # One call, no retry loop. Identity is baked in by the model itself.
                print(f"[{jid}] PuLID generation {i} with master: {ref_url[:60]}...")
                result = fal_client.subscribe("fal-ai/pulid", arguments={
                    "prompt": prompt,
                    "reference_images": [{"image_url": ref_url}],
                    "num_inference_steps": 12,
                    "guidance_scale": 1.5,
                    "image_size": "portrait_4_3",
                    "enable_safety_checker": False
                })
                img_url = result["images"][0]["url"]
                method = "pulid"
            else:
                # ── FLUX Dev: better prompt following for seated/scene portraits ──
                print(f"[{jid}] FLUX Dev generation {i} (no master portrait set)")
                result = fal_client.subscribe("fal-ai/flux/dev", arguments={
                    "prompt": prompt,
                    "image_size": "portrait_4_3",
                    "num_inference_steps": 28,
                    "guidance_scale": 3.5,
                    "num_images": 1,
                    "enable_safety_checker": False
                })
                img_url = result["images"][0]["url"]
                method = "flux"

            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get(img_url)
            img_path.write_bytes(r.content)

            # Quality check only — no retry, no extra cost
            score = -1.0
            if ref_url:
                master_local = MASTER_PORTRAITS.get(persona_id, {}).get("local")
                if master_local and Path(master_local).exists():
                    score = get_face_similarity(str(img_path), master_local)

            images.append(f"/portrait/{jid}/{i}")
            identity_scores.append(round(score, 3) if score >= 0 else None)

        return {
            "job_id": jid,
            "status": "complete",
            "images": images,
            "prompt_ids": images,
            "identity_scores": identity_scores,
            "method": "pulid" if ref_url else "flux",
            "master_used": bool(ref_url)
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

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
    job_id: str = Form(None)
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
    asyncio.create_task(run_video_pipeline(jid, str(portrait_path), str(audio_path), scene_path))
    return {"job_id": jid, "status": "running"}

async def run_video_pipeline(jid, portrait_path, audio_path, scene_path=None):
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

        # Stage 6: Fix aspect ratio + grain filter
        # flashtalk outputs 768x448 landscape — scale up and pad to 9:16
        set_job(jid, {"status": "running", "stage": "processing", "progress": 90})
        final_path = str(OUTPUT_DIR / f"{jid}_final.mp4")
        ret = subprocess.run([
            "ffmpeg", "-i", raw_path,
            "-vf", (
                "scale=448:448:force_original_aspect_ratio=decrease,"
                "pad=448:768:(ow-iw)/2:(oh-ih)/2:black,"
                "noise=alls=15:allf=t+u,"
                "unsharp=5:5:1.8:5:5:0.0,"
                "eq=contrast=1.06:brightness=-0.02:saturation=0.92"
            ),
            "-c:v", "libx264", "-crf", "17", "-c:a", "copy",
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
