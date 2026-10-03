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
    return {"status": "ok", "version": "3.1", "host": "railway"}

@app.get("/")
async def root():
    p = Path(__file__).parent / "static" / "index.html"
    if p.exists():
        return HTMLResponse(p.read_text())
    return {"api": "Persona Studio v3.1"}

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
        if "<think>" in raw:
            raw = raw[raw.rfind("</think>")+8:].strip()
        raw = raw.replace('```json','').replace('```','').strip()
        parsed = json.loads(raw)
        return {"status": "ok", "data": parsed}
    except Exception as e:
        return JSONResponse({"error": "Parse failed: "+str(e), "raw": response.choices[0].message.content[:200]}, status_code=400)

@app.get("/voices")
async def get_voices(gender: str = "female", accent: str = "british"):
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get("https://api.elevenlabs.io/v1/voices",
            headers={"xi-api-key": ELEVENLABS_KEY})
    if r.status_code != 200:
        return JSONResponse({"error": "ElevenLabs error"}, status_code=400)
    all_voices = r.json().get("voices", [])

    british_terms = ["british", "english", "uk", "united kingdom"]
    british_name_hints = ["alice", "lily", "charlotte", "emily", "grace", "sophie", "emma",
                          "james", "william", "george", "oliver", "henry", "thomas", "edward"]

    def voice_score(v):
        labels = v.get("labels", {})
        label_str = " ".join(str(val) for val in labels.values()).lower()
        name_lower = v.get("name", "").lower()
        score = 0
        # Gender match — strong signal
        g = labels.get("gender", labels.get("Gender", "")).lower()
        if gender.lower() in g:
            score += 10
        elif gender.lower() in label_str:
            score += 8
        # British accent — strong signal
        if any(t in label_str for t in british_terms):
            score += 6
        # British name hints
        if any(h in name_lower for h in british_name_hints):
            score += 3
        # Penalise clear wrong gender
        wrong = "male" if gender.lower() == "female" else "female"
        if wrong == g and g:
            score -= 20
        return score

    # Score and sort all voices
    scored = sorted(all_voices, key=voice_score, reverse=True)

    # Split into tiers
    gender_and_british = [v for v in scored if voice_score(v) >= 14]
    gender_only = [v for v in scored if 8 <= voice_score(v) < 14]
    remainder = [v for v in scored if voice_score(v) >= 5]

    # Return best available: prefer British+gender, fallback to gender-only, fallback to remainder
    if len(gender_and_british) >= 3:
        result = gender_and_british
    elif len(gender_only) >= 3:
        result = gender_only
    else:
        result = remainder[:20]

    return [{"id": v["voice_id"], "name": v["name"],
             "description": ", ".join(str(val) for val in v.get("labels", {}).values()),
             "gender": v.get("labels", {}).get("gender", v.get("labels", {}).get("Gender", "")),
             "accent": v.get("labels", {}).get("accent", v.get("labels", {}).get("Accent", ""))}
            for v in result]

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

    # Build skill instruction block — drives all script quality
    if skill_context and len(skill_context.strip()) > 50:
        skill_block = f"""
PERSONA SKILL FILE — follow this precisely:
{skill_context[:3000]}

You must:
- Use ONLY vocabulary listed under VOCABULARY TO USE
- Avoid every word and phrase under VOCABULARY TO AVOID
- Follow the SCRIPT STRUCTURE exactly (Hook / Body / CTA)
- Match the VOICE & TONE description with precision
- Use hooks from HOOKS THAT WORK or HOOKS THAT STOP THE SCROLL as structural models
- End with one of the SIGN-OFFS listed in the skill file
- Write in this persona's specific, distinctive voice — not generic influencer language
- Pull from EXAMPLE SCRIPTS as tone references — not to copy but to match register
"""
    else:
        skill_block = f"""
Persona: {persona_name}, {persona_age} years old, British, {niche} niche.
Voice: authoritative, measured, direct. No hype. No filler. British understatement.
Structure: Hook (5 words max) / Body (one specific concrete insight) / CTA (quiet, natural).
"""

    system_prompt = """You are a professional short-form video scriptwriter specialising in British AI influencer content.
You write scripts that sound like a real, specific person with earned authority — not a content creator.

Every script must have:
1. A HOOK that stops the scroll in the first breath — a hard truth, a specific claim, a reframe, or a disruption of assumption. No warm-up. No preamble.
2. A BODY that delivers ONE specific, concrete insight with a named mechanism — not vague, not motivational filler, not a list
3. A CTA that sounds completely natural — the way this specific person signs off, not a generic call to action

Non-negotiable rules:
- Maximum 42 words per script
- No emojis, no hashtags, no stage directions, no asterisks, no quotation marks around the whole script
- Banned phrases: "game changer", "level up", "hustle", "grind", "passive income", "amazing", "literally", "guys", "awesome"
- Each of the 5 scripts must use a completely different hook, angle and energy
- The body insight must be SPECIFIC — name the mechanism, the number, the exact thing people miss
- Scripts must sound spoken, not written — short sentences, natural rhythm, real pauses
- British English only: colour, realise, whilst, neighbour, practise
- Strip all thinking tags from output
- Return ONLY a valid JSON array of exactly 5 strings. No markdown. No explanation. No preamble."""

    user_prompt = f"""Write 5 distinct 15-second video scripts for {persona_name}, aged {persona_age}.
Topic: {topic}
Niche: {niche}

{skill_block}

The 5 scripts must each approach the topic from a completely different angle:
1. Open with a hard truth or counterintuitive claim that challenges what they think they know
2. Open with a specific number, timeframe, or concrete detail that earns instant credibility
3. Open with the mistake most people make — name it precisely
4. Open with a pattern this persona has observed repeatedly — "thirty years tells you..." style
5. Open with a direct reframe — what they call X is actually Y

Return ONLY: ["script1", "script2", "script3", "script4", "script5"]"""

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=900,
        temperature=0.82
    )
    raw = response.choices[0].message.content.strip()
    try:
        # Strip qwen thinking tags
        if "<think>" in raw:
            raw = raw[raw.rfind("</think>")+8:].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        scripts = json.loads(raw)
        if not isinstance(scripts, list):
            scripts = [scripts]
        # Clean each script
        scripts = [s.strip().strip('"').strip("'").strip() for s in scripts if s and len(s.strip()) > 10]
    except:
        import re
        found = re.findall(r'"([^"]{20,300})"', raw)
        scripts = found if len(found) >= 3 else [raw[:300]]
    return {"scripts": scripts[:5], "script": scripts[0] if scripts else "", "words": len(scripts[0].split()) if scripts else 0}

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

@app.post("/generate-portrait")
async def generate_portrait(
    persona_name: str = Form(...),
    persona_age: str = Form(...),
    appearance: str = Form(...),
    outfit: str = Form("professional blazer"),
    count: int = Form(1)
):
    import fal_client
    os.environ["FAL_KEY"] = FAL_KEY
    jid = str(uuid.uuid4())[:8]
    # Build portrait prompt with strong age, framing and realism signals
    age_int = int(persona_age) if str(persona_age).isdigit() else 40
    # Age-specific descriptors to prevent FLUX from defaulting to ~30s appearance
    if age_int >= 55:
        age_desc = "deep character lines, silver or salt-and-pepper hair, aged hands visible, crow's feet, natural age spots"
    elif age_int >= 48:
        age_desc = "visible laughter lines around eyes, subtle forehead lines, mature skin with natural texture, slight crow's feet"
    elif age_int >= 40:
        age_desc = "fine lines around eyes, natural skin with visible pores, mature refined appearance, subtle laughter lines"
    else:
        age_desc = "natural skin with visible pores, authentic appearance"

    # Always start from the appearance text, strip any existing framing words
    import re as _re
    base = appearance
    # Remove any existing framing instructions so we can replace with waist-up
    base = _re.sub(r'\b(upper body|head shot|headshot|bust shot|close.?up|shoulder up|chest up|torso)\b',
                   '', base, flags=_re.IGNORECASE).strip().strip(',').strip()

    if "studio portrait" in base.lower() or "canon" in base.lower():
        # Skill file already has full prompt — inject age desc and force waist-up
        prompt = f"{base}, {age_desc}, waist up framing showing full torso and hands, in the middle"
    else:
        prompt = (
            f"Studio portrait photograph of a {persona_age} year old {base}, "
            f"{age_desc}, "
            f"{outfit}, seamless white studio backdrop, "
            f"waist up framing showing full torso and hands, facing camera directly, "
            f"Canon EOS 5D Mark IV 85mm f/2.8, single large softbox at 45 degrees camera left, "
            f"white reflector fill on right, visible skin pores, natural skin subsurface scattering, "
            f"fine hair strands, no retouching, no filters, no airbrushing, no digital smoothing, "
            f"editorial photography, in the middle"
        )
    # Append strong negative framing to prevent close-up crops
    prompt += (", full waist-up composition, torso and hands visible in frame, "
               "NOT a headshot, NOT a close-up, NOT cropped at shoulders, "
               "subject occupies upper two-thirds of frame, breathing room below waist")

    images = []
    try:
        for i in range(min(count, 4)):
            result = fal_client.subscribe("fal-ai/flux/dev", arguments={
                "prompt": prompt,
                "image_size": "portrait_4_3",  # 768x1024 — supported named size
                "num_inference_steps": 28,
                "num_images": 1,
                "enable_safety_checker": False,
                "guidance_scale": 4.0  # slightly higher to enforce prompt more strictly
            })
            img_url = result["images"][0]["url"]
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get(img_url)
            img_path = OUTPUT_DIR / f"{jid}_portrait_{i}.png"
            img_path.write_bytes(r.content)
            images.append(f"/portrait/{jid}/{i}")
        return {"job_id": jid, "status": "complete", "images": images}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/portrait/{jid}/{idx}")
async def serve_portrait(jid: str, idx: int):
    path = OUTPUT_DIR / f"{jid}_portrait_{idx}.png"
    if path.exists():
        return FileResponse(str(path), media_type="image/png")
    return JSONResponse({"error": "not found"}, status_code=404)

@app.post("/generate-scene")
async def generate_scene(
    scene_description: str = Form(...),
    width: int = Form(512),
    height: int = Form(768)
):
    import fal_client
    os.environ["FAL_KEY"] = FAL_KEY
    jid = str(uuid.uuid4())[:8]
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

        set_job(jid, {"status": "running", "stage": "removing_background", "progress": 10})
        print(f"[{jid}] Removing background...")
        portrait_rgba = rembg_remove(Image.open(portrait_path))

        # Fix white halo — erode the alpha mask edges to remove fringe pixels
        import numpy as np
        rgba_arr = np.array(portrait_rgba)
        alpha = rgba_arr[:, :, 3]
        # Erode alpha by 2px using a simple minimum filter to remove edge fringe
        from PIL import ImageFilter
        alpha_img = Image.fromarray(alpha)
        alpha_eroded = alpha_img.filter(ImageFilter.MinFilter(3))  # 3px min = 1px erosion
        alpha_eroded = alpha_eroded.filter(ImageFilter.GaussianBlur(radius=1))  # feather edge
        rgba_arr[:, :, 3] = np.array(alpha_eroded)
        portrait_rgba = Image.fromarray(rgba_arr)
        print(f"[{jid}] Background removed + halo fixed.")

        if scene_path and Path(scene_path).exists():
            set_job(jid, {"status": "running", "stage": "compositing", "progress": 20})
            print(f"[{jid}] Compositing...")
            bg = Image.open(scene_path).convert("RGBA")
            # Scale scene to fill portrait dimensions
            bg = bg.resize(portrait_rgba.size, Image.LANCZOS)
            composite = Image.new("RGBA", portrait_rgba.size, (0, 0, 0, 255))
            composite.paste(bg, (0, 0))
            composite.paste(portrait_rgba, (0, 0), portrait_rgba)
            final_portrait_path = str(UPLOAD_DIR / f"{jid}_composite.png")
            composite.convert("RGB").save(final_portrait_path)
            print(f"[{jid}] Composite done.")
        else:
            # No scene — paste onto black background (not transparent) for Flashtalk
            bg = Image.new("RGB", portrait_rgba.size, (0, 0, 0))
            bg.paste(portrait_rgba, (0, 0), portrait_rgba)
            final_portrait_path = str(UPLOAD_DIR / f"{jid}_nobg.png")
            bg.save(final_portrait_path)
            print(f"[{jid}] Portrait on black background saved.")

        set_job(jid, {"status": "running", "stage": "uploading", "progress": 30,
                      "message": "Uploading portrait and audio to fal.ai..."})
        print(f"[{jid}] Uploading...")
        portrait_url = fal_client.upload_file(final_portrait_path)
        audio_url = fal_client.upload_file(audio_path)
        print(f"[{jid}] Uploaded.")

        set_job(jid, {"status": "running", "stage": "generating_video", "progress": 45,
                      "message": "Flashtalk generating lip-sync video... (~30-60 seconds)"})
        print(f"[{jid}] Starting flashtalk...")
        result = fal_client.subscribe(
            "fal-ai/flashtalk",
            arguments={"image_url": portrait_url, "audio_url": audio_url}
        )
        print(f"[{jid}] Flashtalk done: duration={result.get('duration')}s")
        output_url = result.get("video", {}).get("url")
        if not output_url:
            raise Exception("No video URL from flashtalk: " + str(result))

        set_job(jid, {"status": "running", "stage": "downloading", "progress": 80,
                      "message": "Downloading video from fal.ai..."})
        import urllib.request
        raw_path = str(OUTPUT_DIR / f"{jid}_raw.mp4")
        urllib.request.urlretrieve(output_url, raw_path)
        print(f"[{jid}] Downloaded.")

        set_job(jid, {"status": "running", "stage": "processing", "progress": 88,
                      "message": "Applying grain filter and encoding for mobile..."})
        final_path = str(OUTPUT_DIR / f"{jid}_final.mp4")

        # Flashtalk outputs 768x448 (landscape).
        # Scale to fill 9:16 (1080x1920) — video fills width, padded top/bottom.
        # CRF 23 + preset fast = good quality, smaller file, faster streaming.
        ret = subprocess.run([
            "ffmpeg", "-i", raw_path,
            "-vf", (
                "scale=1080:-2,"
                "pad=1080:1920:0:(oh-ih)/2:black,"
                "noise=alls=12:allf=t+u,"
                "unsharp=3:3:1.2:3:3:0.0,"
                "eq=contrast=1.05:brightness=-0.01:saturation=0.95"
            ),
            "-c:v", "libx264", "-crf", "23", "-preset", "fast",
            "-profile:v", "baseline", "-level", "3.1",
            "-c:a", "aac", "-b:a", "96k", "-ar", "44100",
            "-movflags", "+faststart",
            "-maxrate", "2M", "-bufsize", "4M",
            final_path, "-y"
        ], capture_output=True)
        print(f"[{jid}] ffmpeg done. Return code: {ret.returncode}")
        if ret.returncode != 0:
            print(f"[{jid}] ffmpeg stderr: {ret.stderr.decode()[:500]}")

        if not Path(final_path).exists():
            final_path = raw_path

        set_job(jid, {"status": "running", "stage": "uploading_cdn", "progress": 95,
                      "message": "Uploading to CDN for permanent storage..."})
        print(f"[{jid}] Uploading final video to fal.ai storage...")
        final_url = fal_client.upload_file(final_path)
        print(f"[{jid}] Final URL: {final_url}")

        set_job(jid, {"status": "complete", "progress": 100, "stage": "done",
                      "message": "Video ready!", "output": final_path, "url": final_url})
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
        url = job.get("url")
        if url:
            from fastapi.responses import RedirectResponse
            return RedirectResponse(url)
        if job.get("output") and Path(job["output"]).exists():
            return FileResponse(job["output"], media_type="video/mp4",
                              filename=f"persona_{job_id}.mp4")
    return JSONResponse({"error": "not ready"}, status_code=404)
