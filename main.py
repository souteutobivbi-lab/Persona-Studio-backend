import asyncio, os, random, subprocess, uuid, json, mimetypes
from pathlib import Path
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import httpx

def fal_upload(path: str) -> str:
    """Upload a file to fal.ai storage — handles both old and new client APIs."""
    import fal_client
    path = str(path)
    # Try the standard upload_file first
    try:
        return fal_client.upload_file(path)
    except Exception as e1:
        if "Invalid storage type" not in str(e1) and "storage" not in str(e1).lower():
            raise  # Different error — re-raise
        # Fallback: read bytes and use fal_client.upload with explicit content type
        try:
            content_type, _ = mimetypes.guess_type(path)
            if not content_type:
                ext = Path(path).suffix.lower()
                content_type = {
                    '.mp4': 'video/mp4', '.mp3': 'audio/mpeg',
                    '.png': 'image/png', '.jpg': 'image/jpeg',
                    '.zip': 'application/zip', '.webp': 'image/webp'
                }.get(ext, 'application/octet-stream')
            data = Path(path).read_bytes()
            return fal_client.upload(data, content_type)
        except Exception as e2:
            # Final fallback: upload via httpx directly to fal storage
            try:
                import httpx as _httpx
                content_type2, _ = mimetypes.guess_type(path)
                if not content_type2:
                    content_type2 = 'application/octet-stream'
                data2 = Path(path).read_bytes()
                headers = {
                    "Authorization": f"Key {os.environ.get('FAL_KEY','')}",
                    "Content-Type": content_type2,
                    "Accept": "application/json"
                }
                r = _httpx.post(
                    "https://rest.alpha.fal.ai/storage/upload/initiate",
                    headers={"Authorization": headers["Authorization"], "Accept": "application/json"},
                    json={"content_type": content_type2, "file_name": Path(path).name}
                )
                if r.status_code == 200:
                    rd = r.json()
                    upload_url = rd.get("upload_url")
                    file_url = rd.get("file_url")
                    if upload_url:
                        _httpx.put(upload_url, content=data2, headers={"Content-Type": content_type2})
                        return file_url
            except Exception as e3:
                print(f"All upload methods failed: {e1} | {e2} | {e3}")
            raise e2

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
    return {"status": "ok", "version": "3.2", "host": "railway", "brand": "MediaHouz"}

@app.get("/")
async def root():
    p = Path(__file__).parent / "static" / "index.html"
    if p.exists():
        return HTMLResponse(p.read_text())
    return {"api": "MediaHouz v3.2"}

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
    # Hardcoded confirmed British English voices from ElevenLabs
    # These are verified British — no scoring needed, no false positives
    BRITISH_FEMALE = [
        {"id": "EXAVITQu4vr4xnSDxMaL", "name": "Alice",   "description": "British, female, clear, professional, middle-aged"},
        {"id": "pFZP5JQG7iQjIQuC4Bku", "name": "Lily",    "description": "British, female, warm, narration, velvety"},
        {"id": "ThT5KcBeYPX3keUQqHPh", "name": "Dorothy", "description": "British, female, pleasant, authoritative"},
        {"id": "AZnzlk1XvdvUeBnXmlld", "name": "Evelyn",  "description": "British, female, confident, news"},
        {"id": "XB0fDUnXU5powFXDhCwa", "name": "Charlotte","description": "British, female, seductive, mature"},
        {"id": "jBpfuIE2acCO8z3wKNLl", "name": "Serena",  "description": "British, female, composed, editorial"},
    ]
    BRITISH_MALE = [
        {"id": "JBFqnCBsd6RMkjVDRZzb", "name": "George",  "description": "British, male, warm, distinguished, narration"},
        {"id": "GBv7mTt0atIp3Br8iCZE", "name": "Daniel",  "description": "British, male, authoritative, news, deep"},
        {"id": "ODq5zmih8GrVes37Dy39", "name": "Geoffrey", "description": "British, male, strong, broadcast"},
        {"id": "N2lVS1w4EtoT3dr4eOWO", "name": "Callum",  "description": "British, male, gravelly, character"},
        {"id": "IKne3meq5aSn9XLyUdCD", "name": "Peter",   "description": "British, male, neutral, professional"},
        {"id": "onwK4e9ZLuTAKqWW03F9", "name": "Daniel B","description": "British, male, deep, editorial"},
    ]

    curated = BRITISH_FEMALE if gender.lower() == "female" else BRITISH_MALE

    results = []

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            # 1. Get account voices (includes added library voices)
            r1 = await client.get("https://api.elevenlabs.io/v1/voices",
                headers={"xi-api-key": ELEVENLABS_KEY})
            account_voices = r1.json().get("voices", []) if r1.status_code == 200 else []

            # 2. Get shared library British voices
            params = {
                "language": "en",
                "gender": gender,
                "accent": "british",
                "page_size": 100,
                "sort": "clones_count"  # most popular first
            }
            r2 = await client.get("https://api.elevenlabs.io/v1/voices/shared",
                headers={"xi-api-key": ELEVENLABS_KEY},
                params=params)
            shared_voices = r2.json().get("voices", []) if r2.status_code == 200 else []

        # Build result — account voices first (already added), then shared library
        seen_ids = set()

        # Account voices that are British
        british_terms = ["british", "english", "uk"]
        for v in account_voices:
            labels = v.get("labels", {})
            label_str = " ".join(str(val) for val in labels.values()).lower()
            acc = labels.get("accent", labels.get("Accent", "")).lower()
            gen = labels.get("gender", labels.get("Gender", "")).lower()
            is_british = any(t in acc or t in label_str for t in british_terms)
            is_gender = gender.lower() in gen or gender.lower() in label_str
            if is_british and is_gender and v["voice_id"] not in seen_ids:
                results.append({
                    "id": v["voice_id"], "name": v["name"],
                    "description": ", ".join(str(val) for val in labels.values()),
                    "gender": gen, "accent": acc, "source": "account"
                })
                seen_ids.add(v["voice_id"])

        # Shared library British voices
        for v in shared_voices:
            if v.get("voice_id") not in seen_ids:
                labels = v.get("labels", {})
                results.append({
                    "id": v["voice_id"], "name": v["name"],
                    "description": ", ".join(str(val) for val in labels.values()),
                    "gender": labels.get("gender", gender),
                    "accent": labels.get("accent", "british"),
                    "source": "library"
                })
                seen_ids.add(v["voice_id"])

        # Always include curated defaults if not already present
        for v in curated:
            if v["id"] not in seen_ids:
                results.append({"id": v["id"], "name": v["name"],
                                 "description": v["description"],
                                 "gender": gender, "accent": "british", "source": "default"})
                seen_ids.add(v["id"])

        if results:
            return results[:50]  # cap at 50

    except Exception as e:
        print(f"Voice fetch error: {e}")

    # Fallback — confirmed curated list
    return [{"id": v["id"], "name": v["name"], "description": v["description"],
             "gender": gender, "accent": "british"} for v in curated]

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

    # Extract first name for self-introduction
    first_name = persona_name.split()[0] if persona_name else "I"

    # Build skill instruction block
    if skill_context and len(skill_context.strip()) > 50:
        skill_block = f"""
PERSONA SKILL FILE — follow every instruction precisely:
{skill_context[:3000]}

You MUST:
- Use ONLY vocabulary listed under VOCABULARY TO USE
- Avoid every word and phrase listed under VOCABULARY TO AVOID
- Match the VOICE & TONE description exactly
- Use the SCRIPT STRUCTURE from the skill file
- Draw hooks from HOOKS THAT WORK / HOOKS THAT STOP THE SCROLL as structural models
- End with one of the SIGN-OFFS from the skill file
- The self-introduction must use the persona's first name and role as defined in the skill file
"""
    else:
        skill_block = f"""
Persona: {persona_name}, {persona_age} years old, British, {niche} niche.
Voice: authoritative, measured, direct. No hype. No filler. British understatement.
"""

    system_prompt = f"""You are a professional short-form video scriptwriter for British AI influencer content.
You write scripts that build real audiences through genuine expertise — not entertainment, not motivation, not lifestyle. Education and authority.

SCRIPT STRUCTURE — every script must follow this exactly:

1. SELF-INTRODUCTION (first 8-10 words):
   The persona introduces themselves by first name and role every single time.
   Examples:
   - "I'm {first_name}. I've spent twenty years in property development."
   - "My name is {first_name} — interior designer and authority on space."
   - "{first_name} here. Wealth strategist. And I need to tell you something."
   This is non-negotiable. Every script starts with who they are.

2. HOOK / INSIGHT (the body — 30-40 words):
   ONE specific, factual, researched insight that the audience genuinely did not know.
   Must be:
   - Specific and verifiable — cite a number, a principle, a named concept, a mechanism
   - Genuinely educational — teach them something real about the niche
   - Surprising — challenge a common assumption with evidence or expertise
   - Connected to the topic directly
   Not: vague observations, motivational filler, generic advice

3. CTA / SIGN-OFF (final 8-12 words):
   Natural, unhurried, persona-specific. Tells the viewer exactly what to do next and why.
   Must drive followership — give a specific reason to follow, not just "follow me".
   Examples:
   - "Follow for one insight every week that most designers never share."
   - "I cover this in depth tomorrow. Follow so you don't miss it."
   - "More on this. Follow — every video covers one thing that changes how you see this."

TARGET LENGTH: 55-70 words per script (approximately 20 seconds spoken at measured British pace)

NON-NEGOTIABLE RULES:
- Every script opens with first name and role — no exceptions
- All facts and insights must be accurate and specific — no made-up statistics
- No emojis, hashtags, stage directions, asterisks
- Banned phrases: "game changer", "level up", "hustle", "grind", "passive income", "amazing", "literally", "guys", "awesome", "journey"
- British English: colour, realise, whilst, neighbour, practise, grey
- Scripts must sound spoken — short sentences, natural pauses, real rhythm
- Each of the 5 scripts must approach the topic from a genuinely different angle
- Return ONLY a valid JSON array of exactly 5 strings. No markdown. No explanation."""

    user_prompt = f"""Write 5 distinct ~20-second video scripts for {persona_name}, aged {persona_age}, British.
Topic: {topic}
Niche: {niche}

{skill_block}

The 5 scripts must each cover a genuinely different aspect, angle or insight on the topic:
1. A foundational principle most people in the audience violate without knowing it
2. A specific named technique, rule or standard used by professionals in this niche
3. The most common expensive mistake made — with a real consequence named
4. A counterintuitive truth backed by experience or research — what they assume is wrong
5. A practical one-sentence rule the audience can apply immediately after watching

Each script: self-introduction → factual insight → follow CTA. 55-70 words. Spoken naturally.

Return ONLY: ["script1", "script2", "script3", "script4", "script5"]"""

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=1500,
        temperature=0.75
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

def format_script_for_tts(script: str) -> str:
    """Format script text to guide natural ElevenLabs delivery.
    - Ensure sentence-ending pauses with proper punctuation
    - Add natural breaks between hook, body and CTA
    - Remove any double spaces or artifacts
    """
    import re
    text = script.strip()
    # Ensure sentences end with punctuation for natural pauses
    text = re.sub(r'([a-z])\s{2,}([A-Z])', r'\1. \2', text)
    # Add pause after self-introduction line (first sentence ending with name/role)
    text = re.sub(r'(\b(?:here|designer|strategist|builder|advisor|coach)\b\.?)(\s)', r'\1 \2', text, flags=re.IGNORECASE)
    # Clean up
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.post("/generate-voice")
async def generate_voice(
    script: str = Form(...),
    voice_id: str = Form("EXAVITQu4vr4xnSDxMaL"),  # Alice — confirmed British female
    job_id: str = Form(None)
):
    jid = job_id or str(uuid.uuid4())[:8]
    out_path = OUTPUT_DIR / f"{jid}_voice.mp3"

    # Format script for natural delivery
    formatted_script = format_script_for_tts(script)

    # Try eleven_turbo_v2_5 first (best for natural English), fallback to multilingual
    # Settings tuned for natural British delivery:
    # - stability 0.45: more natural variation, less robotic consistency
    # - similarity_boost 0.82: close to voice without over-enunciating
    # - style 0.25: slight expressiveness without being dramatic
    # - use_speaker_boost: true — improves clarity
    voice_settings = {
        "stability": 0.45,
        "similarity_boost": 0.82,
        "style": 0.20,
        "use_speaker_boost": True
    }

    async with httpx.AsyncClient(timeout=45) as client:
        # Try eleven_turbo_v2_5 — best natural English quality
        r = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={"xi-api-key": ELEVENLABS_KEY, "Content-Type": "application/json"},
            json={
                "text": formatted_script,
                "model_id": "eleven_turbo_v2_5",
                "voice_settings": voice_settings,
                "output_format": "mp3_44100_128"
            }
        )
        # Fallback to multilingual_v2 if turbo not available
        if r.status_code != 200:
            r = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={"xi-api-key": ELEVENLABS_KEY, "Content-Type": "application/json"},
                json={
                    "text": formatted_script,
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": voice_settings,
                    "output_format": "mp3_44100_128"
                }
            )

    if r.status_code != 200:
        return JSONResponse({"error": r.text[:300]}, status_code=400)

    out_path.write_bytes(r.content)
    return {"job_id": jid, "audio_path": str(out_path), "status": "done"}

@app.get("/audio/{job_id}")
async def serve_audio(job_id: str):
    path = OUTPUT_DIR / f"{job_id}_voice.mp3"
    if path.exists():
        return FileResponse(str(path), media_type="audio/mpeg")
    return JSONResponse({"error": "not found"}, status_code=404)

@app.post("/train-lora")
async def train_lora(
    persona_name: str = Form(...),
    trigger_word: str = Form(...),
    image_urls: str = Form(...)  # JSON array of image URLs
):
    import fal_client, json as _json
    os.environ["FAL_KEY"] = FAL_KEY
    jid = "lora_" + str(uuid.uuid4())[:8]

    try:
        urls = _json.loads(image_urls)
        # Filter to valid URLs only
        valid_urls = [u for u in urls if u and (u.startswith('http') or u.startswith('data:image'))]
        print(f"[{jid}] Training with {len(valid_urls)} images (from {len(urls)} total)")

        if len(valid_urls) < 1:
            return JSONResponse({"error": "No valid images found. Please regenerate portraits first."}, status_code=400)
        print(f"[{jid}] Valid images: {len(valid_urls)} (CDN: {sum(1 for u in valid_urls if u.startswith('http'))}, base64: {sum(1 for u in valid_urls if u.startswith('data'))})")

        # Download all images and create a zip for training
        import zipfile, io, base64 as _b64
        zip_buffer = io.BytesIO()
        added = 0
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            async with httpx.AsyncClient(timeout=60) as client:
                for i, url in enumerate(valid_urls):
                    try:
                        if url.startswith('data:image'):
                            # Base64 data URL — decode directly
                            header, data = url.split(',', 1)
                            img_bytes = _b64.b64decode(data)
                            zf.writestr(f"image_{i:02d}.png", img_bytes)
                            added += 1
                            print(f"[{jid}] Added base64 image {i} ({len(img_bytes)} bytes)")
                        elif url.startswith('http'):
                            r = await client.get(url)
                            if r.status_code == 200:
                                zf.writestr(f"image_{i:02d}.png", r.content)
                                added += 1
                                print(f"[{jid}] Downloaded image {i} ({len(r.content)} bytes)")
                            else:
                                print(f"[{jid}] Failed to download image {i}: HTTP {r.status_code}")
                    except Exception as ie:
                        print(f"[{jid}] Error processing image {i}: {ie}")

        print(f"[{jid}] Zip created with {added} images")
        if added < 3:
            return JSONResponse({"error": f"Only {added} images could be processed. Need at least 3."}, status_code=400)

        zip_buffer.seek(0)
        zip_path = UPLOAD_DIR / f"{jid}_training.zip"
        zip_path.write_bytes(zip_buffer.read())
        print(f"[{jid}] Zip size: {zip_path.stat().st_size} bytes")

        # Upload zip to fal.ai storage
        zip_url = fal_upload(str(zip_path))

        # Submit LoRA training job
        set_job(jid, {"status": "running", "progress": 0,
                      "message": "Submitting training job to fal.ai..."})

        # Run training asynchronously
        asyncio.create_task(run_lora_training(jid, zip_url, trigger_word, persona_name))
        return {"job_id": jid, "status": "running",
                "message": "LoRA training started — takes 15-20 minutes"}

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[{jid}] train-lora error: {tb}")
        return JSONResponse({"error": str(e), "detail": tb[-500:]}, status_code=500)

async def run_lora_training(jid, zip_url, trigger_word, persona_name):
    import concurrent.futures
    loop = asyncio.get_event_loop()
    # Run blocking fal_client.subscribe in a thread pool so FastAPI stays responsive
    await loop.run_in_executor(
        None,
        lambda: _lora_training_thread(jid, zip_url, trigger_word, persona_name)
    )

def _lora_training_thread(jid, zip_url, trigger_word, persona_name):
    """Runs in a thread — blocking fal_client.subscribe won't freeze FastAPI."""
    try:
        import fal_client
        os.environ["FAL_KEY"] = FAL_KEY
        print(f"[{jid}] Starting LoRA training for {persona_name}, trigger: {trigger_word}")
        set_job(jid, {"status": "running", "progress": 5,
                      "message": "Training submitted — waiting for GPU..."})

        def on_update(update):
            try:
                logs = getattr(update, 'logs', []) or []
                last_log = logs[-1].get('message', '') if logs else ''
                progress = 10
                if last_log:
                    import re as _re
                    m = _re.search(r'(\d+)%|step\s+(\d+)/(\d+)', last_log, _re.IGNORECASE)
                    if m:
                        if m.group(1):
                            progress = min(int(m.group(1)), 95)
                        elif m.group(2):
                            progress = min(int(int(m.group(2))/int(m.group(3))*90)+5, 95)
                msg = last_log or 'Training in progress...'
                set_job(jid, {"status": "running", "progress": progress, "message": msg})
                print(f"[{jid}] {progress}% — {msg[:80]}")
            except Exception as ue:
                print(f"[{jid}] Update error: {ue}")

        result = fal_client.subscribe(
            "fal-ai/flux-lora-fast-training",
            arguments={
                "images_data_url": zip_url,
                "trigger_word": trigger_word,
                "steps": 1000,
                "rank": 16,
                "learning_rate": 0.0004,
                "batch_size": 1,
                "resolution": "512,768,1024",
                "caption_dropout_rate": 0.05,
                "text_encoder_learning_rate": 0.0001,
                "create_masks": True
            },
            with_logs=True,
            on_queue_update=on_update
        )

        print(f"[{jid}] Result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
        lora_url = (result.get("diffusers_lora_file") or {}).get("url") or \
                   (result.get("lora_file") or {}).get("url") or \
                   result.get("url", "")

        print(f"[{jid}] LoRA complete. URL: {lora_url}")
        set_job(jid, {
            "status": "complete", "progress": 100,
            "message": "Identity locked — LoRA trained successfully",
            "lora_url": lora_url,
            "trigger_word": trigger_word
        })

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[{jid}] LoRA error: {tb}")
        set_job(jid, {"status": "error", "error": str(e), "detail": tb[-300:]})

@app.get("/lora-status/{job_id}")
async def lora_status(job_id: str):
    return get_job(job_id)

@app.post("/generate-outfits")
async def generate_outfits(
    persona_name: str = Form(...),
    persona_age: str = Form(...),
    niche: str = Form("general"),
    style_context: str = Form("")
):
    from groq import Groq
    client = Groq(api_key=GROQ_KEY)

    system_prompt = """You are a professional wardrobe stylist and image consultant for British AI influencer personas.
Generate outfit descriptions suitable for professional short-form video content.
Each outfit must:
- Be specific and visual — describe the garment, colour, fabric feel, and one accessory
- Match the persona's niche, authority level and brand identity
- Look professional and appropriate for the target audience
- Be distinct from each other — different colours, styles, formality levels
- Sound like something a real British professional would wear on camera
Return ONLY a JSON array of exactly 5 outfit strings. No markdown, no explanation."""

    user_prompt = f"""Generate 5 distinct outfit options for {persona_name}, aged {persona_age}, British, in the {niche} niche.
Style context from their skill file: {style_context[:500] if style_context else 'Professional British influencer'}

Each outfit description should be 10-20 words covering: main garment + colour + one key accessory or styling detail.
Example format: "Cream structured blazer over black silk top, single diamond pendant, hair swept back"

Return: ["outfit1", "outfit2", "outfit3", "outfit4", "outfit5"]"""

    try:
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=400,
            temperature=0.8
        )
        raw = response.choices[0].message.content.strip()
        if "<think>" in raw:
            raw = raw[raw.rfind("</think>")+8:].strip()
        raw = raw.replace("```json","").replace("```","").strip()
        outfits = json.loads(raw)
        if not isinstance(outfits, list):
            outfits = [outfits]
        outfits = [o.strip().strip('"').strip("'") for o in outfits if o and len(o.strip())>5]
        return {"outfits": outfits[:5]}
    except Exception as e:
        # Fallback outfits based on niche
        fallbacks = {
            "wealth": ["Black tailored blazer, diamond necklace, dark silk top","Cream structured blazer, pearl earrings, ivory blouse","Charcoal power suit, minimal gold jewellery","Navy double-breasted blazer, silk shirt, understated watch","White blazer, black top, sleek updo"],
            "interior design": ["Cream silk blouse, tortoiseshell glasses, delicate gold necklace","Camel structured blazer, minimal jewellery, elegant updo","Deep teal blazer, pearl earrings, refined makeup","Ivory fine knit, gold earrings, effortless styling","Burgundy silk top, understated necklace, natural makeup"],
            "property": ["Dark navy jacket, open collar shirt, no tie","Charcoal blazer, quality checked shirt","Dark suit jacket, white shirt","Navy overcoat, smart casual shirt","Dark jacket, crew neck, understated"],
            "health": ["Clean white linen shirt, natural minimal look","Sage green top, fresh natural makeup","Cream athleisure top, dewy skin","Soft grey knit, understated earrings","White and cream layers, clean appearance"],
        }
        niche_key = next((k for k in fallbacks if k in niche.lower()), "wealth")
        return {"outfits": fallbacks[niche_key]}

@app.post("/generate-portrait")
async def generate_portrait(
    persona_name: str = Form(...),
    persona_age: str = Form(...),
    appearance: str = Form(...),
    outfit: str = Form("professional blazer"),
    count: int = Form(1),
    lora_url: Optional[str] = Form(None),
    trigger_word: Optional[str] = Form(None)
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

    # Strip any framing words from skill file appearance so we control framing
    import re as _re
    base = appearance
    base = _re.sub(
        r'\b(upper body|head\s?shot|bust shot|close.?up|shoulder up|chest up|85mm|f2\.8|f/2\.8)\b',
        '', base, flags=_re.IGNORECASE
    ).strip().strip(',').strip()

    # Extract core description — remove existing studio/camera language we'll rebuild
    core = _re.sub(
        r'(studio portrait photograph of a \d+ year old\s*|'
        r'canon eos.*?(?:,|$)|seamless white studio backdrop.*?(?:,|$)|'
        r'in the middle.*?(?:,|$)|no retouching.*?(?:,|$)|'
        r'single large softbox.*?(?:,|$)|white reflector.*?(?:,|$))',
        '', base, flags=_re.IGNORECASE
    ).strip().strip(',').strip()

    if not core or len(core) < 20:
        core = base  # fallback to full appearance if stripping went too far

    # FLUX framing rule: lead with framing, use 35mm for wider shot that captures waist
    # 35mm lens gives a wider field of view than 50mm — person appears further back
    # "three-quarter length" is the photography term for waist-to-top-of-head framing
    prompt = (
        f"Professional portrait photograph of a {persona_age} year old {core}, "
        f"framed from mid-chest upward showing face, neck, shoulders and upper chest, "
        f"subject looking directly at camera, confident natural expression, "
        f"{age_desc}, "
        f"{outfit}, "
        f"pure white seamless studio backdrop, subject centred in frame, "
        f"shot on Canon EOS 5D Mark IV with 85mm f/1.8 lens, "
        f"large softbox key light at 45 degrees camera left, white fill reflector right, "
        f"natural skin texture, visible pores, no retouching, no airbrushing, "
        f"commercial portrait photography, Getty Images editorial style"
    )

    # If LoRA is provided, inject trigger word at start of prompt
    if lora_url and trigger_word:
        prompt = f"{trigger_word}, {prompt}"
        print(f"Using LoRA: {lora_url}, trigger: {trigger_word}")

    images = []
    try:
        for i in range(min(count, 4)):
            args = {
                "prompt": prompt,
                "image_size": "portrait_4_3",
                "num_inference_steps": 35,
                "num_images": 1,
                "enable_safety_checker": False,
                "guidance_scale": 3.5
            }
            # Add LoRA weights if provided
            if lora_url:
                args["loras"] = [{"path": lora_url, "scale": 1.0}]

            result = fal_client.subscribe("fal-ai/flux/dev", arguments=args)
            img_url = result["images"][0]["url"]
            # Return fal.ai CDN URL directly — permanent, survives redeploys
            # Also save locally as cache for faster serving
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.get(img_url)
                img_path = OUTPUT_DIR / f"{jid}_portrait_{i}.png"
                img_path.write_bytes(r.content)
            except Exception:
                pass  # Local cache is optional
            images.append(img_url)  # Always return CDN URL
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
        # Save locally as cache
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get(img_url)
            img_path = OUTPUT_DIR / f"{jid}_scene.png"
            img_path.write_bytes(r.content)
        except Exception:
            pass
        # Return CDN URL directly — permanent
        return {"job_id": jid, "status": "complete", "image_url": img_url}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/scene/{jid}")
async def serve_scene(jid: str):
    path = OUTPUT_DIR / f"{jid}_scene.png"
    if path.exists():
        return FileResponse(str(path), media_type="image/png")
    return JSONResponse({"error": "not found"}, status_code=404)

def generate_srt(script: str, duration: float) -> str:
    """Generate a timed SRT subtitle file from script text and audio duration.
    Splits into lines of ~8 words, evenly timed across the duration.
    """
    import re
    # Clean script
    text = re.sub(r'\s+', ' ', script.strip())
    words = text.split()
    if not words:
        return ""

    # Group into caption lines — ~7 words per line for readability
    words_per_line = 7
    lines = []
    for i in range(0, len(words), words_per_line):
        lines.append(' '.join(words[i:i+words_per_line]))

    # Calculate timing — evenly distribute duration across lines
    time_per_line = duration / len(lines)

    def fmt_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    srt = ""
    for i, line in enumerate(lines):
        start = i * time_per_line
        end = (i + 1) * time_per_line - 0.05  # small gap between captions
        srt += f"{i+1}\n{fmt_time(start)} --> {fmt_time(end)}\n{line}\n\n"

    return srt

def get_caption_ffmpeg_filter(style: str, ratio: str, srt_path: str) -> str:
    """Return ffmpeg subtitle filter string for each caption style."""
    srt_escaped = srt_path.replace('\\', '/').replace(':', '\\:')

    # Position: bottom for portrait, lower-third for landscape
    is_landscape = ratio == "16:9"
    y_pos = "h-th-60" if is_landscape else "h-th-120"

    styles = {
        "tiktok": (
            f"subtitles='{srt_escaped}':force_style='"
            f"FontName=Arial,FontSize=22,Bold=1,PrimaryColour=&H00FFFFFF,"
            f"OutlineColour=&H00000000,Outline=3,Shadow=1,"
            f"Alignment=2,MarginV=80'"
        ),
        "minimal": (
            f"subtitles='{srt_escaped}':force_style='"
            f"FontName=Arial,FontSize=16,Bold=0,PrimaryColour=&H00FFFFFF,"
            f"OutlineColour=&H00000000,Outline=2,Shadow=0,"
            f"Alignment=2,MarginV=60'"
        ),
        "highlight": (
            f"subtitles='{srt_escaped}':force_style='"
            f"FontName=Arial,FontSize=20,Bold=1,PrimaryColour=&H00FFFFFF,"
            f"BackColour=&H80000000,BorderStyle=4,"
            f"Outline=0,Shadow=0,Alignment=2,MarginV=80'"
        ),
        "broadcast": (
            f"subtitles='{srt_escaped}':force_style='"
            f"FontName=Arial,FontSize=18,Bold=1,PrimaryColour=&H00FFFFFF,"
            f"BackColour=&HCC000000,BorderStyle=4,"
            f"Outline=0,Shadow=0,Alignment=2,MarginV=40'"
        ),
        "cinematic": (
            f"subtitles='{srt_escaped}':force_style='"
            f"FontName=Georgia,FontSize=18,Bold=0,Italic=1,"
            f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            f"Outline=2,Shadow=1,Alignment=2,MarginV=100'"
        ),
    }
    return styles.get(style, styles["tiktok"])

@app.post("/generate-video")
async def generate_video(
    portrait: UploadFile = File(...),
    audio: UploadFile = File(...),
    scene: Optional[UploadFile] = File(None),
    job_id: str = Form(None),
    ratio: str = Form("9:16"),
    out_width: int = Form(512),
    out_height: int = Form(768),
    script: Optional[str] = Form(None),
    caption_style: Optional[str] = Form(None)  # none, tiktok, minimal, highlight, broadcast, cinematic
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
    asyncio.create_task(run_video_pipeline(
        jid, str(portrait_path), str(audio_path), scene_path,
        ratio, out_width, out_height, script, caption_style
    ))
    return {"job_id": jid, "status": "running"}

async def run_video_pipeline(jid, portrait_path, audio_path, scene_path=None, ratio="9:16", out_width=512, out_height=768, script=None, caption_style=None):
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

        # Crop portrait to top 45% before sending to lip-sync model
        # Full waist-up portrait → Flashtalk crops to face only (too tight)
        # Top 45% = head + neck + shoulders + upper chest → Flashtalk output shows this naturally
        orig_w, orig_h = portrait_rgba.size
        crop_h = int(orig_h * 0.45)
        portrait_rgba = portrait_rgba.crop((0, 0, orig_w, crop_h))
        print(f"[{jid}] Cropped to upper body: {orig_w}x{crop_h} (was {orig_w}x{orig_h})")

        if scene_path and Path(scene_path).exists():
            set_job(jid, {"status": "running", "stage": "compositing", "progress": 20})
            print(f"[{jid}] Compositing...")
            bg = Image.open(scene_path).convert("RGBA")
            bg = bg.resize(portrait_rgba.size, Image.LANCZOS)
            composite = Image.new("RGBA", portrait_rgba.size, (0, 0, 0, 255))
            composite.paste(bg, (0, 0))
            composite.paste(portrait_rgba, (0, 0), portrait_rgba)
            final_portrait_path = str(UPLOAD_DIR / f"{jid}_composite.png")
            composite.convert("RGB").save(final_portrait_path)
            print(f"[{jid}] Composite done.")
        else:
            bg = Image.new("RGB", portrait_rgba.size, (0, 0, 0))
            bg.paste(portrait_rgba, (0, 0), portrait_rgba)
            final_portrait_path = str(UPLOAD_DIR / f"{jid}_nobg.png")
            bg.save(final_portrait_path)
            print(f"[{jid}] Portrait on black background saved.")

        set_job(jid, {"status": "running", "stage": "uploading", "progress": 30,
                      "message": "Uploading portrait and audio to fal.ai..."})
        print(f"[{jid}] Uploading portrait and audio...")
        portrait_url = fal_client.upload(Path(final_portrait_path).read_bytes(), "image/png")
        audio_url = fal_client.upload(Path(audio_path).read_bytes(), "audio/mpeg")
        print(f"[{jid}] Uploaded. Portrait: {portrait_url[:50]}...")

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

        # ── FACE COMPOSITE ────────────────────────────────────────────
        # Flashtalk outputs a talking-head video cropped to face area.
        # Strategy: scale Flashtalk output to fill target dimensions,
        # then overlay the ORIGINAL full portrait (with scene) as a static background
        # behind it — this gives us the full body in the background with
        # the animated face in the foreground, scaled to fill the frame.
        # For 9:16/4:5/1:1: the Flashtalk video IS the video — scale to fill.
        # The portrait framing should be handled at generation time (prompt).
        # We keep this simple: scale Flashtalk to fill target, pad if needed.

        set_job(jid, {"status": "running", "stage": "processing", "progress": 88,
                      "message": f"Encoding {ratio} video for streaming..."})
        final_path = str(OUTPUT_DIR / f"{jid}_final.mp4")

        # ── FFmpeg compositing strategy per ratio ──────────────────────────
        # Flashtalk outputs 768x448 landscape talking-head video.
        # Strategy: scale-to-fill (crop) the persona to the target dimensions.
        # For 16:9 with a scene: composite persona over scene background at full size.
        # grain filter appended to all.
        grain = "noise=alls=12:allf=t+u,unsharp=3:3:1.2:3:3:0.0,eq=contrast=1.05:brightness=-0.01:saturation=0.95"

        # Dimensions per ratio
        ratio_dims = {
            "9:16":  (1080, 1920),
            "1:1":   (1080, 1080),
            "4:5":   (1080, 1350),
            "16:9":  (1920, 1080),
        }
        W, H = ratio_dims.get(ratio, (1080, 1920))

        # For 16:9 with a scene: overlay persona on full-frame scene background
        has_scene = scene_path and Path(scene_path).exists()

        # Flashtalk outputs 768x448 landscape — face is centred in frame
        # Strategy per ratio:
        # 9:16  → scale to 1080 wide, pad height to 1920 (black bars top/bottom acceptable for portrait)
        # 1:1   → scale to fit 1080x1080, pad sides
        # 4:5   → scale to fit 1080x1350, pad
        # 16:9  → scale persona to fill 1920x1080 height (upscale 768x448 → 1920x1080 keeping AR)
        #          overlay centred on blurred scene background for broadcast look

        if ratio == "16:9" and has_scene:
            scene_input = str(scene_path)
            final_path_tmp = str(OUTPUT_DIR / f"{jid}_final.mp4")

            # Scale persona to fill 1920 wide while keeping aspect ratio
            # 768x448 → scale to 1920 wide → height becomes 1920*(448/768) = 1120
            # Then centre vertically on 1080 canvas (crop 20px top+bottom)
            # This keeps the face in frame and fills the width entirely
            filter_complex = (
                # Background: scene image scaled to fill 1920x1080, heavily blurred
                f"[0:v]scale=1920:1080:force_original_aspect_ratio=increase,"
                f"crop=1920:1080,gblur=sigma=20[bg];"
                # Persona: scale to fill width (1920), keeping aspect ratio
                f"[1:v]scale=1920:-2[fg_scaled];"
                # Composite: overlay persona centred vertically on background
                f"[bg][fg_scaled]overlay=0:(H-h)/2,"
                f"{grain}[out]"
            )

            ret = subprocess.run([
                "ffmpeg", "-y",
                "-loop", "1", "-i", scene_input,
                "-i", raw_path,
                "-filter_complex", filter_complex,
                "-map", "[out]",
                "-map", "1:a",
                "-c:v", "libx264", "-crf", "22", "-preset", "fast",
                "-profile:v", "baseline", "-level", "4.0",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
                "-movflags", "+faststart",
                "-maxrate", "6M", "-bufsize", "12M",
                "-shortest",
                final_path_tmp
            ], capture_output=True)
            final_path = final_path_tmp

        elif ratio == "16:9":
            # No scene — scale persona to fill 1920x1080, black background
            # Upscale 768x448 to 1920 wide, centre on 1080 canvas
            vf = (
                f"scale=1920:-2,"
                f"pad=1920:1080:0:(oh-ih)/2:black,"
                f"{grain}"
            )
            ret = subprocess.run([
                "ffmpeg", "-y", "-i", raw_path,
                "-vf", vf,
                "-c:v", "libx264", "-crf", "22", "-preset", "fast",
                "-profile:v", "baseline", "-level", "4.0",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
                "-movflags", "+faststart",
                "-maxrate", "6M", "-bufsize", "12M",
                final_path
            ], capture_output=True)

        else:
            # 9:16, 1:1, 4:5 — scale to fit target, pad with black (no cropping = no face cutoff)
            vf = (
                f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
                f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,"
                f"{grain}"
            )
            ret = subprocess.run([
                "ffmpeg", "-y", "-i", raw_path,
                "-vf", vf,
                "-c:v", "libx264", "-crf", "23", "-preset", "fast",
                "-profile:v", "baseline", "-level", "3.1",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "96k", "-ar", "44100",
                "-movflags", "+faststart",
                "-maxrate", "2M", "-bufsize", "4M",
                final_path
            ], capture_output=True)
        print(f"[{jid}] ffmpeg return code: {ret.returncode}")
        # Always print full stderr so we can diagnose failures
        stderr_out = ret.stderr.decode(errors='replace')
        if ret.returncode != 0:
            print(f"[{jid}] ffmpeg FAILED:\n{stderr_out[-2000:]}")
        else:
            # Print last few lines even on success
            print(f"[{jid}] ffmpeg ok: {stderr_out.splitlines()[-1] if stderr_out else 'done'}")

        # Validate ffmpeg output — fall back to raw if empty or missing
        final_size = Path(final_path).stat().st_size if Path(final_path).exists() else 0
        if final_size < 10000:
            print(f"[{jid}] ffmpeg output too small ({final_size} bytes) — using raw video instead")
            final_path = raw_path

        # ── CAPTION BURNING ────────────────────────────────────────────
        if caption_style and caption_style != "none" and script:
            set_job(jid, {"status": "running", "stage": "captions", "progress": 93,
                          "message": "Burning captions into video..."})
            try:
                # Get audio duration
                probe = subprocess.run([
                    "ffprobe", "-v", "quiet", "-print_format", "json",
                    "-show_format", str(audio_path)
                ], capture_output=True)
                probe_data = json.loads(probe.stdout.decode())
                duration = float(probe_data.get("format", {}).get("duration", 20.0))

                # Generate SRT from script
                srt_content = generate_srt(script, duration)
                srt_path = str(UPLOAD_DIR / f"{jid}_captions.srt")
                Path(srt_path).write_text(srt_content, encoding='utf-8')

                # Burn captions into video
                captioned_path = str(OUTPUT_DIR / f"{jid}_captioned.mp4")
                caption_filter = get_caption_ffmpeg_filter(caption_style, ratio, srt_path)

                ret_cap = subprocess.run([
                    "ffmpeg", "-y", "-i", final_path,
                    "-vf", caption_filter,
                    "-c:v", "libx264", "-crf", "23", "-preset", "fast",
                    "-profile:v", "baseline", "-level", "3.1",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "copy",
                    "-movflags", "+faststart",
                    captioned_path
                ], capture_output=True)

                if ret_cap.returncode == 0 and Path(captioned_path).stat().st_size > 10000:
                    final_path = captioned_path
                    print(f"[{jid}] Captions burned successfully ({caption_style})")
                else:
                    print(f"[{jid}] Caption burning failed: {ret_cap.stderr.decode()[:300]}")
            except Exception as e:
                print(f"[{jid}] Caption error (non-fatal): {e}")

        upload_size = Path(final_path).stat().st_size
        print(f"[{jid}] Uploading {upload_size} bytes to fal.ai CDN...")
        set_job(jid, {"status": "running", "stage": "uploading_cdn", "progress": 95,
                      "message": "Uploading to CDN for permanent storage..."})

        # Upload bytes directly to avoid storage_type=gcs 400 error
        video_bytes = Path(final_path).read_bytes()
        final_url = fal_client.upload(video_bytes, "video/mp4")
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
