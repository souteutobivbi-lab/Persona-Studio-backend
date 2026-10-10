import asyncio, os, random, subprocess, uuid, json
from pathlib import Path
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import httpx
from supabase import create_client

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GROQ_KEY       = os.environ.get("GROQ_KEY", "")
ELEVENLABS_KEY = os.environ.get("ELEVENLABS_KEY", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
sb = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
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


@app.get("/robots.txt")
async def robots():
    return HTMLResponse("User-agent: *\nDisallow: /\n", media_type="text/plain")

@app.get("/health")
async def health():
    return {"status": "ok", "version": "3.0", "host": "railway"}

@app.get("/")
async def root():
    p = Path(__file__).parent / "static" / "index.html"
    if p.exists():
        content = p.read_text()
        content = content.replace('<head>', '<head><meta name="robots" content="noindex,nofollow">')
        return HTMLResponse(content)
    return {"api": "Persona Studio v3.0"}

from fastapi.staticfiles import StaticFiles
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/assets", StaticFiles(directory=str(static_path)), name="static")

MUSIC_DIR = Path(__file__).parent / "static" / "music"
MUSIC_TRACKS = {
    "village-hearth-harp":  {"file": "village-hearth-harp.mp3",  "label": "Village Hearth ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â warm harp"},
    "warm-drone-ambience":  {"file": "warm-drone-ambience.mp3",   "label": "Warm Drone ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â meditation ambience"},
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
- role: their role and niche (e.g. "Transformation Authority ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚ -  Wealth")
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

@app.post("/generate-topics")
async def generate_topics(
    persona_name: str = Form(...),
    niche: str = Form(...),
    skill_context: str = Form(""),
    count: str = Form("8")
):
    from groq import Groq
    client = Groq(api_key=GROQ_KEY)
    skill_hint = skill_context[:600] if skill_context else ""
    prompt = f"""Generate 8 specific, emotionally resonant content topic ideas for {persona_name}, a creator in the {niche} niche.

{f'Persona context: {skill_hint}' if skill_hint else ''}

Rules:
- Each topic must be specific, not vague (e.g. "Why you go back to the person who hurt you" NOT "relationships")
- Tap into something the audience is quietly struggling with right now
- 6-14 words each, conversational phrasing, no hashtags
- Make all 8 feel completely different from each other

Return ONLY a JSON array of 8 topic strings. No explanation, no numbering.
["Topic one", "Topic two", ...]"""
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[{"role": "system", "content": "ABSOLUTE RULE: Never begin any script with the phrase A man who or A woman who. Always write in first person (I, my, we) as the persona. Never describe a third-person character."}, {"role": "user", "content": prompt}],
        max_tokens=350, temperature=0.97
    )
    raw = response.choices[0].message.content.strip()
    try:
        import re as _re
        raw = _re.sub(r'<think>.*?</think>', '', raw, flags=_re.DOTALL).strip()
        raw = raw.replace('```json','').replace('```','').strip()
        s = raw.find('['); e = raw.rfind(']')
        if s != -1 and e != -1: raw = raw[s:e+1]
        topics = json.loads(raw)
        if not isinstance(topics, list): topics = []
    except:
        topics = []
    return {"topics": topics}

@app.get("/voices")
async def get_voices():
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get("https://api.elevenlabs.io/v1/voices",
            headers={"xi-api-key": ELEVENLABS_KEY})
    if r.status_code != 200:
        return JSONResponse({"error": "ElevenLabs error"}, status_code=400)
    voices = r.json().get("voices", [])
    custom_voice = {"id": "mGzBE5uXRa6uy9ilKVOD", "name": "Soute - CEO", "description": "Custom cloned voice", "is_cloned": True}
    vlist = [{"id": v["voice_id"], "name": v["name"],
              "description": ", ".join(v.get("labels", {}).values()),
              "is_cloned": v.get("category") == "cloned"} for v in voices]
    if not any(v["id"] == "mGzBE5uXRa6uy9ilKVOD" for v in vlist):
        vlist.insert(0, custom_voice)
    return sorted(vlist, key=lambda x: (0 if x["is_cloned"] else 1))

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

    # Detect Vivienne-style skill vs generic ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â look for structure markers
    has_vivienne_structure = False
    has_ceo_structure = False
    _ceo_topic_kws = ["nutlip","homivis","mediahouz","ceo","founder","launch","platform","investors","funding","startup","building","serial","proptech","yorkshire","second launch"]
    _ceo_skill_kws = ["nutlip","homivis","mediahouz","serial tech founder","building in public"]
    if niche.lower() in ["motivation","entrepreneurship"] or any(k in topic.lower() for k in _ceo_topic_kws) or any(k in (skill_context or "").lower() for k in _ceo_skill_kws):
        has_ceo_structure = True  # CEO/founder always uses CEO path

    structure_rules = ""
    script_examples = ""
    signoffs = ""
    if skill_context and not has_ceo_structure:
        sc = skill_context
        has_vivienne_structure = ("SCRIPT STRUCTURE" in sc or "podcast confession" in sc.lower()
                                   or "EXAMPLE SCRIPTS" in sc or "The Claim" in sc)

        def extract_section(text, heading):
            """Extract content between a heading and the next heading/separator."""
            idx = text.find(heading)
            if idx == -1:
                return ""
            rest = text[idx + len(heading):]
            # Find next section heading (ALL CAPS word followed by newline, or separator line)
            import re as _re
            m = _re.search(r'\n(?=[A-Z][A-Z ]{3,}\n|[-=─-╿]{4,})', rest)
            end = m.start() if m else len(rest)
            return rest[:end].strip()

        if "SCRIPT STRUCTURE" in sc:
            structure_rules = extract_section(sc, "SCRIPT STRUCTURE")
        if "EXAMPLE SCRIPTS" in sc:
            script_examples = extract_section(sc, "EXAMPLE SCRIPTS")[:2000]
        if "SIGN-OFFS" in sc:
            signoffs = extract_section(sc, "SIGN-OFFS")


    # Universal prompt builder - skill sheet drives everything
    skill_block = ""
    if skill_context:
        skill_block = (
            "\n\n" + chr(9473)*39 + "\n"
            "PERSONA SKILL SHEET - READ AND FOLLOW:\n" +
            chr(9473)*39 + "\n" +
            (skill_context[:4000]) + "\n" +
            chr(9473)*39 + "\n"
            "The skill sheet above defines this persona's voice, structure, tone, and examples.\n"
            "Follow it exactly. It overrides any generic defaults below."
        )

    ceo_note = ""
    if has_ceo_structure:
        ceo_note = (
            "\nPERSONA TYPE: Serial tech founder / CEO. Always write in first person (I, my, we).\n"
            "NEVER describe this person from the outside. NEVER open with A man who or A woman who.\n"
        )

    prompt = (
        f"You are writing 5 short-form video scripts for {persona_name}, aged {persona_age}, "
        f"in the {niche} niche.\nTopic: {topic}\n"
        + ceo_note + skill_block +
        "\n\nUNIVERSAL RULES (apply to every script regardless of persona):\n\n"
        "1. HOOK - the first 1-2 sentences must stop the scroll. Rotate these types across 5 scripts:\n"
        "   - Provocative question  e.g. Why do the people who give the most always end up with the least?\n"
        "   - Bold confronting truth  e.g. Nobody tells you that healing feels like grief before it feels like freedom.\n"
        "   - Specific moment  e.g. The morning I stopped explaining myself, everything changed.\n"
        "   - Number or fact  e.g. Three investors said the same thing to me in one week.\n"
        "   - Confession  e.g. I nearly shut everything down in month eight.\n\n"
        "2. MIDDLE - deliver the real value. Specific, earned, no waffle. "
        "Follow the skill sheet structure if one is provided.\n\n"
        "3. REWARD - last 1-2 sentences: the payoff that makes the whole watch worthwhile. "
        "Then one earned sign-off that matches the persona voice.\n\n"
        "LENGTH: 100-140 words per script. Paragraph style only.\n"
        "No lists, no hashtags, no emojis, no stage directions.\n"
        "Each of the 5 scripts must cover a DIFFERENT angle of the topic.\n\n"
        "QUESTION HOOK RULE: When you open with a question, do NOT answer it in the next sentence. "
        "Let it hang. The answer should be earned through the middle and only land in the REWARD at the end.\n"
        "REPETITION RULE: No phrase, sentence, or line may appear in more than ONE of the 5 scripts. "
        "If a line is a persona signature, use it in one script only. Every script must feel distinct.\n\n"
        'Return ONLY a JSON array of 5 objects:\n'
        '[{"title": "Short searchable title", "script": "Full script text", "keywords": ["#tag1"]}]'
    )
    max_tok = 2500
    model = "qwen/qwen3.8-27b"

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tok, temperature=0.88
    )
    raw = response.choices[0].message.content.strip()
    import re as _re
    # Strip <think>...</think> reasoning blocks
    raw = _re.sub(r'<think>.*?</think>', '', raw, flags=_re.DOTALL).strip()
    raw = raw.replace('```json','').replace('```','').strip()
    # Extract JSON array
    start = raw.find('[')
    end = raw.rfind(']')
    if start != -1 and end != -1:
        raw = raw[start:end+1]
    # Fix common model quirks: trailing commas before ] or }
    raw = _re.sub(r',\s*}', '}', raw)
    raw = _re.sub(r',\s*]', ']', raw)

    scripts = []
    all_keywords = []
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, list): parsed = [parsed]
        if parsed and isinstance(parsed[0], dict) and 'script' in parsed[0]:
            scripts = [item['script'] for item in parsed if isinstance(item, dict)]
            all_titles = [item.get('title', f'Script {j+1}') for j, item in enumerate(parsed) if isinstance(item, dict)]
            all_keywords = [item.get('keywords', []) for item in parsed if isinstance(item, dict)]
        else:
            scripts = [str(item) for item in parsed]
            all_keywords = [[] for _ in scripts]
    except Exception as parse_err:
        # Last resort: pull out individual script strings with regex
        found = _re.findall(r'"script"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
        kw_found = _re.findall(r'"keywords"\s*:\s*\[([^\]]*)\]', raw)
        if found:
            scripts = [s.replace('\\"','"').replace('\\n','\n') for s in found]
            all_keywords = []
            for kblock in kw_found:
                kws = _re.findall(r'"([^"]+)"', kblock)
                all_keywords.append(kws)
            while len(all_keywords) < len(scripts):
                all_keywords.append([])
        else:
            scripts = []
            all_keywords = []

    # Filter out any malformed/too-short items
    valid = [(s, k, t) for s, k, t in zip(scripts, all_keywords, all_titles) if s and len(s.split()) >= 70]
    if valid:
        scripts, all_keywords, all_titles = zip(*valid)
        scripts, all_keywords, all_titles = list(scripts), list(all_keywords), list(all_titles)
    if not 'all_titles' in dir():
        all_titles = [f'Script {j+1}' for j in range(len(scripts))]
    elif not scripts:
        scripts = ["Script generation failed ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â please try again."]
        all_keywords = [[]]

    return {
        "titles": all_titles if "all_titles" in dir() else [f"Script {j+1}" for j in range(len(scripts))],
        "scripts": scripts,
        "keywords": all_keywords,
        "script": scripts[0] if scripts else "",
        "words": len(scripts[0].split()) if scripts else 0
    }


@app.post("/clone-voice")
async def clone_voice(
    audio: UploadFile = File(...),
    name: str = Form("Cloned Voice"),
    persona_name: str = Form("")
):
    audio_bytes = await audio.read()
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.elevenlabs.io/v1/voices/add",
            headers={"xi-api-key": ELEVENLABS_KEY},
            data={"name": name, "description": f"Cloned voice for {persona_name or name}"},
            files={"files": (audio.filename, audio_bytes, audio.content_type or "audio/mpeg")}
        )
    if r.status_code != 200:
        return JSONResponse({"error": f"ElevenLabs error: {r.text[:200]}"}, status_code=400)
    voice_id = r.json().get("voice_id")
    if not voice_id:
        return JSONResponse({"error": "No voice_id returned"}, status_code=400)
    return {"voice_id": voice_id, "name": name, "status": "cloned"}

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

def build_portrait_prompt(appearance: str, persona_age: str, outfit: str, outfit_color: str = "", sex: str = "female", skill_context: str = "") -> str:
    """Build podcast-style seated portrait prompt matching linakodi1_ / mayaa_speaks aesthetic.

    The appearance string from the persona is layered ON TOP of a hardcoded base that locks
    the podcast format, skin, body type and setting. This ensures the aesthetic never drifts
    regardless of what text is in the persona skill file.
    """
    color_hint = f"{outfit_color} " if outfit_color else ""

    # Derive gender word and sex flag
    _is_male = sex.lower() in ("male", "m", "man")
    gender_word = "man" if _is_male else "woman"

    # Base identity layer — branches by sex
    if _is_male:
        base_identity = (
            "deep rich warm dark brown skin, sharp angular jawline, high cheekbones, "
            "dark brown eyes, close-cropped or short natural hair, "
            "West African facial bone structure, "
            "broad shoulders, lean athletic build, strong upright posture, "
            "clean well-groomed appearance, confident composed expression"
        )
    else:
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

    # Parse PORTRAIT SCENE from skill file — handles both "PORTRAIT SCENE: x" and section header format
    custom_scene = ""
    if skill_context and "PORTRAIT SCENE" in skill_context:
        idx = skill_context.find("PORTRAIT SCENE")
        rest = skill_context[idx + len("PORTRAIT SCENE"):]
        for _line in rest.splitlines():
            _line = _line.strip()
            # Skip the colon-only line, dividers, and blank lines
            if not _line or _line == ":" or all(c in "━─═-=" for c in _line):
                continue
            # If inline (PORTRAIT SCENE: content), strip leading colon
            custom_scene = _line.lstrip(":").strip()
            break

    # Podcast set - use skill file scene if defined, else sex-based default
    if custom_scene:
        podcast_set = custom_scene
    elif sex.lower() in ('male', 'm', 'man'):
        podcast_set = (
            'seated almost side-on to camera in a dark leather podcast chair, body angled 45 degrees, '
            'leaning forward slightly with elbows resting on knees or a dark wood desk, '
            'large black condenser podcast microphone on boom arm positioned to one side, '
            'warm amber and gold studio lighting, moody low-key professional broadcast setup, '
            'dark charcoal or deep navy acoustic panel wall, subtle warm backlight creating depth, '
            'tight medium shot head to mid-chest, subject angled away from camera looking composed and direct, '
            'CEO thought-leader energy, no plants, no cream furniture, no bright white backgrounds'
        )
    else:
        podcast_set = (
            'seated sideways in a cream upholstered podcast armchair, body angled 45 degrees away from camera, legs crossed at knee, turned slightly to speak to someone off-camera to the left, natural conversational pose, '
            'hands resting naturally in lap, upper body and arms fully visible in frame, '
            'large black podcast microphone on boom arm positioned in front slightly to right, '
            'background: soft warm grey textured wall, '
            'large lush tropical areca palm plant with long green fronds behind right shoulder, '
            'bright professional studio lighting, key light and fill light setup, soft diffused light on face and shoulders, warm amber tones, high-key studio portrait lighting, shallow depth of field, bokeh background, '
            'camera angle at chest level, tight medium shot framing head to mid-torso, subject looking slightly off-camera to the left as if in conversation'
        )
    photo_tech = (
        "Sony A7R IV 85mm f1.4 portrait lens, "
        "natural skin texture visible pores real human skin, "
        "photorealistic authentic photography not airbrushed not CGI, "
        "sharp crisp face soft blurred background, "
        "professional editorial portrait photography, high resolution"
    )

    return (
        f"photorealistic portrait photograph, {persona_age} year old British-Nigerian {gender_word}, "
        f"{base_identity}, "
        f"{persona_layer}, "
        f"wearing a {color_hint}{outfit}, well-fitted, sharp and professional, "
        f"{podcast_set}, "
        f"{photo_tech}"
    )

def get_face_similarity(img_path_a: str, img_path_b: str) -> float:
    """Compare two face images using insightface buffalo_l. Returns cosine similarity 0.0ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ1.0.
    Used only as a post-generation quality check ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â never as a retry trigger for generation."""
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

# Persistent master portrait store ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â survives Railway restarts
# Saved to /app/master_portraits.json; fal CDN URLs are permanent
PORTRAITS_FILE = Path("/app/master_portraits.json")
LORAS_FILE = Path("/app/loras.json")

def load_lora_registry() -> dict:
    try:
        if LORAS_FILE.exists():
            return json.loads(LORAS_FILE.read_text())
    except Exception as e:
        print(f"Warning: could not load lora registry: {e}")
    return {}

def save_lora_registry(registry: dict):
    try:
        LORAS_FILE.write_text(json.dumps(registry, indent=2))
    except Exception as e:
        print(f"Warning: could not save lora registry to disk: {e}")
    if sb:
        try:
            for name, data in registry.items():
                url = data.get("fal_url") or data.get("url", "") if isinstance(data, dict) else str(data)
                trigger = data.get("trigger_word", name.lower()) if isinstance(data, dict) else name.lower()
                sb.table("loras").upsert(
                    {"persona_name": name, "lora_url": url, "trigger_word": trigger},
                    on_conflict="persona_name"
                ).execute()
            print(f"Saved {len(registry)} LoRAs to Supabase")
        except Exception as e:
            print(f"Supabase lora save error: {e}")

LORA_REGISTRY: dict = load_lora_registry()

# Hardcoded permanent LoRA URLs - updated at runtime when new LoRAs are trained
HARDCODED_LORAS = {
    "Vivienne": "https://v3b.fal.media/files/b/0aadab06/QU27nj5UoW1NIJ40q_O9S_vivienne_v1.safetensors",
}

def load_master_portraits() -> dict:
    if sb:
        try:
            rows = sb.table("portraits").select("*").execute().data
            if rows:
                result = {r["persona_name"]: {"url": r["portrait_url"]} for r in rows}
                print(f"Loaded {len(result)} portraits from Supabase: {list(result.keys())}")
                return result
        except Exception as e:
            print("Supabase portrait load error:", e)
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
        print(f"Warning: could not save master portraits to disk: {e}")
    if sb:
        try:
            for name, data in portraits.items():
                url = data.get("url", "") if isinstance(data, dict) else str(data)
                sb.table("portraits").upsert(
                    {"persona_name": name, "portrait_url": url},
                    on_conflict="persona_name"
                ).execute()
            print(f"Saved {len(portraits)} portraits to Supabase")
        except Exception as e:
            print(f"Supabase portrait save error: {e}")

MASTER_PORTRAITS: dict = load_master_portraits()  # persona_id ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ {"local": path, "url": fal_url}

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

    # Resolve master portrait URL ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â must be a public https URL for fal
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

    prompt = build_portrait_prompt(appearance, persona_age, outfit, outfit_color, sex=sex, skill_context=skill_context)
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
        # Scale to true 9:16 (720x1280) ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â flashtalk outputs landscape so we
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
# Outfit swap ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â preserves face, body & background; replaces only clothing
# Uses SAM2 for clothing segmentation mask + FLUX inpainting
# ---------------------------------------------------------------------------
# LoRA training from real uploaded frames (Pathway B)
# ---------------------------------------------------------------------------

@app.post("/train-lora-from-frames")
async def train_lora_from_frames(
    persona_id: str = Form(...),
    trigger_word: str = Form(...),
    appearance: str = Form(""),
    frames: list[UploadFile] = File(...),
):
    if not frames:
        return JSONResponse({"error": "No frames uploaded"}, status_code=400)
    if len(frames) < 5:
        return JSONResponse({"error": "Upload at least 5 images"}, status_code=400)

    jid = str(uuid.uuid4())[:8]
    set_job(jid, {"status": "running", "progress": 0, "stage": "uploading"})

    # Read all frame bytes eagerly (before background thread)
    frame_data = []
    for f in frames:
        data = await f.read()
        frame_data.append((f.filename or f"frame_{len(frame_data)}.jpg", data))

    def _run():
        try:
            import fal_client as _fal
            import zipfile, tempfile, pathlib, io

            tmp_dir = pathlib.Path(tempfile.mkdtemp())

            # ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ Write frames + captions to disk ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬
            set_job(jid, {"status": "running", "progress": 5, "stage": "preparing"})
            caption = (
                f"{trigger_word}, "
                + (appearance.split(',')[0].strip() if appearance else "portrait of a person")
                + ", natural expression, photorealistic"
            )

            img_dir = tmp_dir / "images"
            img_dir.mkdir()
            for fname, data in frame_data:
                stem = pathlib.Path(fname).stem
                (img_dir / fname).write_bytes(data)
                (img_dir / f"{stem}.txt").write_text(caption)

            # ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ Zip everything up ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬
            zip_path = tmp_dir / "training.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in img_dir.iterdir():
                    zf.write(p, p.name)

            set_job(jid, {"status": "running", "progress": 15, "stage": "uploading_zip"})
            zip_fal_url = _fal.upload_file(str(zip_path))
            print(f"[{jid}] Training zip uploaded ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ {zip_fal_url[:60]}")

            # ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ Submit LoRA training job ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬
            set_job(jid, {"status": "running", "progress": 20, "stage": "training"})
            print(f"[{jid}] Submitting LoRA training for '{trigger_word}' ({len(frame_data)} images)...")

            result = _fal.subscribe("fal-ai/flux-lora-fast-training", arguments={
                "images_data_url":   zip_fal_url,
                "trigger_word":      trigger_word,
                "steps":             1000,
                "learning_rate":     4e-4,
                "batch_size":        1,
                "resolution":        "512,768,1024",
                "caption_dropout_rate": 0.05,
                "is_style":          False,
            }, with_logs=True)

            lora_file = result.get("diffusers_lora_file") or {}
            lora_url  = lora_file.get("url") if isinstance(lora_file, dict) else lora_file
            if not lora_url:
                raise ValueError(f"LoRA training returned no weights: {result}")

            print(f"[{jid}] LoRA trained ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ {str(lora_url)[:60]}")

            # ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ Persist LoRA weights permanently on our server ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬
            # fal CDN URLs expire in ~24h. Download and store locally so
            # the weights survive restarts (Railway volume) or at least the
            # current deployment lifetime.
            import requests as _req2
            LORA_DIR = Path("/app/loras")
            LORA_DIR.mkdir(parents=True, exist_ok=True)
            # Name by trigger_word (unambiguous) not persona_id (can be wrong persona)
            safe_trigger = "".join(c for c in trigger_word if c.isalnum() or c in "-_")
            lora_local = LORA_DIR / f"{safe_trigger}.safetensors"
            try:
                dl = _req2.get(lora_url, headers={"Authorization": f"Key {FAL_KEY}"}, timeout=180)
                dl.raise_for_status()
                lora_local.write_bytes(dl.content)
                print(f"[{jid}] LoRA saved locally ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ {lora_local} ({len(dl.content)//1024}KB)")
                # Use a self-hosted URL so it never expires
                lora_serve_url = f"/lora-file/{safe_trigger}"
            except Exception as e:
                print(f"[{jid}] Warning: could not save LoRA locally ({e}), using fal URL")
                lora_serve_url = lora_url
            # Persist fal URL to registry so it survives Railway redeploys
            permanent_lora_url = _fal.upload_file(str(lora_local))
            # Save to Railway env var for permanent persistence
            try:
                import urllib.request as _ur
                _rw_token = os.environ.get("RAILWAY_TOKEN","")
                _rw_svc = os.environ.get("RAILWAY_SERVICE_ID","")
                _env_key = f"LORA_URL_{persona_id.upper().replace(' ','_')}"
                HARDCODED_LORAS[persona_id] = permanent_lora_url
                print(f"[{jid}] Saved LoRA URL to memory: {_env_key}={permanent_lora_url[:60]}")
            except Exception as _re:
                print(f"[{jid}] Could not save env var: {_re}")
            LORA_REGISTRY[persona_id] = {"local": str(lora_local), "fal_url": permanent_lora_url, "trigger_word": trigger_word}
            save_lora_registry(LORA_REGISTRY)

            # Clean up
            for f in img_dir.iterdir():
                try: f.unlink()
                except: pass
            try: img_dir.rmdir()
            except: pass
            try: zip_path.unlink()
            except: pass
            try: tmp_dir.rmdir()
            except: pass

            # Registry already saved above with correct fal_url key — duplicate removed

            set_job(jid, {
                "status":       "complete",
                "progress":     100,
                "lora_url":     permanent_lora_url if "permanent_lora_url" in locals() else lora_serve_url,
                "trigger_word": trigger_word,
                "persona_id":   persona_id,
            })
            print(f"[{jid}] LoRA training complete")

        except Exception as e:
            print(f"[{jid}] LoRA training error: {e}")
            set_job(jid, {"status": "error", "error": str(e)})

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": jid, "status": "running"}


@app.get("/lora-job/{jid}")
async def get_lora_job(jid: str):
    job = get_job(jid)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return job


@app.get("/lora-file/{persona_id}")
async def serve_lora_file(persona_id: str):
    """Serve the locally-stored LoRA weights file."""
    LORA_DIR = Path("/app/loras")
    safe_pid = "".join(c for c in persona_id if c.isalnum() or c in "-_")
    lora_path = LORA_DIR / f"{safe_pid}.safetensors"
    if not lora_path.exists():
        return JSONResponse({"error": f"LoRA file not found for '{persona_id}'"}, status_code=404)
    return FileResponse(str(lora_path), media_type="application/octet-stream",
                        filename=f"{safe_pid}.safetensors")


# ---------------------------------------------------------------------------

@app.post("/swap-outfit")
async def swap_outfit(
    persona_id: str = Form(...),
    outfit: str = Form(...),
    outfit_color: str = Form(""),
    master_url_override: str = Form(""),
    lora_url: str = Form(""),
    trigger_word: str = Form(""),
    appearance: str = Form(""),
    sex: str = Form("female"),
    skill_context: str = Form(""),
):
    """
    Swap outfit on the locked master portrait.
    If lora_url is provided: use FLUX Dev + LoRA (generates entire image ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â best quality).
    Otherwise: FASHN virtual try-on + smart composite fallback.
    Returns {job_id, status:"running"} immediately.
    Poll /portrait-status/{jid} for completion.
    """
    import fal_client
    os.environ["FAL_KEY"] = FAL_KEY

    entry = MASTER_PORTRAITS.get(persona_id)
    # Fallback: check Supabase if not in memory
    if not entry and sb:
        try:
            rows = sb.table("portraits").select("*").execute().data
            rows = [r for r in rows if r["persona_name"].lower() == persona_id.lower()]
            if rows:
                entry = {"url": rows[0]["portrait_url"]}
                MASTER_PORTRAITS[persona_id] = entry
                print(f"[swap_outfit] Loaded portrait for {persona_id} from Supabase")
        except Exception as e:
            print(f"[swap_outfit] Supabase portrait lookup error: {e}")
    # Accept master_url from the browser if server lost it (Railway ephemeral filesystem)
    master_url = (entry or {}).get("url") or master_url_override.strip() or None
    if not master_url:
        return JSONResponse({"error": f"No locked portrait for '{persona_id}' ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â lock a master portrait first"}, status_code=400)
    # Restore to in-memory dict so subsequent calls also work
    if not entry and master_url:
        MASTER_PORTRAITS[persona_id] = {"local": "", "url": master_url}
    jid = str(uuid.uuid4())[:8]
    set_job(jid, {"status": "running", "progress": 0, "stage": "starting"})

    outfit_prompt = f"{outfit_color} {outfit}".strip() if outfit_color else outfit
    use_lora = bool(lora_url.strip())

    def _run():
        try:
            import fal_client as _fal
            import requests as _req
            import tempfile, pathlib, io
            from PIL import Image, ImageFilter
            import numpy as np

            tmp_dir = pathlib.Path(tempfile.mkdtemp())

            # ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚Â
            # PATHWAY A ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â LoRA + FLUX Dev (best quality, full regeneration)
            # Used when the persona has a trained LoRA URL.
            # ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚Â
            if use_lora:
                set_job(jid, {"status": "running", "progress": 10, "stage": "lora_generating"})
                tw = trigger_word.strip() or "person"
                # Build appearance description (first sentence, no outfit mention)
                base_appearance = appearance.split(',')[0].strip() if appearance else "person"

                # Resolve LoRA weights to a local file or re-upload to fal
                raw_lora_url = lora_url.strip()
                set_job(jid, {"status": "running", "progress": 12, "stage": "preparing_lora"})
                lora_tmp = tmp_dir / "lora_weights.safetensors"

                if raw_lora_url.startswith("/lora-file/"):
                    # Locally stored ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â read directly from disk
                    pid_slug = raw_lora_url.split("/lora-file/")[-1]
                    safe_slug = "".join(c for c in pid_slug if c.isalnum() or c in "-_")
                    local_path = pathlib.Path("/app/loras") / f"{safe_slug}.safetensors"
                if not local_path.exists():
                    reg_entry = LORA_REGISTRY.get(persona_id) or LORA_REGISTRY.get(pid_slug)
                    if reg_entry and reg_entry.get("fal_url"):
                        print(f"[{jid}] Local LoRA missing, re-downloading from fal CDN...")
                        dl = _req.get(reg_entry["fal_url"], headers={"Authorization": f"Key {FAL_KEY}"}, timeout=180)
                        dl.raise_for_status()
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        local_path.write_bytes(dl.content)
                        print(f"[{jid}] LoRA restored from registry")
                    else:
                        # Hardcoded permanent fal URLs as last resort
                        HARDCODED_LORAS = {
                            "Vivienne": "https://v3b.fal.media/files/b/0aadab06/QU27nj5UoW1NIJ40q_O9S_vivienne_v1.safetensors",
                        }
                        hardcoded_url = HARDCODED_LORAS.get(persona_id)
                        if hardcoded_url:
                            print(f"[{jid}] Using hardcoded permanent LoRA URL for {persona_id}")
                            dl = _req.get(hardcoded_url, timeout=180)
                            dl.raise_for_status()
                            local_path.parent.mkdir(parents=True, exist_ok=True)
                            local_path.write_bytes(dl.content)
                            print(f"[{jid}] LoRA restored from hardcoded URL")
                        else:
                            # Try Supabase as final fallback
                            sb_url = None
                            if sb:
                                try:
                                    rows = sb.table("loras").select("*").execute().data
                                    rows = [r for r in rows if r["persona_name"].lower() == persona_id.lower()]
                                    if rows:
                                        sb_url = rows[0]["lora_url"]
                                        if sb_url and not sb_url.startswith("http"):
                                            print(f"[{jid}] Supabase lora_url is relative path, ignoring: {sb_url}")
                                            sb_url = None
                                        else:
                                            print(f"[{jid}] LoRA URL found in Supabase for {persona_id}")
                                except Exception as se:
                                    print(f"[{jid}] Supabase LoRA lookup error: {se}")
                            if sb_url:
                                dl = _req.get(sb_url, timeout=180)
                                dl.raise_for_status()
                                local_path.parent.mkdir(parents=True, exist_ok=True)
                                local_path.write_bytes(dl.content)
                                print(f"[{jid}] LoRA restored from Supabase URL")
                            else:
                                raise ValueError(f"LoRA not found locally and no registry entry for {persona_id} -- please retrain")
                    print(f"[{jid}] Using locally stored LoRA: {local_path}")
                    import shutil
                    shutil.copy(str(local_path), str(lora_tmp))
                else:
                    # fal CDN URL ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â download with auth header
                    print(f"[{jid}] Downloading LoRA from fal CDN...")
                    lora_resp = _req.get(
                        raw_lora_url,
                        headers={"Authorization": f"Key {FAL_KEY}"},
                        timeout=120,
                    )
                    lora_resp.raise_for_status()
                    lora_tmp.write_bytes(lora_resp.content)

                print(f"[{jid}] Uploading LoRA weights to fal...")
                fresh_lora_url = _fal.upload_file(str(lora_tmp))
                print(f"[{jid}] LoRA ready ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ {fresh_lora_url[:60]}")

                # Upload master portrait as image reference so pose/props are preserved
                set_job(jid, {"status": "running", "progress": 20, "stage": "uploading_reference"})
                tmp_portrait = tmp_dir / "portrait.png"
                img_resp = _req.get(master_url, timeout=30)
                img_resp.raise_for_status()
                tmp_portrait.write_bytes(img_resp.content)
                portrait_ref_url = _fal.upload_file(str(tmp_portrait))
                print(f"[{jid}] Portrait reference uploaded ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ {portrait_ref_url[:60]}")

                # Use sex form field directly — skill files don't always have SEX: line
                # Use build_portrait_prompt so PORTRAIT SCENE from skill file is respected
                portrait_prompt = build_portrait_prompt(
                    appearance=base_appearance,
                    persona_age="",
                    outfit=outfit_prompt,
                    outfit_color="",
                    sex=sex,
                    skill_context=skill_context,
                )
                # Prepend trigger word so LoRA recognises the face
                portrait_prompt = f"{tw}, {portrait_prompt}"
                print(f"[{jid}] LoRA pathway ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â prompt: {portrait_prompt[:120]}...")
                gen_result = _fal.subscribe("fal-ai/flux-lora", arguments={
                    "prompt": portrait_prompt,
                    "loras": [{"path": fresh_lora_url, "scale": 1.0}],
                    "image_url": portrait_ref_url,
                    "strength": 0.55,
                    "num_inference_steps": 40,
                    "guidance_scale": 3.5,
                    "num_images": 1,
                    "image_size": {"width": 768, "height": 1024},
                    "enable_safety_checker": False,
                    "output_format": "png",
                })
                gen_imgs = gen_result.get("images") or []
                if not gen_imgs:
                    raise ValueError(f"LoRA generation returned no images: {gen_result}")
                out_url = gen_imgs[0]["url"] if isinstance(gen_imgs[0], dict) else gen_imgs[0]
                print(f"[{jid}] LoRA result ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ {str(out_url)[:60]}")

                set_job(jid, {"status": "running", "progress": 85, "stage": "saving"})
                out_bytes = _req.get(out_url, timeout=60).content
                out_img = Image.open(io.BytesIO(out_bytes)).convert("RGB")
                out_path = OUTPUT_DIR / f"{jid}_portrait_0.png"
                out_img.save(str(out_path), "PNG")
                print(f"[{jid}] LoRA result saved ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ {out_path}")

                set_job(jid, {
                    "status": "complete",
                    "progress": 100,
                    "images": [f"/portrait/{jid}/0"],
                    "method": "outfit-swap-lora",
                    "outfit": outfit_prompt,
                })
                print(f"[{jid}] Outfit swap complete (LoRA pathway)")
                return  # ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â Ãƒâ€šÃ‚Â done, skip FASHN pathway below

            # ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚Â
            # PATHWAY B ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â FASHN virtual try-on + smart composite (fallback)
            # ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚ÂÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¢Ãƒâ€šÃ‚Â

            # ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ Step 1: Download master portrait & upload to fal.ai ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬
            set_job(jid, {"status": "running", "progress": 5, "stage": "uploading_portrait"})
            print(f"[{jid}] Fetching master portrait...")
            tmp_portrait = tmp_dir / "portrait.png"
            img_resp = _req.get(master_url, timeout=30)
            img_resp.raise_for_status()
            tmp_portrait.write_bytes(img_resp.content)
            portrait_fal_url = _fal.upload_file(str(tmp_portrait))
            print(f"[{jid}] Portrait uploaded ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ {portrait_fal_url[:60]}")

            # ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ Step 2: Generate flat-lay garment image from text prompt ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬
            # Strengthen the colour name so FLUX doesn't drift to similar hues
            set_job(jid, {"status": "running", "progress": 20, "stage": "generating_garment"})
            # Extract any colour words and amplify them
            colour_emphasis = f"exact colour as described: {outfit_prompt},"
            garment_gen_prompt = (
                f"flat-lay product photo of {outfit_prompt}, "
                f"{colour_emphasis} "
                f"isolated on pure white background, fashion editorial, "
                f"professional clothing photography, no model, no person, "
                f"centered, high detail, sharp focus, "
                f"accurate colour, no colour shift"
            )
            print(f"[{jid}] Generating garment image: {garment_gen_prompt[:80]}...")
            garment_result = _fal.subscribe("fal-ai/flux/dev", arguments={
                "prompt": garment_gen_prompt,
                "num_inference_steps": 40,
                "guidance_scale": 4.5,
                "num_images": 1,
                "image_size": {"width": 768, "height": 1024},
                "enable_safety_checker": False,
            })
            garment_imgs = garment_result.get("images") or []
            if not garment_imgs:
                raise ValueError(f"FLUX garment generation returned no images: {garment_result}")
            garment_url = garment_imgs[0]["url"]
            print(f"[{jid}] Garment image generated ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ {garment_url[:60]}")

            # ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ Step 3: FASHN v1.6 Virtual Try-On ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬
            set_job(jid, {"status": "running", "progress": 50, "stage": "tryon"})
            print(f"[{jid}] Running FASHN v1.6 try-on...")
            tryon_result = _fal.subscribe("fal-ai/fashn/tryon/v1.6", arguments={
                "model_image":        portrait_fal_url,
                "garment_image":      garment_url,
                "category":           "auto",
                "mode":               "quality",
                "garment_photo_type": "flat-lay",
                "segmentation_free":  True,
                "num_samples":        1,
            })

            tryon_imgs = tryon_result.get("images") or []
            if not tryon_imgs:
                raise ValueError(f"FASHN try-on returned no images: {tryon_result}")
            out_url = tryon_imgs[0]["url"] if isinstance(tryon_imgs[0], dict) else tryon_imgs[0]
            print(f"[{jid}] Try-on result ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ {str(out_url)[:60]}")

            # ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ Step 4: Smart composite ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â restore background, hands, face ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬
            # FASHN rewrites the whole image; we want ONLY the clothing pixels
            # from FASHN and everything else (background, microphone, hands,
            # face, jewellery) from the original portrait.
            set_job(jid, {"status": "running", "progress": 75, "stage": "compositing"})
            print(f"[{jid}] Compositing clothing onto original background...")

            from rembg import remove as rembg_remove

            # Load original and FASHN result
            orig_img  = Image.open(str(tmp_portrait)).convert("RGBA")
            fashn_bytes = _req.get(out_url, timeout=60).content
            fashn_img = Image.open(io.BytesIO(fashn_bytes)).convert("RGBA")

            # Match sizes
            if fashn_img.size != orig_img.size:
                fashn_img = fashn_img.resize(orig_img.size, Image.LANCZOS)

            w, h = orig_img.size

            # Person segmentation mask from original (clothing region proxy)
            orig_rgb = orig_img.convert("RGB")
            person_mask_raw = rembg_remove(orig_rgb, only_mask=True)
            person_mask = np.array(person_mask_raw.convert("L")).astype(np.float32) / 255.0

            # Restrict to clothing zone only:
            #   - exclude top 22 % ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ face, neck, hair
            #   - exclude bottom 20 % ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ hands, lap, lower legs
            clothing_mask = person_mask.copy()
            face_cut = int(h * 0.22)
            hand_cut = int(h * 0.80)
            clothing_mask[:face_cut, :] = 0
            clothing_mask[hand_cut:, :]  = 0

            # Feather edges so the seam isn't hard
            mask_pil = Image.fromarray((clothing_mask * 255).astype(np.uint8))
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=10))
            clothing_mask = np.array(mask_pil).astype(np.float32) / 255.0

            # Blend: original everywhere, FASHN only in clothing zone
            orig_arr  = np.array(orig_img).astype(np.float32)
            fashn_arr = np.array(fashn_img).astype(np.float32)
            alpha = clothing_mask[:, :, np.newaxis]
            composited = orig_arr * (1.0 - alpha) + fashn_arr * alpha
            composited = np.clip(composited, 0, 255).astype(np.uint8)

            out_img = Image.fromarray(composited, "RGBA").convert("RGB")

            # ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ Step 5: Save output ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚ÂÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬
            set_job(jid, {"status": "running", "progress": 90, "stage": "saving"})
            out_path = OUTPUT_DIR / f"{jid}_portrait_0.png"
            out_img.save(str(out_path), "PNG")
            print(f"[{jid}] Composited result saved ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ {out_path}")

            # Clean up temp files
            for f in tmp_dir.iterdir():
                try: f.unlink()
                except: pass
            try: tmp_dir.rmdir()
            except: pass

            set_job(jid, {
                "status": "complete",
                "progress": 100,
                "images": [f"/portrait/{jid}/0"],
                "method": "outfit-swap-fashn-composite",
                "outfit": outfit_prompt,
            })
            print(f"[{jid}] Outfit swap complete (FASHN + background restore)")

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
  "tiktok": "TikTok caption ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â hook line, 2-3 lines max, 3-5 relevant hashtags, ends with a CTA question",
  "instagram": "Instagram caption ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â 3-5 sentences expanding slightly on the script's core truth, line breaks between thoughts, 5-8 hashtags on a new line at the end",
  "youtube": "YouTube Shorts description ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â 3 to 5 sentences: open with a strong hook that names the core truth of the video, expand on why it matters, add a line inviting the viewer to follow for more content like this, then 5-8 relevant hashtags on a new line. Rich, complete and worth reading ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â not a throwaway line."
}}

Rules:
- Keep the voice and tone of the script ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â warm, direct, specific
- Never use generic hashtags like #motivation #love ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â make them specific to the content
- The hook line for TikTok should mirror the script opening or amplify the claim
- No emojis except sparingly on Instagram (1-2 max)
- Do not invent facts not in the script
- Hashtags must be lowercase, no spaces

Return only the JSON object, nothing else."""

    try:
        resp = client.chat.completions.create(
            model="openai/gpt-oss-20b",  # fast model ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â no thinking phase, responds in ~3s
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



@app.get("/fal-files")
async def list_fal_files():
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(
            "https://rest.alpha.fal.ai/storage/upload/list",
            headers={"Authorization": f"Key {FAL_KEY}"}
        )
        if r.status_code != 200:
            return {"error": r.text[:200], "status": r.status_code}
        data = r.json()
        # Filter for safetensors files only
        files = [f for f in (data.get("files") or data if isinstance(data, list) else []) 
                 if isinstance(f, dict) and ".safetensors" in str(f.get("url","") or f.get("file_name",""))]
        return {"files": files, "raw_sample": str(data)[:500]}


# ═══════════════════════════════════════════════════════
# VIDEO ENHANCEMENT — ffmpeg shot-variation pass
# ═══════════════════════════════════════════════════════
@app.post("/enhance-video")
async def enhance_video(video_url: str = Form(...)):
    import subprocess, tempfile, os, json as _json
    import httpx, fal_client

    tmp_dir = tempfile.mkdtemp()
    input_path  = os.path.join(tmp_dir, "input.mp4")
    output_path = os.path.join(tmp_dir, "enhanced.mp4")

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.get(video_url)
            resp.raise_for_status()
        with open(input_path, "wb") as f:
            f.write(resp.content)

        probe = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", input_path
        ], capture_output=True, text=True, timeout=30)
        try:
            duration = float(_json.loads(probe.stdout)["format"]["duration"])
        except Exception:
            # last resort: ask for stream duration
            probe2 = subprocess.run([
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=duration",
                "-of", "json", input_path
            ], capture_output=True, text=True, timeout=30)
            try:
                duration = float(_json.loads(probe2.stdout)["streams"][0]["duration"])
            except Exception:
                duration = 8.0

        t1 = round(duration / 3, 3)
        t2 = round(2 * duration / 3, 3)

        aprobe = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "json", input_path
        ], capture_output=True, text=True, timeout=15)
        has_audio = '"audio"' in aprobe.stdout

        if has_audio:
            fc = (
                f"[0:v]split=3[vs1][vs2][vs3];"
                f"[vs1]trim=0:{t1},setpts=PTS-STARTPTS,scale=iw*1.06:ih*1.06,crop=iw/1.06:ih/1.06[v1];"
                f"[vs2]trim={t1}:{t2},setpts=PTS-STARTPTS,hflip[v2];"
                f"[vs3]trim={t2},setpts=PTS-STARTPTS[v3];"
                f"[0:a]asplit=3[as1][as2][as3];"
                f"[as1]atrim=0:{t1},asetpts=PTS-STARTPTS[a1];"
                f"[as2]atrim={t1}:{t2},asetpts=PTS-STARTPTS[a2];"
                f"[as3]atrim={t2},asetpts=PTS-STARTPTS[a3];"
                f"[v1][a1][v2][a2][v3][a3]concat=n=3:v=1:a=1[outv][outa]"
            )
            maps = ["-map", "[outv]", "-map", "[outa]"]
            acodec = ["-c:a", "aac", "-b:a", "128k"]
        else:
            fc = (
                f"[0:v]split=3[vs1][vs2][vs3];"
                f"[vs1]trim=0:{t1},setpts=PTS-STARTPTS,scale=iw*1.06:ih*1.06,crop=iw/1.06:ih/1.06[v1];"
                f"[vs2]trim={t1}:{t2},setpts=PTS-STARTPTS,hflip[v2];"
                f"[vs3]trim={t2},setpts=PTS-STARTPTS[v3];"
                f"[v1][v2][v3]concat=n=3:v=1:a=0[outv]"
            )
            maps = ["-map", "[outv]"]
            acodec = []

        cmd = ["ffmpeg", "-y", "-i", input_path, "-filter_complex", fc,
               *maps, "-c:v", "libx264", "-preset", "fast", "-crf", "23",
               *acodec, output_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-800:])

        enhanced_url = fal_client.upload_file(output_path)
        return JSONResponse({"enhanced_url": enhanced_url})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        for p in [input_path, output_path]:
            try: os.unlink(p)
            except: pass
        try: os.rmdir(tmp_dir)
        except: pass

