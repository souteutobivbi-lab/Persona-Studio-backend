import pathlib, sys

MAIN_PY = pathlib.Path(r"C:\Users\agnie\Projects\persona-studio-backend\main.py")

if not MAIN_PY.exists():
    sys.exit(f"ERROR: {MAIN_PY} not found.")

text = MAIN_PY.read_text(encoding="utf-8")

OLD_SETTINGS = '"voice_settings": {"stability": 0.5, "similarity_boost": 0.75}'
NEW_SETTINGS = '"voice_settings": {"stability": 0.38, "similarity_boost": 0.90, "style": 0.60, "use_speaker_boost": True}'

if OLD_SETTINGS not in text:
    sys.exit("ERROR: voice_settings pattern not found. Already patched?")

text = text.replace(OLD_SETTINGS, NEW_SETTINGS, 1)
print("OK voice_settings updated")

OLD_WRITE = "    out_path.write_bytes(r.content)\n"
NEW_WRITE = '''    out_path.write_bytes(r.content)

    _crowd = pathlib.Path(__file__).parent / "crowd_ambience.mp3"
    if _crowd.exists():
        import subprocess, shutil as _shutil
        _tmp = out_path.with_name(out_path.stem + "_mixed.mp3")
        try:
            subprocess.run(
                ["ffmpeg", "-y",
                 "-i", str(out_path),
                 "-stream_loop", "-1", "-i", str(_crowd),
                 "-filter_complex",
                 "[0:a]volume=1.0[voice];[1:a]volume=0.12[crowd];"
                 "[voice][crowd]amix=inputs=2:duration=first:dropout_transition=2[out]",
                 "-map", "[out]",
                 "-codec:a", "libmp3lame", "-q:a", "4",
                 str(_tmp)],
                check=True, capture_output=True)
            _shutil.move(str(_tmp), str(out_path))
        except Exception:
            if _tmp.exists():
                _tmp.unlink(missing_ok=True)

'''

if OLD_WRITE not in text:
    sys.exit("ERROR: out_path.write_bytes line not found.")

text = text.replace(OLD_WRITE, NEW_WRITE, 1)
print("OK crowd mixing injected")

if "import pathlib" not in text and "from pathlib" not in text:
    text = text.replace("import os\n", "import os\nimport pathlib\n", 1)
    print("OK pathlib import added")

MAIN_PY.write_text(text, encoding="utf-8")
print("Done - main.py patched successfully")
