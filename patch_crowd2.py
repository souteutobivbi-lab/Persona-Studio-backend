import pathlib, sys

MAIN_PY = pathlib.Path(r"C:\Users\agnie\Projects\persona-studio-backend\main.py")
text = MAIN_PY.read_text(encoding="utf-8")

OLD = (
    '                ["ffmpeg", "-y",\n'
    '                 "-i", str(out_path),\n'
    '                 "-stream_loop", "-1", "-i", str(_crowd),\n'
    '                 "-filter_complex",\n'
    '                 "[0:a]volume=1.0[voice];[1:a]volume=0.25[crowd];"\n'
    '                 "[voice][crowd]amix=inputs=2:duration=first:dropout_transition=2[out]",\n'
    '                 "-map", "[out]",\n'
    '                 "-codec:a", "libmp3lame", "-q:a", "4",\n'
    '                 str(_tmp)],'
)

NEW = (
    '                ["ffmpeg", "-y",\n'
    '                 "-i", str(out_path),\n'
    '                 "-stream_loop", "-1", "-i", str(_crowd),\n'
    '                 "-filter_complex",\n'
    '                 "[1:a]volume=0.08[ambient];"\n'
    '                 "[1:a]volume=0.35,afade=t=in:st=0:d=2,afade=t=out:st=0:d=2[swell];"\n'
    '                 "[ambient][swell]amix=inputs=2:duration=longest[crowd];"\n'
    '                 "[0:a]volume=1.0[voice];"\n'
    '                 "[voice][crowd]amix=inputs=2:duration=first:dropout_transition=2[out]",\n'
    '                 "-map", "[out]",\n'
    '                 "-codec:a", "libmp3lame", "-q:a", "4",\n'
    '                 str(_tmp)],'
)

if OLD not in text:
    sys.exit("ERROR: pattern not found - paste the error to Claude")

text = text.replace(OLD, NEW, 1)
MAIN_PY.write_text(text, encoding="utf-8")
print("Done - crowd filter updated")
