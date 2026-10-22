import subprocess, pathlib, sys

OUT = pathlib.Path(r"C:\Users\agnie\Projects\persona-studio-backend\crowd_ambience.mp3")

cmd = [
    "ffmpeg", "-y",
    "-f", "lavfi",
    "-i", "aevalsrc=random(0)*0.4|random(1)*0.4:s=44100:d=30",
    "-af", (
        "highpass=f=80,"
        "lowpass=f=900,"
        "equalizer=f=250:width_type=o:width=2:g=5,"
        "equalizer=f=500:width_type=o:width=1.5:g=2,"
        "volume=0.30,"
        "afade=t=in:st=0:d=2,"
        "afade=t=out:st=28:d=2"
    ),
    "-codec:a", "libmp3lame", "-q:a", "4",
    str(OUT),
]

print("Generating crowd_ambience.mp3 ...")
try:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-1000:])
        sys.exit("FFmpeg error - see above")
except FileNotFoundError:
    sys.exit("ffmpeg not found on PATH")

print(f"Done - {OUT.stat().st_size // 1024} KB saved to {OUT}")
