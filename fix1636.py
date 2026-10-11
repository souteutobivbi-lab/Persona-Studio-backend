path = r"C:\Users\agnie\Projects\persona-studio-backend\main.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Replace lines 1635-1638 (0-indexed) with correct concat list writer
lines[1635:1639] = [
    '        with open(lst,"w") as f:\n',
    '            f.write("file \'" + s1 + "\'\\nfile \'" + s2 + "\'\\nfile \'" + s3 + "\'\\n")\n',
]

with open(path, "w", encoding="utf-8") as f:
    f.writelines(lines)

print("DONE")
for i, ln in enumerate(lines[1633:1640], start=1634):
    print(f"{i}: {repr(ln)}")