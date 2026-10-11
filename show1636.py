path = r"C:\Users\agnie\Projects\persona-studio-backend\main.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Show lines around 1634-1640 (0-indexed)
for i, ln in enumerate(lines[1633:1641], start=1634):
    print(f"{i}: {repr(ln)}")