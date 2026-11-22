"""One-ingredient pizza entry: writes the final 1024x1024 pizza render to the output directory."""
import hashlib, os, sys
from pathlib import Path

SOURCE = Path(__file__).with_name("entry.png")

def main() -> int:
    out_dir = Path(os.getenv("MAKEFUN_OUTPUT_DIR") or ("/output" if os.path.isdir("/output") or not os.path.isdir(os.getcwd()) else "output"))
    if str(out_dir) == "/output" and not out_dir.exists():
        try: out_dir.mkdir(parents=True)
        except OSError: out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "entry.png"
    with SOURCE.open("rb") as src, dest.open("wb") as dst:
        while True:
            chunk = src.read(65536)
            if not chunk: break
            dst.write(chunk)
    if dest.stat().st_size != SOURCE.stat().st_size:
        print("size mismatch", file=sys.stderr); return 1
    print(hashlib.sha256(dest.read_bytes()).hexdigest())
    return 0

if __name__ == "__main__":
    sys.exit(main())
