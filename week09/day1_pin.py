import sys
import platform
import subprocess
from pathlib import Path

def main():
    out_dir = Path("week09")
    out_dir.mkdir(parents=True, exist_ok=True)

    info_lines = []
    info_lines.append(f"Python version: {sys.version}")
    info_lines.append(f"Platform: {platform.platform()}")
    info_lines.append(f"Machine: {platform.machine()}")
    info_lines.append(f"Processor: {platform.processor()}")

    try:
        pip_version = subprocess.check_output(
            [sys.executable, "-m", "pip", "--version"], text=True
        ).strip()
        info_lines.append(f"Pip version: {pip_version}")
    except Exception as e:
        info_lines.append(f"Pip version: could not detect ({e})")

    content = "\n".join(info_lines)
    print(content)

    out_path = out_dir / "environment_info.txt"
    out_path.write_text(content, encoding="utf-8")
    print(f"\nSaved to: {out_path}")

if __name__ == "__main__":
    main()