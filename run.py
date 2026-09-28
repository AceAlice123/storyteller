#!/usr/bin/env python3
"""
Bootstrap script for Storyteller project.
- Creates the output folder if it doesn't exist
- Sample data is expected inside the ".source_code" folder
otherwise run the script from the directory source_code  
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Get the directory where this script is located
    root_dir = Path(__file__).parent.resolve()
    print(f"Working directory: {root_dir}")

    # 1. Create output folder if it doesn't exist
    output_dir = root_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output folder ready: {output_dir}")

    # 2. Define the command (all references updated)
    cmd = [
        sys.executable,                                           # python
        str(root_dir / "source_code" / "storyteller.py"),
        "--script", str(root_dir / "source_code" / "sample_data" / "sample_script.txt"),
        "--images", str(root_dir / "source_code" / "sample_data" / "images"),
        "--output", str(output_dir / "story.mp4")
    ]

    print("\nRunning command:")
    print(" ".join(cmd))
    print("-" * 60)

    # 3. Execute the command
    try:
        subprocess.run(cmd, cwd=root_dir, check=True)
        print("\n✅ Storyteller completed successfully!")
        print(f"Output video saved to: {output_dir / 'story.mp4'}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: Storyteller exited with code {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("\n❌ Error: storyteller.py not found at expected location.")
        print("Expected path:", root_dir / "source_code" / "storyteller.py")
        sys.exit(1)

if __name__ == "__main__":
    main()