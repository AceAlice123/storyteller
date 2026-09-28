#!/usr/bin/env python3
"""
Storyteller - Automated Text-to-Video Synthesis Platform
Main Command-Line Interface (CLI) Entrypoint
Course: IGNOU MCA MCSP-232 Final Project
Author: Ashu Chaurasiya (Enrolment: 2501321326)
Guide: Shubham Jaiswal (SDE-II Turing)

Usage Example:
    python storyteller.py --script ./sample_data/sample_script.txt --images ./sample_data/images --output ./output/story.mp4
"""

import argparse
import sys
import logging
from pathlib import Path

from config import (
    PipelineConfig,
    DATABASE_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TEMP_AUDIO_DIR,
    DEFAULT_TTS_ENGINE
)
from orchestrator.facade import StorytellerFacade
from utils.logger import setup_logger


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the Storyteller CLI platform."""
    parser = argparse.ArgumentParser(
        prog="storyteller",
        description="Storyteller: Automated Text-to-Video Synthesis Platform (IGNOU MCA MCSP-232).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic execution with default offline pyttsx3 and CLIP alignment:
  python storyteller.py --script script.txt --images ./images/

  # Custom output path and project name:
  python storyteller.py --script script.txt --images ./images/ --output ./output/ai_story.mp4 -p "AI History"

  # Fast dry run without video rendering:
  python storyteller.py --script script.txt --images ./images/ --dry-run
        """
    )

    parser.add_argument(
        "-s", "--script",
        dest="script_path",
        type=Path,
        required=True,
        help="Path to the input plain text (.txt) narrative script."
    )

    parser.add_argument(
        "-i", "--images",
        dest="image_dir",
        type=Path,
        required=True,
        help="Path to the local folder containing visual assets (.jpg, .png, .webp)."
    )

    parser.add_argument(
        "-o", "--output",
        dest="output_path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "final_story.mp4",
        help="Target output path for the rendered MP4 video (default: output/final_story.mp4)."
    )

    parser.add_argument(
        "-p", "--project-name",
        dest="project_name",
        type=str,
        default="Storyteller Production Run",
        help="Human-readable project title for database tracking."
    )

    parser.add_argument(
        "--tts-engine",
        dest="tts_engine",
        type=str,
        choices=["pyttsx3", "dummy", "mock"],
        default=DEFAULT_TTS_ENGINE,
        help="Pluggable Text-to-Speech strategy (default: pyttsx3)."
    )

    parser.add_argument(
        "--db",
        dest="db_path",
        type=Path,
        default=DATABASE_PATH,
        help="Path to the SQLite database file (default: storyteller.db)."
    )

    parser.add_argument(
        "--force-cache",
        dest="force_cache",
        action="store_true",
        help="Bypass the SQLite embedding cache and force recalculation of all image tensors."
    )

    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Execute parsing, audio generation, and CLIP matching without rendering the video."
    )

    parser.add_argument(
        "-v", "--verbose",
        dest="verbose",
        action="store_true",
        help="Enable verbose debug logging."
    )

    return parser.parse_args()


def main() -> int:
    """Main execution function for the Storyteller CLI."""
    args = parse_arguments()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger("Storyteller", level=log_level)

    print("\n" + "=" * 70)
    print("  STORYTELLER: Automated Text-to-Video Synthesis Platform")
    print("  IGNOU MCA (MCSP-232) Final Semester Project")
    print("=" * 70 + "\n")

    try:
        config = PipelineConfig(
            script_path=args.script_path,
            image_dir=args.image_dir,
            output_path=args.output_path,
            project_name=args.project_name,
            db_path=args.db_path,
            tts_engine=args.tts_engine,
            force_cache_recompute=args.force_cache,
            dry_run=args.dry_run
        )

        facade = StorytellerFacade()
        result_video = facade.run(config)

        if not args.dry_run:
            print(f"\n[+] Video synthesis successfully completed!")
            print(f"[+] Output Video: {result_video.resolve()}\n")
        else:
            print(f"\n[+] Dry run completed successfully (all assets parsed and aligned).\n")

        return 0

    except FileNotFoundError as e:
        logger.error(f"Filesystem Error: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Validation Error: {e}")
        return 2
    except Exception as e:
        logger.exception(f"Unexpected Pipeline Failure: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
