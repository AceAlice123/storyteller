"""
Storyteller - Automated Text-to-Video Synthesis Platform
Module 4: MoviePy Video Compositor
Course: IGNOU MCA MCSP-232

Responsible for:
1. Executing 3NF SQL JOIN query to retrieve timeline sequence (script, audio path, duration, image).
2. Frame normalization: scaling and letterboxing candidate images to standardized Full HD 1080p (1920x1080).
3. Synchronizing image duration to floating-point audio length (.with_duration / .set_duration).
4. Audio track multiplexing (.with_audio / .set_audio).
5. Dynamic subtitle overlay generation with resilient fallback.
6. Concatenation and H.264 / AAC encoding to render final MP4 video.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from database.repository import StorytellerRepository
from config import (
    VIDEO_RESOLUTION,
    VIDEO_FPS,
    VIDEO_CODEC,
    AUDIO_CODEC,
    SUBTITLE_FONT_SIZE,
    SUBTITLE_FONT_COLOR,
    SUBTITLE_BG_COLOR,
    SUBTITLE_BOTTOM_MARGIN,
    DEFAULT_OUTPUT_DIR
)
from utils.logger import setup_logger

logger = setup_logger("Storyteller.Compositor")


def _apply_clip_duration(clip, duration: float):
    """Compatible duration setter across MoviePy 1.x and 2.x."""
    if hasattr(clip, "with_duration"):
        return clip.with_duration(duration)
    elif hasattr(clip, "set_duration"):
        return clip.set_duration(duration)
    else:
        clip.duration = duration
        return clip


def _apply_clip_audio(clip, audio_clip):
    """Compatible audio setter across MoviePy 1.x and 2.x."""
    if hasattr(clip, "with_audio"):
        return clip.with_audio(audio_clip)
    elif hasattr(clip, "set_audio"):
        return clip.set_audio(audio_clip)
    else:
        clip.audio = audio_clip
        return clip


def _apply_clip_position(clip, position):
    """Compatible position setter across MoviePy 1.x and 2.x."""
    if hasattr(clip, "with_position"):
        return clip.with_position(position)
    elif hasattr(clip, "set_position"):
        return clip.set_position(position)
    else:
        return clip


class VideoCompositor:
    """
    Assembles multimedia sequences into a synchronized, broadcast-ready 1080p MP4 presentation.
    """

    def __init__(
        self,
        repository: StorytellerRepository,
        resolution: Tuple[int, int] = VIDEO_RESOLUTION,
        fps: int = VIDEO_FPS
    ):
        self.repository = repository
        self.width, self.height = resolution
        self.fps = fps

    def normalize_frame(self, image_path: Path | str) -> np.ndarray:
        """
        Loads an image of arbitrary dimensions and letterboxes it onto a 1920x1080 canvas
        to prevent distortion or dimension mismatch during concatenation.

        Args:
            image_path: Path to visual asset.

        Returns:
            np.ndarray: (1080, 1920, 3) RGB numpy array ready for MoviePy ImageClip.
        """
        path = Path(image_path).resolve()
        with Image.open(path) as img:
            rgb_img = img.convert("RGB")
            orig_w, orig_h = rgb_img.size

            # Compute aspect ratio scale factor
            scale = min(self.width / orig_w, self.height / orig_h)
            new_w = max(1, int(orig_w * scale))
            new_h = max(1, int(orig_h * scale))

            resized = rgb_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Center onto black canvas of target resolution
            canvas = Image.new("RGB", (self.width, self.height), color=(10, 10, 14))
            paste_x = (self.width - new_w) // 2
            paste_y = (self.height - new_h) // 2
            canvas.paste(resized, (paste_x, paste_y))

            return np.array(canvas)

    def create_subtitle_overlay(self, text: str, duration: float):
        """
        Creates a subtitle clip for the given text.
        Attempts native MoviePy TextClip; falls back to PIL rendering if font or ImageMagick
        issues arise (addressing Section 12 & 30 of synopsis).

        Args:
            text: Sentence to display.
            duration: Display duration in seconds.

        Returns:
            MoviePy VideoClip or ImageClip with alpha channel.
        """
        # Try native TextClip
        try:
            from moviepy import TextClip
            # In MoviePy 2.x, TextClip accepts text, font_size, color, etc.
            txt_clip = TextClip(
                text=text,
                font_size=SUBTITLE_FONT_SIZE,
                color=SUBTITLE_FONT_COLOR,
                bg_color="black"
            )
            txt_clip = _apply_clip_duration(txt_clip, duration)
            txt_clip = _apply_clip_position(txt_clip, ("center", self.height - 120))
            return txt_clip
        except Exception as e:
            logger.debug(f"Native TextClip fallback triggered: {e}. Rendering with PIL.")
            return self._render_pil_subtitle(text, duration)

    def _render_pil_subtitle(self, text: str, duration: float):
        """
        Fallback subtitle renderer using Pillow. Creates an RGBA overlay with
        a rounded dark translucent badge and clean centered text.
        """
        from moviepy import ImageClip

        # Create transparent overlay matching video dimensions
        overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Attempt to load a default TrueType font or fallback to basic font
        font = None
        try:
            # Common Windows fonts
            for font_name in ["arial.ttf", "calibri.ttf", "segoeui.ttf"]:
                try:
                    font = ImageFont.truetype(font_name, SUBTITLE_FONT_SIZE)
                    break
                except IOError:
                    continue
        except Exception:
            font = None

        if font is None:
            font = ImageFont.load_default()

        # Word wrap text to fit within width minus margins
        max_text_width = self.width - 240
        words = text.split()
        lines = []
        current_line = []

        for word in words:
            test_line = " ".join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            line_w = bbox[2] - bbox[0]
            if line_w <= max_text_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
        if current_line:
            lines.append(" ".join(current_line))

        full_sub_text = "\n".join(lines)
        bbox = draw.multiline_textbbox((0, 0), full_sub_text, font=font, align="center")
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Draw translucent background banner
        padding_x = 24
        padding_y = 12
        banner_w = text_w + padding_x * 2
        banner_h = text_h + padding_y * 2
        banner_x = (self.width - banner_w) // 2
        banner_y = self.height - banner_h - SUBTITLE_BOTTOM_MARGIN

        # Draw dark translucent rounded box
        draw.rounded_rectangle(
            [banner_x, banner_y, banner_x + banner_w, banner_y + banner_h],
            radius=12,
            fill=SUBTITLE_BG_COLOR
        )

        # Draw centered text
        text_x = (self.width - text_w) // 2
        text_y = banner_y + padding_y
        draw.multiline_text(
            (text_x, text_y),
            full_sub_text,
            font=font,
            fill=SUBTITLE_FONT_COLOR,
            align="center"
        )

        # Convert to MoviePy ImageClip with transparency
        np_overlay = np.array(overlay)
        sub_clip = ImageClip(np_overlay, is_mask=False)
        sub_clip = _apply_clip_duration(sub_clip, duration)
        return sub_clip

    def build_video(
        self,
        project_id: int,
        output_path: Path | str = DEFAULT_OUTPUT_DIR / "final_story.mp4"
    ) -> Path:
        """
        Executes Module 4 video compilation:
        1. Queries timeline via 3NF SQL JOIN.
        2. Loops row-by-row, creating synchronized ImageClip + AudioFileClip + Subtitles.
        3. Concatenates sequential blocks.
        4. Encodes and writes final MP4 video.

        Args:
            project_id: Target project ID.
            output_path: Destination path for final .mp4.

        Returns:
            Path: Absolute path to rendered MP4 video.
        """
        from moviepy import ImageClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips

        target_output = Path(output_path).resolve()
        target_output.parent.mkdir(parents=True, exist_ok=True)

        # Step 1: SQL JOIN query
        timeline = self.repository.get_project_timeline(project_id)
        if not timeline:
            raise ValueError(f"No synchronized timeline rows found for Project ID: {project_id}")

        logger.info(
            f"Assembling video from {len(timeline)} timeline sequences into: {target_output.name}"
        )
        video_clips: List[Any] = []

        # Step 3: Loop row-by-row through SQL results
        for row in timeline:
            seq_order = row["sequence_order"]
            script_text = row["script_text"]
            audio_path = row["audio_file_path"]
            duration_sec = float(row["audio_duration"])
            image_path = row["image_file_path"]

            logger.info(
                f"Processing Scene {seq_order:02d}: {Path(image_path).name} "
                f"({duration_sec:.2f}s) | '{script_text[:35]}...'"
            )

            # 3a. Frame normalization to 1080p
            frame_rgb = self.normalize_frame(image_path)
            img_clip = ImageClip(frame_rgb)

            # 3b. Exact duration synchronization
            img_clip = _apply_clip_duration(img_clip, duration_sec)

            # 3c. Audio attachment
            audio_clip = AudioFileClip(audio_path)
            img_clip = _apply_clip_audio(img_clip, audio_clip)

            # 3d. Dynamic Subtitle Overlay
            sub_clip = self.create_subtitle_overlay(script_text, duration_sec)

            # Composite image with subtitle
            composite_scene = CompositeVideoClip(
                [img_clip, sub_clip],
                size=(self.width, self.height)
            )
            composite_scene = _apply_clip_duration(composite_scene, duration_sec)
            composite_scene = _apply_clip_audio(composite_scene, audio_clip)

            video_clips.append(composite_scene)

        # Step 4: Concatenate video clips
        logger.info(f"Concatenating {len(video_clips)} scenes...")
        final_video = concatenate_videoclips(video_clips, method="compose")

        # Step 5: Render final .mp4 video to disk
        logger.info(
            f"Encoding final MP4 video at {self.fps} fps with {VIDEO_CODEC} and {AUDIO_CODEC}...\n...Please wait this may take for 10-15 minutes nown based on the system specifications...  "
        )
        final_video.write_videofile(
            str(target_output),
            fps=self.fps,
            codec=VIDEO_CODEC,
            audio_codec=AUDIO_CODEC,
            preset="ultrafast",
            threads=4,
            logger=None  # Suppress verbose ffmpeg terminal bar in favor of structured logs
        )

        # Close all clip file handles
        final_video.close()
        for c in video_clips:
            try:
                c.close()
            except Exception:
                pass

        if not target_output.is_file() or target_output.stat().st_size == 0:
            raise RuntimeError(f"Video rendering failed; output file is empty: {target_output}")

        file_size_mb = target_output.stat().st_size / (1024 * 1024)
        logger.info(
            f"SUCCESS! Rendered final synchronized video to: {target_output} ({file_size_mb:.2f} MB)"
        )
        return target_output
