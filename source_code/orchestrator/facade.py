"""
Storyteller - Automated Text-to-Video Synthesis Platform
Facade Pattern: StorytellerFacade (Main Pipeline Orchestrator)
Course: IGNOU MCA MCSP-232

Provides a unified, simplified interface to execute the complete four-module pipeline:
1. Module 1: Ingestion & Text Parsing
2. Module 2: Text-to-Speech (TTS) Synthesis & Duration Measurement
3. Module 3: CLIP Machine Learning Semantic Visual Alignment & Caching
4. Module 4: MoviePy Timeline Assembly, Subtitling, & 1080p Video Rendering
"""

from pathlib import Path
from typing import Dict, Any, Optional
import time

from config import PipelineConfig
from database.repository import StorytellerRepository
from modules.parser.text_parser import TextParser
from modules.tts.synthesizer import AudioSynthesizer
from modules.aligner.visual_aligner import VisualAligner
from modules.compositor.video_compositor import VideoCompositor
from utils.logger import setup_logger

logger = setup_logger("Storyteller.Facade")


class StorytellerFacade:
    """
    Facade Design Pattern implementation.
    Hides the underlying subsystem complexity (NLP parsing, neural tensor operations,
    audio encoding, and video compositing) behind a single intuitive run() method.
    """

    def __init__(self, repository: Optional[StorytellerRepository] = None):
        self.repository = repository

    def run(self, config: PipelineConfig) -> Path:
        """
        Executes the end-to-end automated text-to-video synthesis pipeline.

        Args:
            config: PipelineConfig instance with all input, output, and execution parameters.

        Returns:
            Path: Absolute path to the finalized, rendered MP4 video.
        """
        start_time = time.time()
        logger.info("=" * 70)
        logger.info(f"STARTING STORYTELLER PIPELINE: '{config.project_name}'")
        logger.info("=" * 70)

        # 0. Validate configuration & filesystem paths
        config.validate()
        repo = self.repository or StorytellerRepository(db_path=config.db_path)

        # ---------------------------------------------------------------------
        # Stage 1: Text Parsing & Database Instantiation (Module 1)
        # ---------------------------------------------------------------------
        t0 = time.time()
        logger.info(">>> STAGE 1/4: Parsing narrative script & initializing database...")
        parser = TextParser()
        project_id, seq_ids, sentences = parser.ingest_script(
            script_path=config.script_path,
            project_name=config.project_name,
            repository=repo
        )
        t_parse = time.time() - t0
        logger.info(
            f"Stage 1 Complete: {len(sentences)} sequences indexed in {t_parse:.2f}s."
        )

        # ---------------------------------------------------------------------
        # Stage 2: Offline Audio Synthesis & Duration Calculation (Module 2)
        # ---------------------------------------------------------------------
        t0 = time.time()
        logger.info(f">>> STAGE 2/4: Synthesizing offline speech via '{config.tts_engine}'...")
        synthesizer = AudioSynthesizer(
            repository=repo,
            tts_engine=config.tts_engine,
            temp_audio_dir=config.temp_audio_dir
        )
        audio_records = synthesizer.synthesize_project_sequences(project_id)
        total_audio_runtime = sum(a["duration_sec"] for a in audio_records)
        t_audio = time.time() - t0
        logger.info(
            f"Stage 2 Complete: {len(audio_records)} audio clips generated "
            f"({total_audio_runtime:.2f}s runtime) in {t_audio:.2f}s."
        )

        # ---------------------------------------------------------------------
        # Stage 3: Semantic Visual Alignment & Caching via CLIP (Module 3)
        # ---------------------------------------------------------------------
        t0 = time.time()
        logger.info(">>> STAGE 3/4: Matching visual assets using OpenAI CLIP (ViT-B/32)...")
        aligner = VisualAligner(
            repository=repo,
            use_mock=(config.tts_engine == "mock" or config.dry_run)
        )
        alignment_results = aligner.align_project_sequences(
            project_id=project_id,
            image_dir=config.image_dir,
            force_recompute=config.force_cache_recompute
        )
        t_align = time.time() - t0
        logger.info(
            f"Stage 3 Complete: Aligned {len(alignment_results)} visual scenes in {t_align:.2f}s."
        )

        # ---------------------------------------------------------------------
        # Stage 4: Timeline Synchronization & 1080p Video Rendering (Module 4)
        # ---------------------------------------------------------------------
        if config.dry_run:
            logger.info(">>> STAGE 4/4: DRY RUN mode enabled. Skipping MoviePy rendering.")
            output_file = config.output_path
        else:
            t0 = time.time()
            logger.info(">>> STAGE 4/4: Synchronizing timeline & rendering 1080p MP4...")
            compositor = VideoCompositor(repository=repo)
            output_file = compositor.build_video(
                project_id=project_id,
                output_path=config.output_path
            )
            t_render = time.time() - t0
            logger.info(f"Stage 4 Complete: Video rendered in {t_render:.2f}s.")

        total_elapsed = time.time() - start_time
        logger.info("=" * 70)
        logger.info("STORYTELLER PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"Output File:     {output_file}")
        logger.info(f"Total Sequences: {len(sentences)}")
        logger.info(f"Audio Duration:  {total_audio_runtime:.2f}s")
        logger.info(f"Total Execution: {total_elapsed:.2f}s")
        logger.info("=" * 70)

        return output_file
