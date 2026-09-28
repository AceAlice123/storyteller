-- ============================================================================
-- Storyteller: Automated Text-to-Video Synthesis Platform
-- Database Schema (SQLite 3NF Normalized)
-- Course: IGNOU MCA MCSP-232 Final Project
-- Author: Ashu Chaurasiya (Enrolment: 2501321326)
-- Guide: Shubham Jaiswal (SDE-II Turing)
-- ============================================================================

PRAGMA foreign_keys = ON;

-- ----------------------------------------------------------------------------
-- Table 1: projects
-- Stores high-level metadata for each video generation pipeline execution.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS projects (
    project_id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_name VARCHAR(100) NOT NULL,
    creation_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ----------------------------------------------------------------------------
-- Table 2: sequences
-- Stores ordered sentence/block level representations of the input script.
-- Acts as the foundational timeline backbone for audio and video assembly.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sequences (
    seq_id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    sequence_order INTEGER NOT NULL,
    script_text TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(project_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sequences_project_order 
    ON sequences(project_id, sequence_order);

-- ----------------------------------------------------------------------------
-- Table 3: audio_assets
-- Stores localized Text-to-Speech audio clips and their exact durations.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audio_assets (
    audio_id INTEGER PRIMARY KEY AUTOINCREMENT,
    seq_id INTEGER NOT NULL UNIQUE,
    file_path VARCHAR(255) NOT NULL UNIQUE,
    duration_sec REAL NOT NULL,
    FOREIGN KEY (seq_id) REFERENCES sequences(seq_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_audio_assets_seq 
    ON audio_assets(seq_id);

-- ----------------------------------------------------------------------------
-- Table 4: image_assets
-- Records the CLIP-selected visual asset for each sequence with its match score.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS image_assets (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT,
    seq_id INTEGER NOT NULL UNIQUE,
    file_path VARCHAR(255) NOT NULL,
    clip_score REAL NOT NULL,
    FOREIGN KEY (seq_id) REFERENCES sequences(seq_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_image_assets_seq 
    ON image_assets(seq_id);

-- ----------------------------------------------------------------------------
-- Table 5: image_embeddings_cache
-- Caches 512-dimensional CLIP embeddings for local image assets.
-- Prevents redundant matrix operations across repeated runs on the same folder.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS image_embeddings_cache (
    cache_id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path VARCHAR(255) NOT NULL UNIQUE,
    file_mtime REAL NOT NULL,
    file_size INTEGER NOT NULL,
    embedding_blob BLOB NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_embedding_cache_path 
    ON image_embeddings_cache(image_path);
