#!/usr/bin/env python3
"""
full_data_preparation.py

One-stop script performing:
1) Data ingestion (metadata)
2) Speaker diarization using pyannote.audio
3) Forced alignment using whisperx
4) Saving cleaned audio and alignment output

Usage example:
  python full_data_preparation.py --metadata metadata.csv --output_dir data/processed --log_level INFO

Metadata CSV format:
  participant_id,audio_path,transcript_path,label
  p1,data/raw/p1.wav,data/transcripts/p1.txt,1
  p2,data/raw/p2.wav,data/transcripts/p2.txt,0
"""

import os
import argparse
import pandas as pd
import torch
import torchaudio
from typing import List, Tuple, Optional
from pathlib import Path
import logging
from pyannote.audio import Pipeline  # For real diarization
import whisperx  # For real alignment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

########################################
# 1. Diarization (Real Pipeline)
########################################

def perform_speaker_diarization(audio_path: str) -> List[Tuple[float, float]]:
    """
    Perform speaker diarization on the audio file using pyannote.audio.

    Args:
        audio_path (str): Path to the input audio file.

    Returns:
        List[Tuple[float, float]]: List of (start_sec, end_sec) segments for the target speaker.
    """
    try:
        # Load pretrained diarization pipeline (requires huggingface token or local setup)
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=True)
        diarization = pipeline(audio_path)
        
        # For simplicity, assume the most prominent speaker (longest total duration)
        speaker_durations = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            duration = turn.end - turn.start
            speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration
        
        if not speaker_durations:
            logger.warning(f"No speakers detected in {audio_path}")
            return [(0, 0)]
        
        target_speaker = max(speaker_durations, key=speaker_durations.get)
        segments = [(turn.start, turn.end) for turn, _, speaker in diarization.itertracks(yield_label=True) if speaker == target_speaker]
        logger.info(f"Diarized {audio_path}: {len(segments)} segments for speaker {target_speaker}")
        return segments
    except Exception as e:
        logger.error(f"Diarization failed for {audio_path}: {str(e)}")
        return [(0, 0)]  # Default to empty segment on failure

def extract_speaker_audio(audio_path: str, segments: List[Tuple[float, float]]) -> Tuple[torch.Tensor, int]:
    """
    Extract and concatenate audio segments for the target speaker.

    Args:
        audio_path (str): Path to the input audio file.
        segments (List[Tuple[float, float]]): List of (start_sec, end_sec) segments.

    Returns:
        Tuple[torch.Tensor, int]: Concatenated waveform and sample rate.
    """
    try:
        waveform, sr = torchaudio.load(audio_path)
        if not segments or (len(segments) == 1 and segments[0] == (0, 0)):
            logger.warning(f"No valid segments for {audio_path}, returning full audio")
            return waveform, sr

        segment_waveforms = []
        for start_sec, end_sec in segments:
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            if start_sample >= end_sample or end_sample > waveform.shape[1]:
                logger.warning(f"Invalid segment ({start_sec}, {end_sec}) for {audio_path}, skipping")
                continue
            segment = waveform[:, start_sample:end_sample]
            segment_waveforms.append(segment)

        if not segment_waveforms:
            logger.warning(f"No valid segments extracted for {audio_path}, using silence")
            return torch.zeros(1, sr), sr
        
        concatenated = torch.cat(segment_waveforms, dim=1)
        return concatenated, sr
    except Exception as e:
        logger.error(f"Failed to extract speaker audio from {audio_path}: {str(e)}")
        raise RuntimeError(f"Audio extraction failed: {str(e)}")

########################################
# 2. Forced Alignment (Real Pipeline)
########################################

def perform_forced_alignment(audio_path: str, transcript: str) -> List[Tuple[str, float, float]]:
    """
    Perform forced alignment between audio and transcript using whisperx.

    Args:
        audio_path (str): Path to the audio file.
        transcript (str): Text transcript to align.

    Returns:
        List[Tuple[str, float, float]]: List of (word, start_time, end_time) tuples.
    """
    if not transcript.strip():
        logger.warning(f"Empty transcript for {audio_path}, skipping alignment")
        return []

    try:
        # Load WhisperX alignment model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        align_model = whisperx.load_align_model(language_code="en", device=device)
        audio = whisperx.load_audio(audio_path)
        result = whisperx.align([{"text": transcript}], align_model, audio, device=device)
        
        aligned = []
        for segment in result["segments"]:
            for word in segment["words"]:
                if "start" in word and "end" in word:  # Ensure timing exists
                    aligned.append((word["word"], word["start"], word["end"]))
        
        logger.info(f"Aligned {len(aligned)} words for {audio_path}")
        return aligned
    except Exception as e:
        logger.error(f"Forced alignment failed for {audio_path}: {str(e)}")
        return []

def save_alignment_result(alignment: List[Tuple[str, float, float]], out_path: str) -> None:
    """
    Save alignment data to a CSV file.

    Args:
        alignment (List[Tuple[str, float, float]]): List of (word, start, end) tuples.
        out_path (str): Path to save the alignment CSV.
    """
    try:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("word,start,end\n")
            for word, start, end in alignment:
                f.write(f"{word},{start:.2f},{end:.2f}\n")
        logger.info(f"Saved alignment to {out_path}")
    except Exception as e:
        logger.error(f"Failed to save alignment to {out_path}: {str(e)}")

########################################
# 3. Main Data Preparation
########################################

def load_metadata(csv_path: str) -> pd.DataFrame:
    """
    Load metadata from a CSV file.

    Args:
        csv_path (str): Path to the metadata CSV.

    Returns:
        pd.DataFrame: Loaded metadata.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    required_cols = {"participant_id", "audio_path"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    return df

def process_single_audio(
    pid: str,
    audio_path: str,
    transcript_path: Optional[str],
    out_dir: str
) -> None:
    """
    Process a single audio file: diarize, extract speaker audio, align, and save results.

    Args:
        pid (str): Participant ID.
        audio_path (str): Path to the raw audio file.
        transcript_path (Optional[str]): Path to the transcript file, if available.
        out_dir (str): Directory to save processed files.
    """
    logger.info(f"Processing participant {pid}: {audio_path}")
    
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return

    # Step 1: Diarize
    segments = perform_speaker_diarization(audio_path)
    if not segments or all(s == (0, 0) for s in segments):
        logger.warning(f"No valid speaker segments for {pid}, skipping further processing")
        return

    # Step 2: Extract & save clean audio
    try:
        waveform, sr = extract_speaker_audio(audio_path, segments)
        clean_wav_path = os.path.join(out_dir, f"{pid}_clean.wav")
        Path(clean_wav_path).parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(clean_wav_path, waveform, sample_rate=sr)
        logger.info(f"Saved clean audio to {clean_wav_path}")
    except RuntimeError as e:
        logger.error(f"Skipping {pid} due to audio extraction error: {str(e)}")
        return

    # Step 3: Forced alignment
    alignment = []
    if transcript_path and os.path.exists(transcript_path):
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript = f.read().strip()
            alignment = perform_forced_alignment(clean_wav_path, transcript)
        except Exception as e:
            logger.error(f"Transcript processing failed for {transcript_path}: {str(e)}")

    # Step 4: Save alignment
    if alignment:
        align_out = os.path.join(out_dir, f"{pid}_alignment.csv")
        save_alignment_result(alignment, align_out)

def main(metadata_csv: str, out_dir: str) -> None:
    """
    Main function to process all audio files in the metadata CSV.

    Args:
        metadata_csv (str): Path to the metadata CSV.
        out_dir (str): Directory to save processed files.
    """
    df = load_metadata(metadata_csv)
    total = len(df)
    for idx, row in df.iterrows():
        pid = row["participant_id"]
        audio_path = row["audio_path"]
        transcript_path = row.get("transcript_path", None)
        logger.info(f"Processing {idx + 1}/{total}")
        process_single_audio(pid, audio_path, transcript_path, out_dir)

########################################
# 4. Command-Line Interface
########################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full data preparation: diarization + alignment")
    parser.add_argument("--metadata", type=str, required=True, help="Path to CSV with audio/transcript info")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store processed files")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    args = parser.parse_args()

    # Set logging level from CLI
    logger.setLevel(getattr(logging, args.log_level.upper()))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args.metadata, args.output_dir)
