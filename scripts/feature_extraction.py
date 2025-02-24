import os
import numpy as np
import torch
import torchaudio
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2Model, AutoTokenizer, AutoModel
import librosa
import logging
from typing import Optional, Tuple
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Class to handle audio and text feature extraction using pretrained models."""
    
    def __init__(
        self,
        audio_model_name: str = "facebook/wav2vec2-large-960h",
        text_model_name: str = "xlm-roberta-base",
        audio_sample_rate: int = 16000,
        max_text_length: int = 512
    ):
        """
        Initialize feature extractor with pretrained models.
        
        Args:
            audio_model_name (str): Hugging Face model name for audio (default: "facebook/wav2vec2-large-960h").
            text_model_name (str): Hugging Face model name for text (default: "xlm-roberta-base").
            audio_sample_rate (int): Target sample rate for audio (default: 16000).
            max_text_length (int): Max length for text tokenization (default: 512).
        """
        self.audio_sample_rate = audio_sample_rate
        self.max_text_length = max_text_length
        
        # Load audio model and processor
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(audio_model_name)
            self.wav2vec_model = Wav2Vec2Model.from_pretrained(audio_model_name)
            self.wav2vec_model.eval()
        except Exception as e:
            logger.error(f"Failed to load audio model {audio_model_name}: {str(e)}")
            raise
        
        # Load text model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
            self.text_model = AutoModel.from_pretrained(text_model_name)
            self.text_model.eval()
        except Exception as e:
            logger.error(f"Failed to load text model {text_model_name}: {str(e)}")
            raise

    def extract_wav2vec_embeddings(self, audio_path: str) -> np.ndarray:
        """
        Extract audio embeddings using Wav2Vec2 model.
        
        Args:
            audio_path (str): Path to the audio file.
        
        Returns:
            np.ndarray: Audio embedding of shape (feature_dim,) or zeros if extraction fails.
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.audio_sample_rate)
            inputs = self.processor(y, sampling_rate=sr, return_tensors="pt")
            with torch.no_grad():
                outputs = self.wav2vec_model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
            return emb
        except Exception as e:
            logger.warning(f"Failed to extract audio embedding from {audio_path}: {str(e)}")
            return np.zeros(1024)  # Default size matches Wav2Vec2-large output

    def extract_text_embeddings(self, txt_path: str) -> np.ndarray:
        """
        Extract text embeddings using a transformer model (CLS token).
        
        Args:
            txt_path (str): Path to the text file.
        
        Returns:
            np.ndarray: Text embedding of shape (feature_dim,) or zeros if extraction fails.
        """
        if not os.path.exists(txt_path):
            logger.warning(f"Text file not found: {txt_path}")
            return np.zeros(768)  # Default size matches XLM-RoBERTa output
        
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if not text:
                logger.warning(f"Empty text file: {txt_path}")
                return np.zeros(768)
            
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_text_length,
                padding=True
            )
            with torch.no_grad():
                outputs = self.text_model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
            return cls_emb
        except Exception as e:
            logger.warning(f"Failed to extract text embedding from {txt_path}: {str(e)}")
            return np.zeros(768)

def main_feature_extraction(
    metadata_csv: str,
    output_dir: str = "data/processed",
    audio_model_name: str = "facebook/wav2vec2-large-960h",
    text_model_name: str = "xlm-roberta-base"
) -> None:
    """
    Extract and save audio and text embeddings for all participants in the metadata CSV.
    
    Args:
        metadata_csv (str): Path to the metadata CSV file.
        output_dir (str): Directory to save extracted embeddings (default: "data/processed").
        audio_model_name (str): Audio model name for feature extraction.
        text_model_name (str): Text model name for feature extraction.
    
    Raises:
        FileNotFoundError: If metadata CSV does not exist.
        ValueError: If required columns are missing in the CSV.
    """
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    
    df = pd.read_csv(metadata_csv)
    required_cols = {"participant_id", "transcript_path"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize extractor
    extractor = FeatureExtractor(audio_model_name=audio_model_name, text_model_name=text_model_name)

    # Process each row
    for idx, row in df.iterrows():
        pid = row["participant_id"]
        audio_clean = os.path.join(output_dir, f"{pid}_clean.wav")
        transcript_path = row["transcript_path"]

        logger.info(f"Processing participant {pid} (row {idx + 1}/{len(df)})")

        # Extract and save audio embedding
        audio_emb = extractor.extract_wav2vec_embeddings(audio_clean) if os.path.exists(audio_clean) else np.zeros(1024)
        out_audio = os.path.join(output_dir, f"{pid}_audio_emb.npy")
        np.save(out_audio, audio_emb)
        logger.info(f"Saved audio embedding to {out_audio}")

        # Extract and save text embedding
        text_emb = extractor.extract_text_embeddings(transcript_path) if pd.notna(transcript_path) else np.zeros(768)
        out_text = os.path.join(output_dir, f"{pid}_text_emb.npy")
        np.save(out_text, text_emb)
        logger.info(f"Saved text embedding to {out_text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract audio and text embeddings from metadata.")
    parser.add_argument("--metadata", type=str, default="metadata.csv", help="Path to metadata CSV")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory for embeddings")
    parser.add_argument("--audio_model", type=str, default="facebook/wav2vec2-large-960h", help="Audio model name")
    parser.add_argument("--text_model", type=str, default="xlm-roberta-base", help="Text model name")
    args = parser.parse_args()

    main_feature_extraction(
        metadata_csv=args.metadata,
        output_dir=args.output_dir,
        audio_model_name=args.audio_model,
        text_model_name=args.text_model
    )
