# Psychosis Speech Analysis

This repository contains code for the paper "Uncertainty Modeling in Multimodal Speech Analysis Across the Psychosis Spectrum." It implements a multimodal machine learning pipeline to analyze speech patterns using audio and text data.

## Structure
- `full_data_preparation.py`: Preprocesses raw audio with diarization and alignment.
- `scripts/`:
  - `model_tcf.py`: Temporal Context Fusion (TCF) model with uncertainty modeling.
  - `train_tcf.py`: Trains the TCF model.
  - `fusion_models.py`: Early and late fusion baseline models.
  - `dataset.py`: Dataset class for loading audio/text embeddings.
  - `feature_extraction.py`: Extracts embeddings using `wav2vec 2.0` and `XLM-RoBERTa`.
  - `train_fusion.py`: Trains fusion models.
  - `evaluate_models.py`: Evaluates all models (TCF, early/late fusion).
  - `cross_corpus_train.py`: Cross-corpus training for TCF.
  - `baseline_models.py`: Evaluates RF, LDA, SVM baselines.

## Usage
1. Preprocess: `python full_data_preparation.py --metadata metadata.csv --output_dir data/processed`
2. Extract Features: `python scripts/feature_extraction.py --metadata metadata.csv`
3. Train TCF: `python scripts/train_tcf.py --metadata metadata.csv`
4. Train Fusion: `python scripts/train_fusion.py --metadata metadata.csv --model_type early`
5. Evaluate: `python scripts/evaluate_models.py --metadata metadata.csv --model_path checkpoints/tcf.ckpt --model_type coldtcf`
6. Cross-Corpus: `python cross_corpus_train.py --metadata_dir path/to/metadata/`
7. Baselines: `python baseline_models.py --metadata metadata.csv`

## Requirements
See `requirements.txt`.

## Notes
- PELICAN is an in-house tool; contact authors for access.
- Requires Hugging Face token for `pyannote.audio`.
