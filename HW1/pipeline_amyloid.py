#!/usr/bin/env python3
"""
ProtBERT Embedding Pipeline for Amyloid Sequence Classification

This pipeline:
1. Loads waltzdb.csv (amino acid sequences with amyloid/non-amyloid labels)
2. Computes ProtBERT embeddings for each sequence (protein-specific BERT)
3. Outputs an Embedding Atlas-compatible .parquet file

The pipeline is CPU-friendly and uses small batches to minimize memory usage.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from sklearn.decomposition import PCA


warnings.filterwarnings("ignore")

# Try to import required packages
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except ImportError as e:
    print(f"Error: Required package missing: {e}")
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

# Use ProtBERT for protein sequences (pre-trained on UniRef100)
MODEL_NAME = "Rostlab/prot_bert"  # ProtBERT: protein-specific BERT
BATCH_SIZE = 8  # Small batches for CPU
DEVICE = "cpu"  # Force CPU mode
OUTPUT_DIR = Path("./output")
OUTPUT_FILE = OUTPUT_DIR / "embeddings.parquet"

# ============================================================================
# Helper Functions
# ============================================================================

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load and clean the waltzdb.csv file.
    
    Args:
        csv_path: Path to waltzdb.csv
        
    Returns:
        DataFrame with columns: id, sequence, label
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Rename columns for consistency
    df = df.rename(columns={
        "Sequence": "sequence",
        "Classification": "label"
    })
    
    # Keep only required columns
    df = df[["sequence", "label"]].copy()
    
    # Drop empty sequences
    df = df[df["sequence"].notna() & (df["sequence"].str.len() > 0)]
    
    # Normalize labels to binary (0/1)
    df["label"] = (df["label"] == "amyloid").astype(int)
    
    # Create stable ID based on row index
    df["id"] = range(len(df))
    
    # Reorder columns
    df = df[["id", "sequence", "label"]]
    
    print(f"✓ Loaded {len(df)} sequences")
    print(f"  - Amyloid (1): {(df['label'] == 1).sum()}")
    print(f"  - Non-amyloid (0): {(df['label'] == 0).sum()}")
    
    return df


def tokenize_sequences(sequences: List[str], tokenizer) -> dict:
    """
    Tokenize protein sequences for ProtBERT.
    
    ProtBERT expects amino acids separated by spaces.
    
    Args:
        sequences: List of amino acid sequences
        tokenizer: HuggingFace tokenizer
        
    Returns:
        Tokenized sequences (dict with input_ids, attention_mask, etc.)
    """
    # Convert sequences to space-separated amino acids for ProtBERT
    spaced_seqs = [" ".join(seq) for seq in sequences]
    
    # Tokenize
    tokens = tokenizer(
        spaced_seqs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )
    
    return tokens


def compute_embeddings(
    sequences: List[str],
    model,
    tokenizer,
    batch_size: int = 8,
    device: str = "cpu"
) -> np.ndarray:
    """
    Compute ProtBERT embeddings for a list of protein sequences.
    
    Uses CLS token embedding (first token) as the sequence representation.
    
    Args:
        sequences: List of amino acid sequences
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size for processing
        device: Device to use (cpu or cuda)
        
    Returns:
        Array of embeddings (n_sequences, embedding_dim)
    """
    embeddings = []
    model.to(device)
    model.eval()
    
    total_batches = (len(sequences) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in range(0, len(sequences), batch_size):
            batch_seqs = sequences[batch_idx:batch_idx + batch_size]
            
            # Tokenize batch
            tokens = tokenize_sequences(batch_seqs, tokenizer)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            
            # Forward pass
            outputs = model(**tokens)
            
            # Extract CLS token embedding (first token of last hidden state)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)
            
            # Progress
            current_batch = (batch_idx // batch_size) + 1
            print(f"  Batch {current_batch}/{total_batches} ({len(batch_seqs)} sequences)")
    
    # Concatenate all batches
    embeddings = np.vstack(embeddings)
    
    return embeddings


def save_embeddings(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    output_path: Path
) -> None:
    """
    Save embeddings to Embedding Atlas-compatible .parquet file
    with optional 2D projection.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # copy base data
    output_df = df.copy()
    output_df["embedding"] = [emb.tolist() for emb in embeddings]

    # =========================
    # 2D PROJECTION (PCA)
    # =========================
    print("Computing 2D projection (PCA)...")
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(embeddings)

    output_df["projection_x"] = proj[:, 0]
    output_df["projection_y"] = proj[:, 1]

    # =========================
    # SAVE
    # =========================
    output_df.to_parquet(output_path, index=False)

    print(f"\n✓ Saved embeddings to {output_path}")
    print(f"  Shape: {output_df.shape}")
    print(f"  Columns: {list(output_df.columns)}")
    print(f"  Embedding dim: {embeddings.shape[1]}")


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Run the complete embedding pipeline."""
    
    print("=" * 70)
    print("ProtBERT Embedding Pipeline")
    print("=" * 70)
    
    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    csv_path = "waltzdb.csv"
    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)
    
    df = load_data(csv_path)
    
    # Step 2: Load model and tokenizer
    print(f"\n[Step 2] Loading ProtBERT model ({MODEL_NAME})...")
    print("  (This may take a minute on first run)")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
        model = AutoModel.from_pretrained(MODEL_NAME)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Step 3: Compute embeddings
    print(f"\n[Step 3] Computing ProtBERT embeddings (batch_size={BATCH_SIZE}, device={DEVICE})...")
    sequences = df["sequence"].tolist()
    embeddings = compute_embeddings(
        sequences,
        model,
        tokenizer,
        batch_size=BATCH_SIZE,
        device=DEVICE
    )
    print(f"✓ Computed {len(embeddings)} embeddings (dimension: {embeddings.shape[1]})")
    
    # Step 4: Save output
    print(f"\n[Step 4] Saving output to Embedding Atlas format...")
    save_embeddings(df, embeddings, OUTPUT_FILE)
    
    print("\n" + "=" * 70)
    print("Pipeline completed successfully!")
    print("=" * 70)
    print(f"\nOutput file: {OUTPUT_FILE.absolute()}")
    print("\nNext steps:")
    print("1. Open Embedding Atlas (https://atlas.nomic.ai/)")
    print("2. Upload the embeddings.parquet file")
    print("3. Color by 'label' to visualize amyloid vs non-amyloid clusters")
    print("\nNote: This pipeline uses ProtBERT (protein-specific BERT) to compute")
    print("learned representations of amyloid/non-amyloid sequences. The embedding")
    print("space should reveal whether the model captures amyloid propensity.")


if __name__ == "__main__":
    main()
