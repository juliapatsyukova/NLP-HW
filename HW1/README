# ProtBERT Embedding Pipeline for Amyloid Sequences

This project contains a Python pipeline to compute ProtBERT embeddings for amino acid sequences to distinguish between amyloid and non-amyloid types. The script processes a CSV file of sequences, generates embeddings using a pre-trained ProtBERT model, and outputs the results in a `.parquet` file suitable for use with visualization tools like Nomic Embedding Atlas.

## Components

*   `pipeline_amyloid.py`: The main script that loads data, tokenizes sequences, computes embeddings, and saves the output.
*   `waltzdb.csv`: The input dataset containing amino acid sequences and their corresponding classification (`amyloid` or `non-amyloid`).
*   `requirements.txt`: A list of Python packages required to run the pipeline.

## Requirements

The pipeline requires Python 3 and the packages listed in `requirements.txt`. The main dependencies are:

*   `torch`
*   `transformers`
*   `pandas`
*   `numpy`
*   `scikit-learn`
*   `pyarrow`

## Installation

To install the required dependencies, run the following command:

```sh
pip install -r requirements.txt
```

## Usage

Place the `waltzdb.csv` file in the same directory as the script. Execute the pipeline by running:

```sh
python3 pipeline_amyloid.py
```

The script will create an `output/` directory containing the `embeddings.parquet` file. This file includes the original data along with the generated embeddings and a 2D PCA projection (`projection_x`, `projection_y`).
