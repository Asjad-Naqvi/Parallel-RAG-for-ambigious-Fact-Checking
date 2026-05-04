# Parallel RAG for Ambiguous Fact Checking

> **📧 Data and Checkpoints**: Pre-trained model checkpoints and datasets can be requested via email at **asjadnaqvi1104@gmail.com**

## Overview

This repository contains the implementation of a parallel Retrieval Augmented Generation (RAG) system for fact-checking ambiguous claims. The project combines evidence retrieval, classification, and veracity prediction to handle claims that may have multiple interpretations or ambiguous phrasings.

## Project Structure

```
.
├── src/                          # Source code
│   ├── data/                     # Data preparation and processing
│   │   ├── prepare_ambifc.py    # Prepare the AmbiFCDataset
│   │   ├── make_pipeline_data.py # Create pipeline-ready data
│   │   └── inspect_processed_ambifc.py # Inspect processed data
│   │
│   ├── models/                   # Model training scripts
│   │   ├── train_veracity.py    # Train veracity prediction models
│   │   ├── train_evidence.py    # Train binary evidence classification
│   │   └── train_evidence_ternary.py # Train ternary evidence classification
│   │
│   ├── retrieval/                # Retrieval Augmented Generation
│   │   ├── build_indexes.py     # Build search indexes (BM25, FAISS)
│   │   ├── make_rag_data_mp.py  # Create RAG data with multiprocessing
│   │   ├── test_retrieval.py    # Test retrieval performance
│   │   └── pdc_retrieval_benchmark.py # Benchmark retrieval system
│   │
│   ├── evaluation/               # Evaluation metrics
│   │   └── metrics.py           # Custom evaluation metrics
│   │
│   └── utils/                    # Utility functions
│       └── check_project_state.py # Check project setup and dependencies
│
├── results/                       # Results and outputs
│   ├── benchmark/                # Retrieval benchmarks
│   ├── original_paper/           # Results from original paper experiments
│   ├── pdc_rag/                  # RAG pipeline results
│   └── tables/                   # Result tables and summaries
│
├── logs/                         # Training and experiment logs
├── requirements.txt              # Core dependencies
├── requirements_full.txt         # Full dependency list with versions
└── README.md                     # This file
```

## Key Components

### Data Preparation (`src/data/`)

- **prepare_ambifc.py**: Processes and prepares the ambiguous fact-checking dataset
- **make_pipeline_data.py**: Converts data into formats suitable for pipeline training
- **inspect_processed_ambifc.py**: Analysis tools for inspecting processed datasets

### Model Training (`src/models/`)

Implements DeBERTa-based models for:

- **Veracity Prediction**: Predicts claim truthfulness (SUPPORTS, REFUTES, NEI)
- **Binary Evidence Classification**: Classifies if evidence is relevant
- **Ternary Evidence Classification**: Fine-grained evidence classification

Supports multiple configurations:

- Full-text processing
- Distilled models for efficiency
- Pipeline-based training

### Retrieval System (`src/retrieval/`)

Implements parallel RAG with:

- **BM25 Indexing**: Traditional information retrieval
- **FAISS Indexing**: Dense vector-based retrieval using sentence transformers
- **Multiprocessing Support**: For efficient large-scale retrieval
- **Benchmarking**: Performance evaluation of retrieval components

### Evaluation (`src/evaluation/`)

Custom metrics for:

- Precision, Recall, F1-scores
- Macro and weighted averages
- Multi-class classification metrics

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU support)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/Asjad-Naqvi/Parallel-RAG-for-ambigious-Fact-Checking.git
cd Parallel-RAG-for-ambigious-Fact-Checking
```

2. Install dependencies:

```bash
# Install core requirements
pip install -r requirements.txt

# Or use full requirements with pinned versions
pip install -r requirements_full.txt
```

3. Verify installation:

```bash
python src/utils/check_project_state.py
```

## Usage

### Data Preparation

```bash
# Prepare the ambiguous fact-checking dataset
python src/data/prepare_ambifc.py

# Create pipeline training data
python src/data/make_pipeline_data.py

# Inspect processed data
python src/data/inspect_processed_ambifc.py
```

### Building Retrieval Indexes

```bash
# Build BM25 and FAISS indexes
python src/retrieval/build_indexes.py

# Create RAG data with multiprocessing
python src/retrieval/make_rag_data_mp.py
```

### Training Models

```bash
# Train veracity prediction model
python src/models/train_veracity.py

# Train binary evidence classifier
python src/models/train_evidence.py

# Train ternary evidence classifier
python src/models/train_evidence_ternary.py
```

### Evaluation

```bash
# Test retrieval performance
python src/retrieval/test_retrieval.py

# Run retrieval benchmarks
python src/retrieval/pdc_retrieval_benchmark.py
```

## Dependencies

### Core Libraries

- **PyTorch**: Deep learning framework
- **Transformers**: Pre-trained models (DeBERTa)
- **Sentence-Transformers**: Dense retrieval embeddings
- **Rank-BM25**: Sparse retrieval
- **FAISS**: Vector similarity search
- **Datasets**: HuggingFace datasets
- **Scikit-learn**: ML utilities
- **Pandas/NumPy**: Data processing

See `requirements.txt` for full dependency list.

## Experiments

The results directory contains outputs from various experiments:

### Benchmark Results

- Retrieval performance metrics
- Sample retrievals and profiling data

### Original Paper Experiments

- Single full-text models
- Distilled model variants
- Binary and ternary evidence classification
- Pipeline-based approaches

### RAG Pipeline Results

- Multi-process RAG training/dev/test results
- Profile breakdowns
- Retrieval samples

## Model Checkpoints & Datasets

**Pre-trained model checkpoints and datasets are not included in this repository due to size constraints.**

To request:

- Pre-trained model weights
- Full datasets used in experiments
- Processed data files

**Please email: asjadnaqvi1104@gmail.com**

Include in your email:

- Your name and affiliation
- Intended use case
- Which specific checkpoints/datasets you need

## Performance

Results from various model configurations are available in the `results/` directory:

- Benchmark tables with accuracy, precision, recall, and F1 scores
- Detailed experiment summaries
- Retrieval performance metrics

## Citation

If you use this code or models in your research, please cite:

```bibtex
@repository{Naqvi2024ParallelRAG,
  author = {Naqvi, Asjad},
  title = {Parallel RAG for Ambiguous Fact Checking},
  year = {2024},
  url = {https://github.com/Asjad-Naqvi/Parallel-RAG-for-ambigious-Fact-Checking}
}
```

## License

This project is provided for research purposes. Please contact the author for licensing information.

## Contact

**Author**: Asjad Naqvi  
**Email**: asjadnaqvi1104@gmail.com
**Author**: Sameer Khan  
**Email**: khanhamzazai5@gmail.com


For questions, bug reports, or collaboration inquiries, please reach out via email.

## Acknowledgments

- DeBERTa models from Microsoft
- Sentence-Transformers from SBERT
- FAISS from Meta AI
- BM25 implementation from rank-bm25
- Fact-checking dataset from AmbiFCDataset authors

---

**Last Updated**: May 2024
