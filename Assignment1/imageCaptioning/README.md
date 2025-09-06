## Remote Sensing Image Captioning Assignment

This repository contains code for image captioning on the RSICD dataset using deep learning models with LSTM and Transformer decoders, and ResNet-18 and MobileNet encoder backbones.

### Directory Structure

```
imageCaptioning/
│
├── config.py                # Configuration settings and logger setup
├── main.py                  # Entry point for training and evaluation
│
├── models/                  # Model definitions
│   ├── encoder.py           # CNN encoder (ResNet-18, MobileNet)
│   ├── lstmDecoder.py       # LSTM-based decoder
│   ├── transformerDecoder.py# Transformer-based decoder
│   └── captioner.py         # Combines encoder and decoder
│
├── data/                    # Data handling and preprocessing
│   ├── dataset.py           # Custom dataset class for RSICD
│   ├── preprocess.py        # Preprocessing utilities
│   ├── prepareFromCSV.py    # Prepare images and annotations from CSV
│   └── vocabulary.py        # Vocabulary/tokenizer management
│
├── evaluation/              # Evaluation scripts and metrics
│   ├── metrics.py           # BLEU, METEOR, and other metrics
│   ├── decoding.py          # Decoding utilities for inference
│   └── getCaptions.py       # Caption extraction and formatting
│
├── training/                # Training utilities
│   ├── train.py             # Training loop and logic
│   ├── lossFunction.py      # Loss functions
│   ├── optimizer.py         # Optimizer setup
│   └── utils.py             # Helper functions
│
├── outputs/                 # Model outputs, predictions, and statistics
├── checkpoints/             # Saved model weights
└── rsicdDataset/            # Dataset files and images (train, valid, test splits)
```

### How to Run

#### 1. Install Requirements
```bash
pip install -r requirements.txt
```

#### 2. Prepare Dataset
Place the RSICD dataset in the `rsicdDataset/` folder. Use `prepareFromCSV.py` to preprocess and organize images and annotations.

#### 3. Train a Model
Run training from the root directory:
```bash
python -m imageCaptioning.main --model-type lstm --encoder-name resnet18
```
Replace `lstm` and `resnet18` with `transformer` and/or `mobilenet` as needed.

#### 4. Evaluate and Decode Captions
Run decoding and evaluation:
```bash
python -m imageCaptioning.evaluation.decoding --model-type lstm --encoder-name resnet18
```

#### 5. View Outputs
Model outputs, metrics, and qualitative results are saved in the `outputs/` directory. Check logs for training and evaluation details.

### General Information

- The assignment demonstrates modular deep learning code for image captioning in remote sensing.
- Models support both LSTM and Transformer decoders, and ResNet-18 and MobileNet encoders.
- Evaluation includes BLEU-4 and METEOR metrics, qualitative analysis, error slice analysis, and explainability (Grad-CAM, token occlusion, attention maps).
- The code is organized for clarity, extensibility, and reproducibility.

For more details, see the code comments and the assignment notebook/report.
