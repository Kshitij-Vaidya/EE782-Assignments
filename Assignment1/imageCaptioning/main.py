import os
from torch.utils.data import DataLoader
import argparse

# Import project modules
from imageCaptioning.config import getCustomLogger, DATA_ROOT, OUTPUT_DIRECTORY, DEVICE
from imageCaptioning.data.dataset import RSICDDataset
from imageCaptioning.data.vocabulary import Vocabulary
from imageCaptioning.data.preprocess import getTransforms
from imageCaptioning.models.captioner import Captioner
from imageCaptioning.training.train import trainModel

LOGGER = getCustomLogger("Main")

def loadVocabulary(vocabPath: str) -> Vocabulary:
    """Load the preprocessed vocabulary"""
    if not os.path.exists(vocabPath):
        raise FileNotFoundError(f"Vocabulary file not found at {vocabPath}")
    
    LOGGER.info(f"Loading vocabulary from {vocabPath}")
    vocabulary = Vocabulary.load(vocabPath)
    LOGGER.info(f"Loaded vocabulary with {len(vocabulary)} tokens")
    return vocabulary

def createDataLoaders(vocabulary: Vocabulary, 
                     batchSize: int = 32,
                     maxLength: int = 24,
                     numWorkers: int = 4) -> tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders"""
    
    # Define transforms
    transform = getTransforms()
    
    # Create datasets
    trainDataset = RSICDDataset(
        root=DATA_ROOT,
        split="train",
        vocab=vocabulary,
        transform=transform,
        maxLength=maxLength
    )
    
    valDataset = RSICDDataset(
        root=DATA_ROOT,
        split="valid",
        vocab=vocabulary,
        transform=transform,
        maxLength=maxLength
    )
    
    LOGGER.info(f"Training dataset size: {len(trainDataset)}")
    LOGGER.info(f"Validation dataset size: {len(valDataset)}")
    
    # Create data loaders
    trainLoader = DataLoader(
        trainDataset,
        batch_size=batchSize,
        shuffle=True,
        num_workers=numWorkers,
        pin_memory=True if DEVICE == "cuda" else False
    )
    
    valLoader = DataLoader(
        valDataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=numWorkers,
        pin_memory=True if DEVICE == "cuda" else False
    )
    
    return trainLoader, valLoader

def main():
    parser = argparse.ArgumentParser(description="Train Image Captioning Model on RSICD Dataset")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr-cnn", type=float, default=1e-4, help="Learning rate for CNN encoder")
    parser.add_argument("--lr-decoder", type=float, default=2e-4, help="Learning rate for decoder")
    parser.add_argument("--lr-transformer", type=float, default=2e-5, help="Learning rate for transformer")
    parser.add_argument("--model-type", type=str, default="lstm", choices=["lstm", "transformer"], 
                       help="Type of decoder to use")
    parser.add_argument("--encoder-name", type=str, default="resnet18", 
                       help="CNN encoder architecture")
    parser.add_argument("--max-length", type=int, default=24, help="Maximum caption length")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", 
                       help="Directory to save checkpoints")
    parser.add_argument("--finetune", action="store_true", help="Fine-tune the encoder")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    LOGGER.info("Starting Image Captioning Training on RSICD Dataset")
    LOGGER.info(f"Device: {DEVICE}")
    LOGGER.info(f"Model Type: {args.model_type}")
    LOGGER.info(f"Encoder: {args.encoder_name}")
    LOGGER.info(f"Fine-tune Encoder: {args.finetune}")
    LOGGER.info(f"Batch Size: {args.batch_size}")
    LOGGER.info(f"Epochs: {args.epochs}")
    
    # Load vocabulary
    vocabPath = os.path.join(OUTPUT_DIRECTORY, "vocab.json")
    vocabulary = loadVocabulary(vocabPath)
    
    # Create data loaders
    trainLoader, valLoader = createDataLoaders(
        vocabulary=vocabulary,
        batchSize=args.batch_size,
        maxLength=args.max_length,
        numWorkers=args.num_workers
    )
    
    # Initialize model
    model = Captioner(
        vocabSize=len(vocabulary),
        modelType=args.model_type,
        encoderName=args.encoder_name,
        finetune=args.finetune
    )
    
    # Move model to device
    model = model.to(DEVICE)
    
    # Log model information
    totalParams = sum(p.numel() for p in model.parameters())
    trainableParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info(f"Total parameters: {totalParams:,}")
    LOGGER.info(f"Trainable parameters: {trainableParams:,}")
    
    # Define checkpoint path
    checkpointPath = os.path.join(args.checkpoint_dir, f"model_{args.model_type}_{args.encoder_name}.pt")
    
    # Get padding index from vocabulary
    paddingIndex = vocabulary.STOI.get(vocabulary.padToken, 0)
    
    # Train the model
    LOGGER.info("Starting training...")
    trainModel(
        model=model,
        trainLoader=trainLoader,
        valLoader=valLoader,
        paddingIndex=paddingIndex,
        epochs=args.epochs,
        lrCNN=args.lr_cnn,
        lrDecoder=args.lr_decoder,
        lrTransformer=args.lr_transformer
    )
    
    LOGGER.info("Training completed!")
    LOGGER.info(f"Best model saved to: {checkpointPath}")

if __name__ == "__main__":
    main()
