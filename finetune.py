import os
import argparse
from datetime import datetime
import torch
from utils import set_seed
from model import Resnet, BERT, DINO_ViTB16, TRANSFORMER_MODEL_NAMES
from transformers import get_cosine_schedule_with_warmup


def get_model(
        model_type: str,
        num_classes: int,
        pretrained: bool = True,
        model_name_or_path: str = None,
        pretrained_checkpoint: str = None,
        dropout: float = None):
    """Instantiate a model by name."""
    if model_type in TRANSFORMER_MODEL_NAMES:
        model_name = model_name_or_path or TRANSFORMER_MODEL_NAMES[model_type]
        return BERT(output_dim=num_classes, pretrained=pretrained, model_name=model_name, dropout=dropout)
    if model_type in {'dino_vitb16', 'vit_b_16'}:
        return DINO_ViTB16(
            output_dim=num_classes,
            pretrained=pretrained,
            checkpoint_path=pretrained_checkpoint,
        )
    variant_map = {'resnet18': 18, 'resnet34': 34, 'resnet50': 50}
    if model_type not in variant_map:
        choices = list(variant_map.keys()) + ['dino_vitb16', 'vit_b_16'] + list(TRANSFORMER_MODEL_NAMES.keys())
        raise ValueError(f"Unknown model type: {model_type}. Choose from {choices}")
    return Resnet(variant=variant_map[model_type], output_dim=num_classes, pretrained=pretrained)


def get_optimizer(model, optimizer_name: str, lr: float,
                  momentum: float = 0.9, weight_decay: float = 0.0,
                  adam_beta1: float = 0.9, adam_beta2: float = 0.999) -> torch.optim.Optimizer:
    """Create an optimizer for the model."""
    params = model.model.parameters()
    if optimizer_name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay,
                                betas=(adam_beta1, adam_beta2))
    elif optimizer_name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay,
                                 betas=(adam_beta1, adam_beta2))
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Choose from ['sgd', 'adam', 'adamw']")


def get_scheduler(optimizer: torch.optim.Optimizer, schedule_name: str,
                  total_steps: int, warmup_steps: int = 0):
    """Create a learning-rate scheduler."""
    warmup_steps = max(0, int(warmup_steps or 0))
    total_steps = max(1, int(total_steps))
    if warmup_steps >= total_steps:
        raise ValueError(
            f"warmup_steps ({warmup_steps}) must be smaller than total_steps ({total_steps})."
        )
    if schedule_name in {None, '', 'none'}:
        if warmup_steps:
            raise ValueError("warmup_steps requires a learning-rate schedule; use --lr_schedule cosine.")
        return None
    if schedule_name == 'cosine':
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    raise ValueError(f"Unknown lr_schedule: {schedule_name}. Choose from ['none', 'cosine']")
from data import get_data_obj
from metric import get_bce_loss, get_ce_loss
from settings import DEVICE

# Datasets that are image-based and use create_image_loaders().
# Text / embedding-based datasets use create_loaders() instead.
_IMAGE_DATASETS = {'wb', 'celeba'}

# Datasets that require load_bert_features() rather than load().
_BERT_FEATURE_DATASETS = {'multinli'}
_MODEL_CHOICES = ['resnet18', 'resnet34', 'resnet50', 'dino_vitb16', 'vit_b_16'] + list(TRANSFORMER_MODEL_NAMES.keys())


def finetune(args):
    """
    Finetune a model on the specified dataset.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Dictionary with training information
    """
    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"Seed set to {args.seed}")
    
    # Set device
    device = torch.device(DEVICE)
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading dataset: {args.dataset}")
    data = get_data_obj(args.dataset)
    if args.dataset in _BERT_FEATURE_DATASETS:
        feature_model = args.feature_model or args.model
        data.load_bert_features(
            feature_model=feature_model,
            features_dir=args.features_dir,
            max_seq_length=args.max_seq_length,
        )
    else:
        data.load()

    # Optionally reset train/val split using the run seed for reproducibility.
    if args.reset_split:
        data.reset_train_val_split(args.seed)
        print(f"Train/val split reset [random resplit] with seed {args.seed}")

    # Get num_classes from dataset
    num_classes = data.num_classes
    print(f"Dataset has {num_classes} output class(es)")
    
    # Create data loaders — image datasets support augmentation/normalisation;
    # text / embedding datasets use the simpler tensor-based loaders.
    if args.dataset in _IMAGE_DATASETS:
        loaders = data.create_image_loaders(
            batch_size=args.batch_size,
            shuffle_train=True,
            augmentation=args.augmentation,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        )
        print(
            f"Created image loaders with batch size {args.batch_size}, "
            f"augmentation={args.augmentation}, num_workers={args.num_workers}"
        )
    else:
        if args.augmentation:
            print("Warning: --augmentation has no effect for non-image dataset; ignoring.")
        loaders = data.create_loaders(batch_size=args.batch_size, shuffle_train=True)
        print(f"Created tensor loaders with batch size {args.batch_size}")
    
    # Create model
    resolved_model_name = args.model_name_or_path or TRANSFORMER_MODEL_NAMES.get(args.model)
    if resolved_model_name:
        print(f"Creating {args.model} model ({resolved_model_name}) with {num_classes} classes")
    else:
        print(f"Creating {args.model} model with {num_classes} classes")
    model = get_model(
        model_type=args.model,
        num_classes=num_classes,
        pretrained=args.pretrained,
        model_name_or_path=args.model_name_or_path,
        pretrained_checkpoint=args.pretrained_checkpoint,
        dropout=args.dropout,
    )
    model.model.to(device)
    
    # Create optimizer
    optimizer = get_optimizer(
        model=model,
        optimizer_name=args.optimizer,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
    )
    print(f"Optimizer: {args.optimizer}, LR: {args.lr}, Weight Decay: {args.weight_decay}, Adam betas: ({args.adam_beta1}, {args.adam_beta2})")
    
    scheduler = get_scheduler(
        optimizer=optimizer,
        schedule_name=args.lr_schedule,
        total_steps=args.epochs * len(loaders['train']),
        warmup_steps=args.warmup_steps,
    )
    print(f"LR schedule: {args.lr_schedule}, warmup_steps={args.warmup_steps}")

    # Define loss function.
    # Binary datasets (num_classes == 1) → BCEWithLogitsLoss.
    # Multiclass datasets (num_classes > 1, e.g. MultiNLI with 3 classes) → CrossEntropyLoss.
    if num_classes == 1:
        loss_func = get_bce_loss(class_balance=False)
    else:
        loss_func = get_ce_loss(class_balance=False)
        print(f"Using CrossEntropyLoss for {num_classes}-class classification.")

    # Set up validation selection
    val_selection = args.val_selection
    val_criterion = None
    val_save_path = None
    if val_selection:
        if num_classes == 1:
            val_criterion = get_bce_loss(class_balance=True)
        else:
            val_criterion = get_ce_loss(class_balance=True)
        os.makedirs(args.save_dir, exist_ok=True)
        val_save_path = os.path.join(args.save_dir, "best_model_checkpoint.pt")
        print(f"Val selection enabled (class-balanced loss), checkpoint: {val_save_path}")
    
    # Train the model
    print(f"Starting training for {args.epochs} epochs...")
    model.train_model(
        train_loader=loaders['train'],
        optimizer=optimizer,
        loss_func=loss_func,
        epochs=args.epochs,
        val_loader=loaders.get('val'),
        val_selection=val_selection,
        val_criterion=val_criterion,
        save_path=val_save_path,
        early_stopping=args.early_stopping,
        scheduler=scheduler,
        use_amp=args.amp,
    )
    
    # Evaluate on test set if available
    test_loss = None
    if args.skip_test_eval:
        print("Skipping test-set evaluation after training.")
    elif 'test' in loaders:
        try:
            test_loss = model.evaluate(loaders['test'], loss_func, device, use_amp=args.amp)
        except TypeError:
            test_loss = model.evaluate(loaders['test'], loss_func, device)
        print(f"Test Loss: {test_loss:.4f}")
    
    # Prepare training info dictionary
    training_info = {
        'dataset': args.dataset,
        'model': args.model,
        'model_name_or_path': resolved_model_name,
        'feature_model': args.feature_model or args.model,
        'features_dir': args.features_dir,
        'max_seq_length': args.max_seq_length,
        'num_classes': num_classes,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'prefetch_factor': args.prefetch_factor,
        'amp': args.amp,
        'optimizer': args.optimizer,
        'lr': args.lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'adam_beta1': args.adam_beta1,
        'adam_beta2': args.adam_beta2,
        'lr_schedule': args.lr_schedule,
        'warmup_steps': args.warmup_steps,
        'dropout': args.dropout,
        'pretrained_checkpoint': args.pretrained_checkpoint,
        'seed': args.seed,
        'test_loss': test_loss,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Save model
    os.makedirs(args.save_dir, exist_ok=True)
    if args.model_filename:
        model_filename = args.model_filename
    else:
        model_filename = f"{args.model}_{args.dataset}_seed{args.seed}.pt"
    model_path = os.path.join(args.save_dir, model_filename)
    
    save_dict = {
        'model_state_dict': model.model.state_dict(),
        'training_info': training_info,
    }
    torch.save(save_dict, model_path)
    print(f"Model saved to {model_path}")
    
    return training_info


def main():
    parser = argparse.ArgumentParser(description="Finetune a model on a dataset")
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='wb', 
                        help="Dataset name (e.g., 'wb' for Waterbirds)")
    
    # Model arguments
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=_MODEL_CHOICES,
                        help="Model type")
    parser.add_argument('--model_name_or_path', type=str, default=None,
                        help="HuggingFace model name/path for transformer models. Defaults to the selected model alias.")
    parser.add_argument('--feature_model', type=str, default=None,
                        help="MultiNLI transformer feature cache alias/name. Defaults to --model.")
    parser.add_argument('--features_dir', type=str, default=None,
                        help="Directory containing cached MultiNLI transformer features.")
    parser.add_argument('--max_seq_length', type=int, default=128,
                        help="Sequence length used in cached MultiNLI transformer feature filenames.")
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help="Use pretrained weights")
    parser.add_argument('--pretrained_checkpoint', type=str, default=None,
                        help="Path to a pretrained image checkpoint, e.g. DINO ViT-B/16 .pth.")
    parser.add_argument('--dropout', type=float, default=None,
                        help="Override transformer dropout probabilities, e.g. 0.1 for DeBERTa-v3-base.")
    
    # Optimizer arguments
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam', 'adamw'],
                        help="Optimizer type. 'adamw' is recommended for transformer-based models (e.g. BERT on MultiNLI)")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help="Momentum (for SGD)")
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help="Weight decay (L2 regularization)")
    parser.add_argument('--adam_beta1', type=float, default=0.9,
                        help="Adam/AdamW beta1")
    parser.add_argument('--adam_beta2', type=float, default=0.999,
                        help="Adam/AdamW beta2")
    parser.add_argument('--lr_schedule', type=str, default='none', choices=['none', 'cosine'],
                        help="Learning-rate schedule.")
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help="Number of linear warmup steps before scheduled decay.")
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for data loaders")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="Number of image DataLoader workers.")
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help="Number of batches prefetched by each image DataLoader worker.")
    parser.add_argument('--amp', action='store_true', default=False,
                        help="Use CUDA automatic mixed precision during fine-tuning.")
    parser.add_argument('--augmentation', action='store_true', default=False,
                        help="Apply training data augmentation (RandomResizedCrop + RandomHorizontalFlip)")
    parser.add_argument('--val_selection', action='store_true', default=False,
                        help="Select best model based on class-balanced val accuracy")
    parser.add_argument('--early_stopping', type=int, default=None,
                        help="Stop training after this many epochs without val improvement. Requires --val_selection.")
    parser.add_argument('--skip_test_eval', action='store_true', default=False,
                        help="Skip final test-set evaluation during fine-tuning.")

    # Split-reset arguments
    parser.add_argument('--reset_split', action='store_true', default=False,
                        help="Reset the train/val split before training using the run seed.")

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                    help="Random seed for reproducibility")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device to use (cuda/cpu/mps)")
    parser.add_argument('--save_dir', type=str, default='models',
                        help="Directory to save the model")
    parser.add_argument('--model_filename', type=str, default=None,
                        help="Custom filename for the saved model (overrides auto-generated name)")
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 50)
    print("Finetuning Configuration:")
    print("=" * 50)
    
    # check: if val_selection is false but early_stopping is set, set it to None and print a warning
    if not args.val_selection and args.early_stopping is not None:
        print("Warning: early_stopping is set but val_selection is False. Disabling early_stopping.")
        args.early_stopping = None
    
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("=" * 50)
    
    # Run finetuning
    training_info = finetune(args)
    
    print("\nTraining complete!")
    

if __name__ == "__main__":
    main()
