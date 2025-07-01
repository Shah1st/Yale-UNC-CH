
from types import SimpleNamespace
import torch

cfg= SimpleNamespace()
cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.local_rank = 0
cfg.seed = 123
cfg.subsample = None # None=full dataset, int32=first n rows with respect to "dataset"
cfg.subsample_fraction = 0.3  # Use % of data from each dataset family, or None 

cfg.backbone = "convnext_small.fb_in22k_ft_in1k"
cfg.ema = True
cfg.ema_decay = 0.99

cfg.epochs = 5
cfg.batch_size = 32
cfg.batch_size_val = 32

cfg.early_stopping = {"patience": 3, "streak": 0}
cfg.logging_steps = 100

cfg.use_wandb = True
cfg.wandb_project = "seismic-inversion"

# ADDED NEW LINES:
cfg.run_name = "convnext_0.3_[-2]layer"

cfg.mixed_precision = False  # Set to True for A100, False for T4
cfg.precision_dtype = torch.float32  # torch.float16 or torch.bfloat16
cfg.use_grad_scaler = False  # Automatically set based on mixed_precision

cfg.compile_model = True

# GPU Logic: Auto-detect based on environment
cfg.distributed = True  # Will be overridden by auto-detection

# 72x72 vs Full Resolution Mode
cfg.use_72x72_mode = False  # Set to True for 72x72 training, False for full resolution
cfg.data_path_72x72 = "/kaggle/input/openfwi-preprocessed-72x72/openfwi_72x72/"

# 256x72 unusual preprocessing 
cfg.use_smart_256x72_mode = False

cfg.smoothness_loss = False

# Augmentation settings
cfg.use_augmentations = False  # Enable/disable training augmentations
cfg.aug_noise_prob = 0.4      # Probability of noise augmentation
cfg.aug_noise_level = 0.005    # Noise level as fraction of signal std
cfg.aug_scale_prob = 0.3      # Probability of scale augmentation  
cfg.aug_scale_range = (0.99, 1.01)  # Scale range (min, max)

# TTA settings  
cfg.enhanced_tta = False       # Enhanced TTA vs basic flip TTA
cfg.tta_noise_level = 0.005   # TTA noise level as fraction of signal std
cfg.tta_scale_values = [0.98, 1.02]  # Fixed scale values for TTA

# Decoder layer selection
cfg.decoder_layer_index = -3  # -1 = last layer (x[-1]), -2 = second-to-last (x[-2]), etc.
cfg.use_feature_fusion = False  # True = fuse multiple layers, False = single layer

# Feature fusion settings (only used if use_feature_fusion = True)
cfg.fusion_layers = [-1, -2]   # Which layers to combine
cfg.fusion_weights = [0.7, 0.3]  # Weights for each layer (must match fusion_layers length)