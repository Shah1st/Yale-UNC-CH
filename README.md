# ConvNeXt Seismic Inversion Enhancement

This repository contains my enhanced version of the ConvNeXt baseline for seismic inversion, implementing multiple training modes, advanced augmentations, and architectural improvements for the seismic velocity model prediction task.

**Note**: The competition dataset contains 400+GB of data including additional datasets. Due to computational resource limitations, I experimented with resizing approaches and trained most experiments on a 30% subset of the data.

## Original Architecture

The original ConvNeXt baseline uses a UNet architecture with a ConvNeXt encoder and standard decoder. The encoder backbone is pretrained on ImageNet and modified with custom stem layers to handle the rectangular seismic input shape through aggressive height downsampling. The decoder uses standard upsampling with skip connections. Training uses distributed setup with EMA, basic L1 loss, and simple flip augmentation. Test-time augmentation only applies flip transformation.


## Original Architecture

The original ConvNeXt baseline uses a UNet architecture with a ConvNeXt encoder and standard decoder. The encoder backbone is pretrained on ImageNet and modified with custom stem layers to handle the rectangular seismic input shape through aggressive height downsampling. The decoder uses standard upsampling with skip connections. Training uses distributed setup with EMA, basic L1 loss, and simple flip augmentation. Test-time augmentation only applies flip transformation.

## My Essential Changes

### Multiple Training Modes
Added three operational modes - 72x72 for preprocessed small inputs, smart 256x72 with engineered preprocessing pipeline, and enhanced full resolution mode.

### Smart Preprocessing
For 256x72 mode, implemented preprocessing that focuses on early seismic arrivals, crops to 512 time samples, downsamples to 256x70, and pads to 72x72. Created a PerfectSegmentationHead that produces exact 70x70 outputs without cropping.

### Enhanced Decoder
Added configurable decoder layer selection instead of always using the final layer, and feature fusion capability to combine multiple decoder layers with weighted averaging.

### Advanced Augmentations
Implemented configurable noise augmentation and scale augmentation with probability controls and safety checks to prevent NaN values.

### Training Infrastructure
Added mixed precision support with configurable precision types, Weights and Biases integration for experiment tracking, model compilation for faster training, and smoothness loss regularization.

### Enhanced TTA
Expanded from simple flip TTA to include noise TTA variants, scale TTA, and flip TTA with weighted averaging of 5+ augmented predictions.

### Improved Data Loading
Added percentage-based subsampling within dataset families and intelligent GPU detection for automatic single/multi-GPU configuration.

### Learning Rate Scheduling
Added CosineLRScheduler for better training dynamics.

## Experimental Results

### What Worked
- **Configurable decoder**: Successfully improved performance by using different decoder layers
- **Smoothness loss**: Little positive changes in model quality
- **CosineLRScheduler**: Little positive changes with more stability during training

### What Didn't Work 
- **72x72 mode**: Failed to produce good results - likely due to too aggressive downsampling losing critical seismic details and temporal resolution needed for accurate velocity reconstruction
- **256x72 mode**: Failed to produce good results - the preprocessing pipeline may have removed important information or the aspect ratio mismatch caused architectural issues with the UNet skip connections
- **Augmentations**: Caused NaN issues on Kaggle T4 GPUs (possibly due to mixed precision instability with T4's limited numerical precision), slightly better results on Colab A100 where better numerical stability allowed augmentations to work
- **Enhanced TTA**: Didn't really test thoroughly - insufficient experimentation to determine effectiveness

## Post-Processing Experiment Summary
We tested multiple post-processing techniques to improve the baseline 25.6 MAE score. I tried to make bronze in a last few hours))

### Approaches Tested
- **Ensemble simulation** - Multi-model averaging
- **Adaptive fusion** - Attention-based spatial processing
- **Statistical correction** - Bias and outlier adjustment
- **Frequency filtering** - Noise reduction in frequency domain
- **Spatial smoothing** - Gaussian regularization

### Results
All post-processing attempts **failed to improve** performance:
- Ensemble simulation: 25.8 MAE (+0.2)
- Adaptive fusion: 25.8 MAE (+0.2)
- Statistical correction: 29.7 MAE (+4.1)
- Frequency filtering: 35.6 MAE (+10.0)
- Spatial smoothing: 41.0 MAE (+15.4)

## Resources

- **Experiment Tracking**: [W&B Report](https://api.wandb.ai/links/shah1st-work-ua-igor-sikorsky-kyiv-polytechnic-institute/yhxiuioi)
- **Model Weights & Submissions**: [Kaggle Dataset](https://www.kaggle.com/datasets/oleksandrkharytonov/yaleunc-ch-models/data)

## Repository Structure

```
├── convnext_improved_baseline.ipynb          # Main enhanced training script (Jupyter notebook)
├── convnext_full_resolution_baseline.ipynb  # Original baseline for comparison
├── _cfg.py                       # Configuration settings
├── _dataset.py                   # Data loading and preprocessing
├── _model.py                     # Model architecture and enhancements
├── _train.py                     # Training loop and utilities
└── _utils.py                     # Helper functions
```

## Key Findings

These changes transform the basic baseline into a comprehensive training system, though only some modifications proved beneficial in practice. The resolution-based approaches suggest that seismic data requires sufficient spatial and temporal detail for effective velocity inversion. The most successful improvements were architectural (configurable decoder) and training dynamics (smoothness loss, learning rate scheduling) rather than data preprocessing modifications.


## Usage

The enhanced system maintains backward compatibility with the original approach through the configuration system, allowing easy switching between different training modes and experimental settings.

---

**P.S.** I plan to run the successful methods (configurable decoder, smoothness loss, augmentations) on the full resolution dataset with all available data once Kaggle GPU limits reset, or will ask friends to help with computational resources for the complete evaluation.
