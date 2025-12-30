# Alzheimer-MRI-Classifier-CNN-Transfer-Learning

## Project Description

This project implements a comprehensive five-stage deep learning framework for automated multi-class Alzheimer's disease detection and severity classification from magnetic resonance imaging data. The research systematically compares custom convolutional neural network architectures against pre-trained transfer learning models, incorporating Bayesian hyperparameter optimization and class imbalance mitigation strategies to achieve state-of-the-art classification performance on the OASIS dataset comprising 86,437 brain MRI scans across four diagnostic categories.

## Dataset Information

**Source:** OASIS Alzheimer's Detection dataset from Kaggle  
**URL:** https://www.kaggle.com/datasets/ninadaithal/imagesoasis  
**Size:** 86,437 MRI images  
**Categories:** Non-Demented (67,222), Very Mild Dementia (13,725), Mild Dementia (5,002), Moderate Dementia (488)  
**Format:** 496×248 pixel RGB JPEG images  
**License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International

### Dataset Download Instructions

1. Create a Kaggle account and accept the dataset terms
2. Download the complete OASIS dataset from the URL above
3. Extract the downloaded archive to your working directory
4. Verify the dataset structure contains four subdirectories corresponding to the diagnostic categories

## Installation Requirements

**Python Version:** 3.10 or higher

**Required Dependencies:**
```
tensorflow==2.19.0
keras==2.19.0
scikit-learn==1.3.0
numpy==1.24.0
pandas==2.0.0
matplotlib==3.7.0
seaborn==0.12.0
keras-tuner==1.3.5
pillow==10.0.0
```

**Hardware Requirements:**

- Minimum 16GB RAM recommended
- GPU with 8GB+ VRAM strongly recommended for transfer learning experiments
- Approximately 20GB free disk space for dataset and model checkpoints

## Code Structure

The project notebook `FPR-Code-CNN-Transfer-Learning-Hyperparameter-tuning-finetuning.ipynb` is organized into sequential sections corresponding to the five-stage experimental framework:

### Section 1: Data Loading and Preprocessing

- Dataset path configuration
- Image loading and validation
- Exploratory data analysis and visualization
- Class distribution analysis

### Section 2: Data Partitioning and Augmentation

- Stratified 60-20-20 train-validation-test splitting
- Dual-resolution preprocessing pipelines (128×128 for custom CNNs, 224×224 for transfer learning)
- Data augmentation configuration with rotation, flipping, shifting, brightness adjustment, and zoom
- Class weight computation for imbalance mitigation

### Section 3: Baseline Custom CNN

- Simple CNN architecture implementation (2 convolutional blocks, 2.1M parameters)
- Model compilation and training configuration
- Training execution with class weights
- Performance evaluation and confusion matrix generation

### Section 4: Bayesian Hyperparameter Optimization

- Keras Tuner BayesianOptimization setup
- Hyperparameter search space definition (layers, filters, kernels, dropout, learning rate)
- Five-trial optimization execution
- Optimal configuration selection and documentation

### Section 5: Optimized Custom CNN

- Implementation of Bayesian-optimized architecture (1.05M parameters)
- Larger 5×5 kernels with strategic dropout placement
- Model training and evaluation
- Performance comparison with baseline

### Section 6: Transfer Learning - Frozen Mode

- ResNet50, MobileNetV2, DenseNet121, and EfficientNetB0 implementation
- Pre-trained ImageNet weight loading
- Custom classification head construction
- Frozen base model training and evaluation

### Section 7: Transfer Learning - Fine-Tuned Mode

- Two-phase training procedure implementation
- Top 30 layers unfreezing for domain adaptation
- Reduced learning rate and early stopping configuration
- Comprehensive performance evaluation across all architectures

### Section 8: Results Compilation and Visualization

- Consolidated performance metrics across all eleven models
- Confusion matrix generation and analysis
- Training history visualization
- Comparative performance plotting

## How to Run the Code

### Complete Pipeline Execution

1. Open the Jupyter notebook: `jupyter notebook FPR-Code-CNN-Transfer-Learning-Hyperparameter-tuning-finetuning.ipynb`
2. Ensure the dataset path in Section 1 points to your OASIS dataset location
3. Execute cells sequentially from top to bottom
4. Monitor training progress and GPU utilization
5. Review generated visualizations and performance metrics

### Individual Section Execution

- Each section can be executed independently if required model dependencies from previous sections are satisfied
- Baseline CNN training (Section 3) requires only data preprocessing completion
- Transfer learning sections (6-7) require dataset preparation and class weight computation
- Results compilation (Section 8) requires all model training sections completed

## Model Training Instructions

### Baseline and Optimized Custom CNNs

- Batch size: 32
- Epochs: 30
- Optimizer: Adam with learning rate 0.00001 (baseline) or 0.0001 (optimized)
- Loss function: Categorical cross-entropy with computed class weights
- Training automatically saves best model based on validation accuracy

### Transfer Learning - Frozen Mode

- Batch size: 32
- Epochs: 20
- Optimizer: Adam with learning rate 0.0001
- Only classification head trainable (0.6-3.5% of total parameters)

### Transfer Learning - Fine-Tuned Mode

- Phase 1: 10 epochs frozen base, classification head training
- Phase 2: 20 additional epochs with top 30 layers unfrozen
- Reduced learning rate: 0.00001
- Early stopping patience: 7 epochs on validation loss
- ReduceLROnPlateau: patience 3 epochs, factor 0.5
- Class weights applied during fine-tuning
- Best model saved based on validation performance

## Expected Outputs

### Performance Metrics

- Test accuracy, precision, recall, F1-score, and Cohen's kappa for all eleven models
- Per-class performance breakdowns using macro-averaging
- Confusion matrices visualizing classification patterns across diagnostic categories

### Visualizations

- Class distribution plots across train-validation-test partitions
- Data augmentation demonstration images
- Training and validation accuracy/loss curves for all models
- Confusion matrices for all eleven model configurations
- Comparative performance bar charts
- Class weight visualization

### Key Findings

- Baseline CNN: 92.81% accuracy
- Tuned CNN: 99.73% accuracy with 50% parameter reduction
- Best overall performance: ResNet50 fine-tuned at 99.94% accuracy
- All fine-tuned transfer learning models exceed 99.7% accuracy
- Successful class imbalance mitigation with recall >0.99 across all categories

## Reproducibility Notes

All experiments employ fixed random seed (42) across NumPy, TensorFlow, and Python random number generators to ensure reproducibility. Results may vary slightly across different hardware configurations due to GPU floating-point precision differences, though overall performance trends remain consistent. Training on CPU-only systems will produce identical results but require significantly longer execution time.

## Citation and Attribution

Marcus, D.S., Wang, T.H., Parker, J., Csernansky, J.G., Morris, J.C. and Buckner, R.L. (2007) 'Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI Data in Young, Middle Aged, Nondemented, and Demented Older Adults', Journal of Cognitive Neuroscience, 19(9), pp. 1498-1507.

Dataset available via Kaggle: Aithal, N. (2021) OASIS Alzheimer's Detection. https://www.kaggle.com/datasets/ninadaithal/imagesoasis
