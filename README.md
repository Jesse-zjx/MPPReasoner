# MPPReasoner
Reasoning-Enhanced Large Language Models for Molecular Property Prediction

## Overview

MPPReasoner is a multimodal large language model that systematically incorporates chemical reasoning for molecular property prediction. Built upon Qwen2.5-VL-7B-Instruct, it integrates molecular images with SMILES strings and employs a novel two-stage training framework with Reinforcement Learning from Principle-Guided Rewards (RLPGR).

## Hardware Requirements

- **Minimum**: 4 × NVIDIA A100 GPUs (recommended: 8 × A100 80GB)
- **Memory**: Each GPU should have sufficient VRAM (≥40GB recommended)
- **Storage**: At least 100GB free space for models and datasets

## Environment Setup

Create the required conda environments:

```bash
# For SFT training
conda env create -f swift.yml

# For RL training  
conda env create -f verl.yml
```

## Data and Model Preparation

### Download Datasets and Checkpoints
```bash
# Download datasets and pre-trained checkpoints from anonymous repository
# [Dataset and checkpoint links will be provided upon paper acceptance]
# Extract to ./data/ and ./ckpts/ directories respectively
```

### Directory Structure
```
MPPReasoner/
├── data/
│   ├── train/          # Training data (bace.jsonl, bbbp.jsonl, etc.)
│   ├── test/           # Test data (parquet files)
│   └── deepchem_data/  # DeepChem datasets cache
├── ckpts/              # Model checkpoints
├── sft/                # Supervised fine-tuning scripts
├── rl/                 # Reinforcement learning scripts
└── infer/              # Inference scripts
```

## Training Pipeline

### Stage 1: Supervised Fine-Tuning (SFT)

```bash
conda activate swift

# Train with default settings (uses GPUs 0-7)
bash sft/train.sh

# Or specify custom GPUs
bash sft/train.sh 0,1,2,3
```

**SFT Configuration:**
- **Input**: 16,000 curated reasoning trajectories from ID datasets
- **Epochs**: 3
- **Batch Size**: 2 per device (effective batch size: 16 with gradient accumulation)
- **Learning Rate**: 1e-5
- **Training Time**: ~2 hours on 8 A100 GPUs
- **Output**: `ckpts/sft/`

### Stage 2: Reinforcement Learning with RLPGR

```bash
conda activate verl

# Train RL model using SFT checkpoint
bash rl/train.sh ckpts/sft/

# Merge model after training (specify step number)
bash rl/merge.sh 300
```

**RL Configuration:**
- **Algorithm**: Group Relative Policy Optimization (GRPO)
- **Reward**: Hierarchical RLPGR (Foundation + Reasoning + Chemistry layers)
- **Steps**: 300 optimization steps
- **Learning Rate**: 1e-6
- **Training Time**: ~12 hours on 8 A100 GPUs
- **Output**: `ckpts/rl/merged_{step}/`

## Inference and Evaluation

### Run Inference

```bash
# Using SFT model
bash infer/infer.sh ckpts/sft/

# Using RL merged model
bash infer/infer.sh ckpts/rl/merged_300/
```

**Inference Configuration:**
- **Datasets**: Evaluates on all 8 datasets (4 ID + 4 OOD)
- **Batch Size**: 256
- **Temperature**: 0.0 (deterministic)
- **Output**: Results saved to `./results/` with ROC-AUC metrics

### Custom Inference

```bash
python ./infer/infer_new.py \
    --model Qwen2.5-VL-7B-Instruct \
    --model_path "path/to/your/model" \
    --data_dir "./data/test" \
    --dataset bace,bbbp \
    --gpus 0,1,2,3 \
    --batch_size 256 \
    --output_dir "./custom_results"
```

## Key Features

### RLPGR Reward Framework
- **Foundation Layer**: Answer correctness + Format compliance
- **Reasoning Layer**: Logical consistency + Comparative analysis  
- **Chemistry Layer**: Chemical principle application + Molecular structure analysis

### Multimodal Input Processing
- **SMILES strings**: Sequential chemical information
- **2D molecular images**: Spatial structural relationships
- **Few-shot examples**: Retrieved by Tanimoto similarity

### Evaluation Capabilities
- **8 datasets**: 4 ID (BACE, BBBP, SIDER, HIV) + 4 OOD (Bioavailability, CYP2C9_V, CYP2D6_V, AMES)
- **Automated metrics**: ROC-AUC calculation with multi-task support

## Output Format

### Inference Results
```json
{
  "original_index": 0,
  "input_smiles": "CCN(CC)C(=O)...",
  "true_label": 1.0,
  "parsed_predicted_score": 0.85,
  "raw_predicted_output": "<think>...</think><answer>True</answer>",
  "probabilities": {"true_ratio": 0.85, "token_probs": {...}}
}
```

### Metrics Summary
```json
{
  "model": "MPPReasoner",
  "datasets": {
    "bace": {"roc_auc": 0.9090},
    "bbbp": {"roc_auc": 0.7436}
  }
}
```

## Reproduction Guide

To reproduce the paper results:

1. **Environment Setup**: Create both conda environments
2. **Data Preparation**: Download and place datasets in correct directories
3. **SFT Training**: Train for 3 epochs (~2 hours)
4. **RL Training**: Train with RLPGR for 300 steps (~12 hours)  
5. **Evaluation**: Run inference on all 8 datasets
6. **Analysis**: Compare results with Table 1 in the paper
