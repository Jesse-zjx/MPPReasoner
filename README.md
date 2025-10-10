# MPPReasoner
Reasoning-Enhanced Large Language Models for Molecular Property Prediction

## Create Environment

```bash
conda env create -f swift.yaml
conda env create -f verl.yaml
```

## SFT

```bash
conda activate swift
```

train:

```bash
bash sft/train.sh
```

## RLPGR

```bash
conda activate verl
```

train:

```bash
bash rl/train.sh <sft_model_path>
```

merge:

```bash
bash rl/merge.sh <step>
```

## INFER

infer:

```bash
bash infer/infer.sh <(sft_or_rl_merge)_model_path>
```
