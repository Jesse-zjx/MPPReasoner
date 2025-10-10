import numpy as np
import deepchem as dc
import pandas as pd
import os
from tdc.single_pred import ADME, Tox
from typing import List, Tuple, Dict, Any, Optional, Union

from pathlib import Path
import sys


project_root = Path(__file__).parent.parent
path = project_root / "data" / "deepchem_data"


def load_bioavailability_ma_tdc(splitter=None, reload=True):
    print("Loading Bioavailability_Ma from TDC...")
    data = ADME(name="Bioavailability_Ma", path=path)
    split = data.get_split()

    train_df = split["train"]
    valid_df = split["valid"]
    test_df = split["test"]
    print(train_df)

    train_X = train_df["Drug"].values.astype(str)
    train_y = train_df["Y"].values.astype(int)
    train_ids = train_X

    valid_X = valid_df["Drug"].values.astype(str)
    valid_y = valid_df["Y"].values.astype(int)
    valid_ids = valid_X

    test_X = test_df["Drug"].values.astype(str)
    test_y = test_df["Y"].values.astype(int)
    test_ids = test_X

    tasks = ["Bioavailability_Ma"]

    train_dataset = dc.data.DiskDataset.from_numpy(
        train_X, train_y.reshape(-1, 1), ids=train_ids, tasks=tasks
    )
    valid_dataset = dc.data.DiskDataset.from_numpy(
        valid_X, valid_y.reshape(-1, 1), ids=valid_ids, tasks=tasks
    )
    test_dataset = dc.data.DiskDataset.from_numpy(
        test_X, test_y.reshape(-1, 1), ids=test_ids, tasks=tasks
    )

    transformers = []
    print(train_dataset)
    print("Bioavailability_Ma loaded and converted to DiskDataset.")
    return tasks, (train_dataset, valid_dataset, test_dataset), transformers


def load_cyp2c9_substrate_carbonmangels_tdc(splitter=None, reload=True):
    print("Loading CYP2C9_Substrate_CarbonMangels from TDC...")
    data = ADME(name="CYP2C9_Substrate_CarbonMangels", path=path)
    split = data.get_split()

    train_df = split["train"]
    valid_df = split["valid"]
    test_df = split["test"]

    train_X = train_df["Drug"].values.astype(str)
    train_y = train_df["Y"].values.astype(int)
    train_ids = train_X

    valid_X = valid_df["Drug"].values.astype(str)
    valid_y = valid_df["Y"].values.astype(int)
    valid_ids = valid_X

    test_X = test_df["Drug"].values.astype(str)
    test_y = test_df["Y"].values.astype(int)
    test_ids = test_X

    tasks = ["CYP2C9_Substrate_CarbonMangels"]

    train_dataset = dc.data.DiskDataset.from_numpy(
        train_X, train_y.reshape(-1, 1), ids=train_ids, tasks=tasks
    )
    valid_dataset = dc.data.DiskDataset.from_numpy(
        valid_X, valid_y.reshape(-1, 1), ids=valid_ids, tasks=tasks
    )
    test_dataset = dc.data.DiskDataset.from_numpy(
        test_X, test_y.reshape(-1, 1), ids=test_ids, tasks=tasks
    )

    transformers = []
    print("CYP2C9_Substrate_CarbonMangels loaded and converted to DiskDataset.")
    return tasks, (train_dataset, valid_dataset, test_dataset), transformers


def load_cyp2d6_substrate_carbonmangels_tdc(splitter=None, reload=True):
    print("Loading CYP2D6_Substrate_CarbonMangels from TDC...")
    data = ADME(name="CYP2D6_Substrate_CarbonMangels", path=path)
    split = data.get_split()

    train_df = split["train"]
    valid_df = split["valid"]
    test_df = split["test"]

    train_X = train_df["Drug"].values.astype(str)
    train_y = train_df["Y"].values.astype(int)
    train_ids = train_X

    valid_X = valid_df["Drug"].values.astype(str)
    valid_y = valid_df["Y"].values.astype(int)
    valid_ids = valid_X

    test_X = test_df["Drug"].values.astype(str)
    test_y = test_df["Y"].values.astype(int)
    test_ids = test_X

    tasks = ["CYP2D6_Substrate_CarbonMangels"]

    train_dataset = dc.data.DiskDataset.from_numpy(
        train_X, train_y.reshape(-1, 1), ids=train_ids, tasks=tasks
    )
    valid_dataset = dc.data.DiskDataset.from_numpy(
        valid_X, valid_y.reshape(-1, 1), ids=valid_ids, tasks=tasks
    )
    test_dataset = dc.data.DiskDataset.from_numpy(
        test_X, test_y.reshape(-1, 1), ids=test_ids, tasks=tasks
    )

    transformers = []
    print("CYP2D6_Substrate_CarbonMangels loaded and converted to DiskDataset.")
    return tasks, (train_dataset, valid_dataset, test_dataset), transformers


def load_cyp2c9_veith_tdc(splitter=None, reload=True):
    print("Loading CYP2C9_Veith from TDC...")
    data = ADME(name="CYP2C9_Veith", path=path)
    split = data.get_split()

    train_df = split["train"]
    valid_df = split["valid"]
    test_df = split["test"]

    train_X = train_df["Drug"].values.astype(str)
    train_y = train_df["Y"].values.astype(int)
    train_ids = train_X

    valid_X = valid_df["Drug"].values.astype(str)
    valid_y = valid_df["Y"].values.astype(int)
    valid_ids = valid_X

    test_X = test_df["Drug"].values.astype(str)
    test_y = test_df["Y"].values.astype(int)
    test_ids = test_X

    tasks = ["CYP2C9_Veith"]

    train_dataset = dc.data.DiskDataset.from_numpy(
        train_X, train_y.reshape(-1, 1), ids=train_ids, tasks=tasks
    )
    valid_dataset = dc.data.DiskDataset.from_numpy(
        valid_X, valid_y.reshape(-1, 1), ids=valid_ids, tasks=tasks
    )
    test_dataset = dc.data.DiskDataset.from_numpy(
        test_X, test_y.reshape(-1, 1), ids=test_ids, tasks=tasks
    )

    transformers = []
    print("CYP2C9_Veith loaded and converted to DiskDataset.")
    return tasks, (train_dataset, valid_dataset, test_dataset), transformers


def load_cyp2d6_veith_tdc(splitter=None, reload=True):
    print("Loading CYP2D6_Veith from TDC...")
    data = ADME(name="CYP2D6_Veith", path=path)
    split = data.get_split()

    train_df = split["train"]
    valid_df = split["valid"]
    test_df = split["test"]

    train_X = train_df["Drug"].values.astype(str)
    train_y = train_df["Y"].values.astype(int)
    train_ids = train_X

    valid_X = valid_df["Drug"].values.astype(str)
    valid_y = valid_df["Y"].values.astype(int)
    valid_ids = valid_X

    test_X = test_df["Drug"].values.astype(str)
    test_y = test_df["Y"].values.astype(int)
    test_ids = test_X

    tasks = ["CYP2D6_Veith"]

    train_dataset = dc.data.DiskDataset.from_numpy(
        train_X, train_y.reshape(-1, 1), ids=train_ids, tasks=tasks
    )
    valid_dataset = dc.data.DiskDataset.from_numpy(
        valid_X, valid_y.reshape(-1, 1), ids=valid_ids, tasks=tasks
    )
    test_dataset = dc.data.DiskDataset.from_numpy(
        test_X, test_y.reshape(-1, 1), ids=test_ids, tasks=tasks
    )

    transformers = []
    print("CYP2D6_Veith loaded and converted to DiskDataset.")
    return tasks, (train_dataset, valid_dataset, test_dataset), transformers


def load_ames_tdc(splitter=None, reload=True):
    print("Loading AMES from TDC...")
    data = Tox(name="AMES", path=path)
    split = data.get_split()

    train_df = split["train"]
    valid_df = split["valid"]
    test_df = split["test"]

    train_X = train_df["Drug"].values.astype(str)
    train_y = train_df["Y"].values.astype(int)
    train_ids = train_X

    valid_X = valid_df["Drug"].values.astype(str)
    valid_y = valid_df["Y"].values.astype(int)
    valid_ids = valid_X

    test_X = test_df["Drug"].values.astype(str)
    test_y = test_df["Y"].values.astype(int)
    test_ids = test_X

    tasks = ["AMES"]

    train_dataset = dc.data.DiskDataset.from_numpy(
        train_X, train_y.reshape(-1, 1), ids=train_ids, tasks=tasks
    )
    valid_dataset = dc.data.DiskDataset.from_numpy(
        valid_X, valid_y.reshape(-1, 1), ids=valid_ids, tasks=tasks
    )
    test_dataset = dc.data.DiskDataset.from_numpy(
        test_X, test_y.reshape(-1, 1), ids=test_ids, tasks=tasks
    )

    transformers = []
    print("AMES loaded and converted to DiskDataset.")
    return tasks, (train_dataset, valid_dataset, test_dataset), transformers


if __name__ == "__main__":
    print("Testing custom TDC dataset loaders...")

    try:
        tasks_bm, datasets_bm, transformers_bm = load_bioavailability_ma_tdc()
        print(f"Bioavailability_Ma tasks: {tasks_bm}")
        print(f"Bioavailability_Ma train dataset size: {len(datasets_bm[0])}")
        print(f"Bioavailability_Ma valid dataset size: {len(datasets_bm[1])}")
        print(f"Bioavailability_Ma test dataset size: {len(datasets_bm[2])}\n")
    except Exception as e:
        print(f"Error loading Bioavailability_Ma: {e}\n")

    # try:
    #     tasks_cyp2c9s, datasets_cyp2c9s, transformers_cyp2c9s = load_cyp2c9_substrate_carbonmangels_tdc()
    #     print(f"CYP2C9_Substrate_CarbonMangels tasks: {tasks_cyp2c9s}")
    #     print(f"CYP2C9_Substrate_CarbonMangels train dataset size: {len(datasets_cyp2c9s[0])}")
    #     print(f"CYP2C9_Substrate_CarbonMangels valid dataset size: {len(datasets_cyp2c9s[1])}")
    #     print(f"CYP2C9_Substrate_CarbonMangels test dataset size: {len(datasets_cyp2c9s[2])}\n")
    # except Exception as e:
    #     print(f"Error loading CYP2C9_Substrate_CarbonMangels: {e}\n")

    # try:
    #     tasks_cyp2d6s, datasets_cyp2d6s, transformers_cyp2d6s = load_cyp2d6_substrate_carbonmangels_tdc()
    #     print(f"CYP2D6_Substrate_CarbonMangels tasks: {tasks_cyp2d6s}")
    #     print(f"CYP2D6_Substrate_CarbonMangels train dataset size: {len(datasets_cyp2d6s[0])}")
    #     print(f"CYP2D6_Substrate_CarbonMangels valid dataset size: {len(datasets_cyp2d6s[1])}")
    #     print(f"CYP2D6_Substrate_CarbonMangels test dataset size: {len(datasets_cyp2d6s[2])}\n")
    # except Exception as e:
    #     print(f"Error loading CYP2D6_Substrate_CarbonMangels: {e}\n")

    # try:
    #     tasks_cyp2c9v, datasets_cyp2c9v, transformers_cyp2c9v = load_cyp2c9_veith_tdc()
    #     print(f"CYP2C9_Veith tasks: {tasks_cyp2c9v}")
    #     print(f"CYP2C9_Veith train dataset size: {len(datasets_cyp2c9v[0])}")
    #     print(f"CYP2C9_Veith valid dataset size: {len(datasets_cyp2c9v[1])}")
    #     print(f"CYP2C9_Veith test dataset size: {len(datasets_cyp2c9v[2])}\n")
    # except Exception as e:
    #     print(f"Error loading CYP2C9_Veith: {e}\n")

    # try:
    #     tasks_cyp2d6v, datasets_cyp2d6v, transformers_cyp2d6v = load_cyp2d6_veith_tdc()
    #     print(f"CYP2D6_Veith tasks: {tasks_cyp2d6v}")
    #     print(f"CYP2D6_Veith train dataset size: {len(datasets_cyp2d6v[0])}")
    #     print(f"CYP2D6_Veith valid dataset size: {len(datasets_cyp2d6v[1])}")
    #     print(f"CYP2D6_Veith test dataset size: {len(datasets_cyp2d6v[2])}\n")
    # except Exception as e:
    #     print(f"Error loading CYP2D6_Veith: {e}\n")

    # try:
    #     tasks_ames, datasets_ames, transformers_ames = load_ames_tdc()
    #     print(f"AMES tasks: {tasks_ames}")
    #     print(f"AMES train dataset size: {len(datasets_ames[0])}")
    #     print(f"AMES valid dataset size: {len(datasets_ames[1])}")
    #     print(f"AMES test dataset size: {len(datasets_ames[2])}\n")
    # except Exception as e:
    #     print(f"Error loading AMES: {e}\n")

    print("Custom TDC dataset loaders testing complete.")
