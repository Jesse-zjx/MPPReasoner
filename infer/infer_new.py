import json, os
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
from typing import List, Dict, Any, Tuple, Optional, Union
import asyncio
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Queue, Event, Process
from tqdm import tqdm
import traceback
import base64
from PIL import Image
import io
from transformers import AutoTokenizer
import re

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from vllm_logit_client import VLLMLogitClient
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score

import deepchem as dc

deepchem_data_path = project_root / "data" / "deepchem_data"
os.environ["DEEPCHEM_DATA_DIR"] = deepchem_data_path

from tdc_dataset_loaders import (
    load_bioavailability_ma_tdc,
    load_cyp2c9_substrate_carbonmangels_tdc,
    load_cyp2d6_substrate_carbonmangels_tdc,
    load_cyp2c9_veith_tdc,
    load_cyp2d6_veith_tdc,
    load_ames_tdc,
)

DATASET_LOADERS = {
    "bace": dc.molnet.load_bace_classification,
    "bbbp": dc.molnet.load_bbbp,
    "sider": dc.molnet.load_sider,
    "hiv": dc.molnet.load_hiv,
    "tox21": dc.molnet.load_tox21,
    "toxcast": dc.molnet.load_toxcast,
    "muv": dc.molnet.load_muv,
    "bioavailability_ma": load_bioavailability_ma_tdc,
    "cyp2c9_substrate_carbonmangels": load_cyp2c9_substrate_carbonmangels_tdc,
    "cyp2d6_substrate_carbonmangels": load_cyp2d6_substrate_carbonmangels_tdc,
    "cyp2c9_veith": load_cyp2c9_veith_tdc,
    "cyp2d6_veith": load_cyp2d6_veith_tdc,
    "ames": load_ames_tdc,
}


def _is_base64_util(s: str) -> bool:
    if s.startswith("data:"):
        parts = s.split(",", 1)
        if len(parts) == 2:
            s = parts[1]
        else:
            return False
    try:
        return base64.b64decode(s, validate=True) is not None
    except (base64.binascii.Error, TypeError):
        return False


def calculate_roc_auc(true_labels, predicted_scores):
    if len(np.unique(true_labels)) < 2:
        print("ROC-AUC cannot be calculated when true labels have only one class.")
        return None
    try:
        return roc_auc_score(true_labels, predicted_scores)
    except ValueError as e:
        print(f"Error calculating ROC-AUC: {e}")
        return None


def _apply_chat_template_for_preprocessor(
    tokenizer,
    messages: List[Dict[str, Any]],
    images_data: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, Optional[Dict[str, List[Any]]]]:
    """
    Applies chat template logic, including image processing, during preprocessing.
    It prepares the prompt text and loads PIL image objects.
    """
    processed_messages = []
    last_user_message_idx = -1
    for i, msg in enumerate(messages):
        if msg["role"] == "user":
            last_user_message_idx = i
        processed_messages.append(msg.copy())

    if images_data is not None and len(images_data) > 0 and last_user_message_idx != -1:
        # replaced = "<|image_pad|>"
        replaced = "<|vision_start|><|image_pad|><|vision_end|>"
    else:
        replaced = ""
    last_user_msg = processed_messages[last_user_message_idx]
    current_user_content = last_user_msg["content"]

    if isinstance(current_user_content, str):
        text_content = current_user_content.replace("<image>", replaced).strip()
        if len(text_content) > 0:
            processed_messages[last_user_message_idx]["content"] = text_content
    elif isinstance(current_user_content, list):
        text_parts = []
        for part in current_user_content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part["text"].replace("<image>", replaced)
                text_parts.append(text)
        if len(text_parts) > 0:
            processed_messages[last_user_message_idx]["content"] = " ".join(text_parts)
        else:
            processed_messages[last_user_message_idx]["content"] = ""

    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        raise RuntimeError(
            "Tokenizer does not have chat template. Check if the model supports multimodal conversation templates."
        )

    try:
        prompt_text = tokenizer.apply_chat_template(
            processed_messages, tokenize=False, add_generation_prompt=True
        )

        multi_modal_data = None
        if (
            images_data is not None
            and len(images_data) > 0
            and last_user_message_idx != -1
        ):
            image_pil_list = []
            for img_dict in images_data:
                if isinstance(img_dict, dict) and "image" in img_dict:
                    image_b64 = img_dict["image"]
                    if isinstance(image_b64, str) and _is_base64_util(image_b64):
                        try:
                            if "," in image_b64:
                                image_b64 = image_b64.split(",", 1)[1]
                            image_bytes = base64.b64decode(image_b64)
                            image_pil = Image.open(io.BytesIO(image_bytes))
                            image_pil_list.append(image_pil)
                        except Exception as e:
                            print(
                                f"[Preprocessor Error] Failed to decode or load Base64 image: {e}. Skipping this image."
                            )
                    else:
                        print(
                            "[Preprocessor Warn] Image data is not a valid Base64 string, skipping this image."
                        )
            if len(image_pil_list) > 0:
                multi_modal_data = {"image": image_pil_list}

        return prompt_text, multi_modal_data
    except Exception as e:
        print(f"[Preprocessor Error] Failed to apply chat template: {e}")
        raise


class DataPreprocessor:
    def __init__(
        self, df_data: pd.DataFrame, batch_size: int, use_images: bool, model_path: str
    ):
        self.df_data = df_data
        self.batch_size = batch_size
        self.use_images = use_images
        self.model_path = model_path
        self.num_samples = len(df_data)
        self.queue = Queue(maxsize=50)
        self.stop_event = Event()
        self.tokenizer = None
        # print(f"[Preprocessor] Initialized data preprocessor, total samples: {self.num_samples}, batch size: {self.batch_size}")

    def preprocess_batch(self, start_idx: int) -> Tuple[
        List[Union[str, Dict]],
        Optional[List[Optional[Dict[str, List[Any]]]]],
        List[int],
    ]:
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            # print(f"[Preprocessor Worker] Tokenizer loaded in worker process for {self.model_path}")

        end_idx = min(start_idx + self.batch_size, self.num_samples)
        batch_df = self.df_data.iloc[start_idx:end_idx]

        preprocessed_prompts = []
        preprocessed_multi_modal_data = []
        original_indices_in_batch = []

        for i, (_, row) in enumerate(batch_df.iterrows()):
            current_original_idx = start_idx + i
            try:
                prompt_content = row["prompt"]
                images_data = (
                    row["images"] if self.use_images and "images" in row else None
                )

                processed_prompt_text, processed_mmd = (
                    _apply_chat_template_for_preprocessor(
                        self.tokenizer, prompt_content, images_data
                    )
                )
                preprocessed_prompts.append(processed_prompt_text)
                preprocessed_multi_modal_data.append(processed_mmd)
                original_indices_in_batch.append(current_original_idx)
            except Exception as e:
                print(
                    f"[Preprocessor Error] Error preprocessing sample {current_original_idx}: {e}"
                )
                traceback.print_exc()
                preprocessed_prompts.append(None)
                preprocessed_multi_modal_data.append(None)
                original_indices_in_batch.append(current_original_idx)
                continue

        print(
            f"[Preprocessor] Batch starting index: {start_idx} preprocessed {len(preprocessed_prompts)} samples."
        )
        return (
            preprocessed_prompts,
            preprocessed_multi_modal_data,
            original_indices_in_batch,
        )

    def preload_worker_process(
        self_obj,
        num_samples,
        batch_size,
        df_data,
        use_images,
        model_path,
        queue,
        stop_event,
    ):
        worker_preprocessor = DataPreprocessor(
            df_data, batch_size, use_images, model_path
        )
        worker_preprocessor.queue = queue
        worker_preprocessor.stop_event = stop_event

        idx = 0
        while (
            not worker_preprocessor.stop_event.is_set()
            and idx < worker_preprocessor.num_samples
        ):
            try:
                batch_prompts, batch_multi_modal_data, original_indices_in_batch = (
                    worker_preprocessor.preprocess_batch(idx)
                )
                worker_preprocessor.queue.put(
                    (
                        idx,
                        batch_prompts,
                        batch_multi_modal_data,
                        original_indices_in_batch,
                    )
                )
                # print(f"[Preprocessor Worker] Successfully preloaded batch starting index: {idx}")
                idx += worker_preprocessor.batch_size
            except Exception as e:
                print(f"[Preprocessor Worker Error] Preload worker process error: {e}")
                traceback.print_exc()
                worker_preprocessor.stop_event.set()
                break
        # print("[Preprocessor Worker] Preload worker process finished.")

    def start(self):
        self.worker_process = Process(
            target=self.preload_worker_process,
            args=(
                self.num_samples,
                self.batch_size,
                self.df_data,
                self.use_images,
                self.model_path,
                self.queue,
                self.stop_event,
            ),
        )
        self.worker_process.daemon = True
        self.worker_process.start()
        # print("[Preprocessor] Preload worker process started.")

    def stop(self):
        self.stop_event.set()
        if self.worker_process.is_alive():
            self.worker_process.join(timeout=30)  # Increased timeout
            if self.worker_process.is_alive():
                print(
                    "[Preprocessor Warn] Preload worker process did not stop in time, terminating."
                )
                self.worker_process.terminate()
                self.worker_process.join()
        # print("[Preprocessor] Preload worker process stopped.")


async def process_batch_async(
    client: VLLMLogitClient,
    preprocessed_prompts: List[Union[str, Dict]],
    preprocessed_multi_modal_data: Optional[List[Optional[Dict[str, List[Any]]]]],
    target_phrase: str,
    max_total_tokens: int,
    temperature: float,
    top_k: int,
) -> List[Tuple[Optional[Dict], str]]:
    loop = asyncio.get_event_loop()

    try:
        with ThreadPoolExecutor() as pool:
            results = await loop.run_in_executor(
                pool,
                lambda: client.batch_get_probabilities(
                    preprocessed_prompts=preprocessed_prompts,
                    preprocessed_multi_modal_data=preprocessed_multi_modal_data,
                    target_phrase=target_phrase,
                    max_total_tokens=max_total_tokens,
                    temperature=temperature,
                    top_k=top_k,
                ),
            )
        return results
    except Exception as e:
        traceback.print_exc()
        return [(None, "") for _ in preprocessed_prompts]


def save_metrics_summary(
    dataset: str, model: str, metrics: Dict[str, float], output_path: str
) -> None:
    """Saves the calculated metrics to a summary JSON file."""
    summary_dir = f"{output_path}/{dataset}/{model}"
    os.makedirs(summary_dir, exist_ok=True)
    summary_file = f"{summary_dir}/metrics_summary.json"

    existing_metrics = {}
    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                existing_metrics = json.load(f)
        except json.JSONDecodeError:
            print(
                f"Warning: Could not decode existing metrics file {summary_file}. Will overwrite."
            )
            existing_metrics = {}

    existing_metrics.update({"dataset": dataset, "model": model, "metrics": metrics})

    try:
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(existing_metrics, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Save Metrics Error] Failed to save metrics file {summary_file}: {e}")
        traceback.print_exc()


def save_model_metrics_summary(
    model: str, dataset_metrics: Dict[str, Dict[str, float]], output_path: str
) -> None:
    """Saves the metrics summary organized by model for all datasets."""
    summary_dir = f"{output_path}/model_summaries"
    os.makedirs(summary_dir, exist_ok=True)
    summary_file = f"{summary_dir}/{model}_metrics_summary.json"

    existing_metrics = {}
    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                existing_metrics = json.load(f)
        except json.JSONDecodeError:
            print(
                f"Warning: Could not decode existing model metrics file {summary_file}. Will overwrite."
            )
            existing_metrics = {}

    existing_metrics.update({"model": model, "datasets": dataset_metrics})

    try:
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(existing_metrics, f, ensure_ascii=False, indent=2)
        print(f"Saved summary metrics for model {model} to {summary_file}")
    except Exception as e:
        print(
            f"[Save Model Metrics Error] Failed to save model metrics file {summary_file}: {e}"
        )
        traceback.print_exc()


def load_and_concatenate_parquet_files(
    data_dir: str, dataset_name: str
) -> Optional[pd.DataFrame]:
    """
    Finds and loads a base parquet file and any split parts, concatenating them into a single DataFrame.
    It looks for files matching `{dataset_name}_test.parquet` and `{dataset_name}_test_1.parquet`, etc.
    """
    search_path = Path(data_dir)
    # This regex matches '_test.parquet' or '_test_1.parquet', '_test_123.parquet', etc.
    file_pattern_re = re.compile(rf"{dataset_name}_test(_\d+)?\.parquet$")

    # First, glob for a wider pattern to find potential candidates
    potential_files = search_path.glob(f"{dataset_name}_test*.parquet")

    # Then, filter with the precise regex and sort the results
    files_to_load = sorted(
        [f for f in potential_files if file_pattern_re.search(os.path.basename(f))]
    )

    if not files_to_load:
        print(
            f"Error: No test data files found matching pattern '{dataset_name}_test(_\\d+)?.parquet' in directory {data_dir}."
        )
        return None

    print(
        f"Found the following files to load: {[os.path.basename(f) for f in files_to_load]}"
    )

    df_list = []
    for file_path in files_to_load:
        try:
            df_part = pd.read_parquet(file_path)
            df_list.append(df_part)
            print(f"Loaded {len(df_part)} records from {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error: Failed to load Parquet file {file_path}: {e}")
            # Continue to try loading other parts even if one fails
            continue

    if not df_list:
        print("Error: Files were found, but none could be loaded.")
        return None

    # Concatenate all loaded dataframes into a single one
    full_df = pd.concat(df_list, ignore_index=True)
    return full_df


async def main():
    parser = argparse.ArgumentParser(
        description="Run inference with specified model and pre-generated data."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., Qwen2.5-VL-7B-Instruct)",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the pre-generated parquet files",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="bace,bbbp,sider",
        help="Dataset name(s) to evaluate (e.g., bace,sider)",
    )
    parser.add_argument(
        "--gpus", type=str, default="0,1,2,3", help="GPU IDs to use (comma-separated)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for inference"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_results_v2",
        help="Output directory for results",
    )
    parser.add_argument(
        "--use_images",
        type=str,
        default="True",
        help="Whether to use image modality (True/False)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature for inference"
    )
    parser.add_argument("--top_k", type=int, default=5, help="Top-k for inference")
    args = parser.parse_args()

    datasets_to_process = args.dataset.split(",")
    use_images = args.use_images.lower() == "true"

    client = VLLMLogitClient(
        model_name=args.model,
        model_path=args.model_path,
        gpus=args.gpus,
        batch_size=args.batch_size,
    )

    all_dataset_metrics = {}

    with client:
        for dataset_name in datasets_to_process:
            print(f"\n>>>> Processing dataset: {dataset_name} <<<<")

            # Load the DeepChem dataset to determine if it's multi-task
            # This is the key change to align with generate_rl_data_deepchem.py
            deepchem_tasks = []
            if dataset_name in DATASET_LOADERS:
                try:
                    # Only load to get the tasks, we don't need the actual datasets here
                    deepchem_tasks, _, _ = DATASET_LOADERS[dataset_name](
                        splitter="scaffold", reload=True
                    )
                except Exception as e:
                    print(
                        f"Error loading DeepChem tasks for {dataset_name}: {e}. Assuming single-task."
                    )
                    deepchem_tasks = [
                        dataset_name
                    ]  # Fallback to a single task if loading fails
            else:
                print(
                    f"Warning: Dataset {dataset_name} not found in DATASET_LOADERS. Assuming single-task."
                )
                deepchem_tasks = [
                    dataset_name
                ]  # Assume single task if not in known loaders
            print(f"deepchem_tasks: {deepchem_tasks}")
            # Determine if it's a multi-task dataset based on the number of deepchem_tasks
            is_multi_task = len(deepchem_tasks) > 1

            # MODIFIED: Load all parquet parts (split or single)
            print(f"Loading data for {dataset_name} from {args.data_dir}...")
            df_test = load_and_concatenate_parquet_files(args.data_dir, dataset_name)

            if df_test is None or df_test.empty:
                print(
                    f"Data for {dataset_name} not found or could not be loaded from {args.data_dir}. Skipping this dataset."
                )
                continue

            print(f"Successfully loaded a total of {len(df_test)} records.")

            tasks = []
            if is_multi_task:
                # For multi-task datasets, we should use the deepchem_tasks list
                tasks = deepchem_tasks
                print(f"Detected {len(tasks)} tasks: {tasks}")
            # else: for single task, the 'tasks' list will remain empty or contain just the dataset name,
            # which is handled by the metrics calculation logic for single-task datasets.

            metrics = await process_dataset_with_model(
                dataset_name,
                tasks,
                df_test,
                client,
                args.output_dir,
                is_multi_task,
                use_images,
                args.model_path,
                args.temperature,
                args.top_k,
            )

            if metrics:
                all_dataset_metrics[dataset_name] = metrics

    if all_dataset_metrics:
        save_model_metrics_summary(args.model, all_dataset_metrics, args.output_dir)

    print("\nAll datasets processed!")


async def process_dataset_with_model(
    dataset_name: str,
    tasks: List[str],
    df_data: pd.DataFrame,
    client: VLLMLogitClient,
    output_path: str,
    is_multi_task: bool,
    use_images: bool,
    model_path: str,
    temperature: float,
    top_k: int,
) -> Optional[Dict[str, float]]:
    """Processes the given dataset with the VLLM client and calculates metrics."""

    model_name = client.model_name
    experiment_name = f"{model_name}"

    print(
        f"\n--- Starting evaluation for dataset: {dataset_name}, model: {experiment_name} ---"
    )
    print(f"--- Using image modality: {use_images} ---")

    all_true_labels = []
    all_predicted_scores = []
    sider_results_by_task = {
        task: {"true_labels": [], "predicted_scores": []} for task in tasks
    }

    out_dir = f"{output_path}/{dataset_name}/{experiment_name}"
    os.makedirs(out_dir, exist_ok=True)

    # --- MODIFICATION START: Periodic Saving Logic ---
    all_results_to_save = []
    part_number = 0
    SAVE_INTERVAL_BATCHES = 10000 // client.batch_size
    # --- MODIFICATION END ---

    preprocessor = DataPreprocessor(df_data, client.batch_size, use_images, model_path)
    preprocessor.start()

    num_batches = int(np.ceil(len(df_data) / client.batch_size))
    pbar = tqdm(total=num_batches, desc=f"Processing {dataset_name}")

    processed_batch_count = 0
    while processed_batch_count < num_batches:
        try:
            (
                idx,
                batch_prompts,
                batch_multi_modal_data,
                original_indices_in_batch,
            ) = await asyncio.get_event_loop().run_in_executor(
                ThreadPoolExecutor(max_workers=1),
                lambda: preprocessor.queue.get(timeout=120),
            )

            if not batch_prompts and not any(p is not None for p in batch_prompts):
                print(
                    f"[Main Loop Warn] Batch starting index {idx} preprocessed to empty or all invalid data, skipping this batch."
                )
                pbar.update(1)
                processed_batch_count += 1
                continue

            batch_results_from_vllm = await process_batch_async(
                client=client,
                preprocessed_prompts=batch_prompts,
                preprocessed_multi_modal_data=batch_multi_modal_data,
                target_phrase="<answer>",
                max_total_tokens=8196 * 2,
                temperature=temperature,
                top_k=top_k,
            )

            if batch_results_from_vllm:
                for k, (probabilities, generated_text) in enumerate(
                    batch_results_from_vllm
                ):
                    if k >= len(original_indices_in_batch):
                        print(
                            f"[Main Loop Warn] Result index {k} out of bounds for original_indices_in_batch. Skipping."
                        )
                        continue

                    original_row_idx = original_indices_in_batch[k]
                    if original_row_idx >= len(df_data):
                        print(
                            f"[Main Loop Warn] Original row index {original_row_idx} out of DataFrame range, skipping."
                        )
                        continue

                    row_data = df_data.iloc[original_row_idx]

                    try:
                        true_label = row_data["reward_model"]["ground_truth"][0]
                        smiles_string = row_data["extra_info"]["smiles"]
                    except KeyError as ke:
                        print(
                            f"[Main Loop Error] Sample {original_row_idx} is missing key fields (reward_model/extra_info): {ke}. Skipping this sample."
                        )
                        continue
                    except IndexError as ie:
                        print(
                            f"[Main Loop Error] Sample {original_row_idx} has ground_truth index error: {ie}. Skipping this sample."
                        )
                        continue

                    predicted_score = None
                    if probabilities and "true_ratio" in probabilities:
                        predicted_score = probabilities["true_ratio"]

                    if predicted_score is not None:
                        all_true_labels.append(true_label)
                        all_predicted_scores.append(predicted_score)

                        if is_multi_task:
                            task_name_from_data = row_data["extra_info"].get(
                                "task_name"
                            )
                            if (
                                task_name_from_data
                                and task_name_from_data in sider_results_by_task
                            ):
                                sider_results_by_task[task_name_from_data][
                                    "true_labels"
                                ].append(true_label)
                                sider_results_by_task[task_name_from_data][
                                    "predicted_scores"
                                ].append(predicted_score)
                            else:
                                print(
                                    f"[Main Loop Warn] Sample {original_row_idx} has missing or unknown task name: {task_name_from_data}"
                                )

                        result_to_add = {
                            "original_index": original_row_idx,
                            "input_smiles": smiles_string,
                            "messages": row_data["prompt"],
                            "images": row_data.get("images"),
                            "true_label": float(true_label),
                            "raw_predicted_output": generated_text,
                            "parsed_predicted_score": (
                                float(predicted_score)
                                if predicted_score is not None
                                else None
                            ),
                            "ground_truth_features": row_data["extra_info"].get(
                                "ground_truth_features"
                            ),
                            "task_name": row_data["extra_info"].get("task_name"),
                            "probabilities": probabilities,
                        }
                        all_results_to_save.append(result_to_add)

                    else:
                        print(
                            f"[Main Loop Warn] Sample {original_row_idx} could not get valid prediction score, skipping saving result."
                        )
            else:
                print(
                    f"[Main Loop Warn] Batch starting index {idx} returned no results from VLLM."
                )

            pbar.update(1)
            processed_batch_count += 1

            # --- MODIFICATION START: Periodic Saving Logic ---
            if (
                processed_batch_count > 0
                and processed_batch_count % SAVE_INTERVAL_BATCHES == 0
                and all_results_to_save
            ):
                part_file = f"{out_dir}/results_part_{part_number}.parquet"
                print(
                    f"\nReached save interval, saving {len(all_results_to_save)} results to {part_file}..."
                )
                df_part_results = pd.DataFrame(all_results_to_save)
                try:
                    df_part_results.to_parquet(part_file, engine="pyarrow", index=False)
                    print(f"Successfully saved partial results.")
                    all_results_to_save.clear()
                    part_number += 1
                except Exception as e:
                    print(
                        f"[Save Results Error] Failed to save partial results to {part_file}: {e}"
                    )
                    traceback.print_exc()
            # --- MODIFICATION END ---

        except Exception as e:
            if isinstance(e, asyncio.TimeoutError):
                print(
                    f"[Main Loop Timeout] Timeout getting data from queue. Queue status: empty={preprocessor.queue.empty()}, worker_alive={preprocessor.worker_process.is_alive()}"
                )
                if (
                    preprocessor.queue.empty()
                    and not preprocessor.worker_process.is_alive()
                ):
                    print(
                        "[Main Loop] Queue is empty and preload process has stopped, finishing all batch processing."
                    )
                    break
                continue
            print(f"[Main Loop Error] Unexpected error processing batch: {e}")
            traceback.print_exc()
            pbar.update(1)
            processed_batch_count += 1
            continue

    # --- MODIFICATION START: Save any remaining results ---
    if all_results_to_save:
        part_file = f"{out_dir}/results_part_{part_number}.parquet"
        print(
            f"\nSaving final {len(all_results_to_save)} remaining results to {part_file}..."
        )
        df_final_results = pd.DataFrame(all_results_to_save)
        try:
            df_final_results.to_parquet(part_file, engine="pyarrow", index=False)
            print("Final results saved successfully.")
        except Exception as e:
            print(
                f"[Save Results Error] Failed to save final results to {part_file}: {e}"
            )
            traceback.print_exc()
    else:
        print("No remaining results to save.")
    # --- MODIFICATION END ---

    # Cleanup
    preprocessor.stop()
    pbar.close()

    if len(all_true_labels) > 0 and len(all_predicted_scores) == len(all_true_labels):
        print(f"\n--- Evaluation Summary: {dataset_name} - {experiment_name} ---")
        metrics = {}

        if is_multi_task:
            for task_name, data in sider_results_by_task.items():
                if len(data["true_labels"]) > 0:
                    roc_auc = calculate_roc_auc(
                        data["true_labels"], data["predicted_scores"]
                    )
                    if roc_auc is not None:
                        print(f"Task {task_name} ROC-AUC: {roc_auc:.4f}")
                        metrics[f"{task_name}_roc_auc"] = roc_auc
                    else:
                        print(
                            f"--- Could not calculate ROC-AUC for task {task_name} (insufficient data) ---"
                        )
                else:
                    print(
                        f"--- No valid samples for task {task_name} to calculate ROC-AUC ---"
                    )

            overall_roc_auc = calculate_roc_auc(all_true_labels, all_predicted_scores)
            if overall_roc_auc is not None:
                print(f"Overall ROC-AUC for {dataset_name}: {overall_roc_auc:.4f}")
                metrics["overall_roc_auc"] = overall_roc_auc
            else:
                print(
                    f"--- Could not calculate Overall ROC-AUC for {dataset_name} (insufficient data) ---"
                )

        else:
            roc_auc = calculate_roc_auc(all_true_labels, all_predicted_scores)
            if roc_auc is not None:
                print(f"Calculated ROC-AUC: {roc_auc:.4f}")
                metrics["roc_auc"] = roc_auc
            else:
                print(
                    f"--- Could not calculate ROC-AUC for {dataset_name} - {experiment_name} (insufficient data) ---"
                )

        save_metrics_summary(dataset_name, experiment_name, metrics, output_path)
        print("--------------------------------------------------")
        return metrics
    else:
        print(
            f"\n--- Not enough valid samples to calculate metrics for {dataset_name} - {experiment_name} ---"
        )
        return None


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    asyncio.run(main())
