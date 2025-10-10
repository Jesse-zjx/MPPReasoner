"""
vLLM client module for handling model inference and probability calculation.
Supports multi-GPU inference, batch processing, and probability analysis.
"""

import os
import sys
import math
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple, Any

import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent))


class VLLMLogitClient:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        gpus: str = "0,1,2,3",
        max_model_len: int = 16384,
        dtype: str = "bfloat16",
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.gpus = gpus
        self.num_gpus = len(self.gpus.split(","))
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.batch_size = batch_size

        self.model: Optional[LLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None

        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpus
        print(f"[VLLMClient Init] Using GPUs: {self.gpus}")
        print(f"[VLLMClient Init] Number of GPUs: {self.num_gpus}")
        print(f"[VLLMClient Init] Batch size: {self.batch_size}")
        print(f"[VLLMClient Init] Max model length: {self.max_model_len}")
        print(f"[VLLMClient Init] Data type: {self.dtype}")

    def load_model(self) -> None:
        if self.model is None:
            print(
                f"[VLLMClient] Loading vLLM model: {self.model_name} from {self.model_path}"
            )
            try:
                self.model = LLM(
                    model=self.model_path,
                    tensor_parallel_size=self.num_gpus,
                    max_model_len=self.max_model_len,
                    dtype=self.dtype,
                    gpu_memory_utilization=0.9,
                    enforce_eager=True,
                    # max_num_batched_tokens=self.batch_size * self.max_model_len,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                print(f"[VLLMClient] Model loaded successfully")
            except Exception as e:
                print(f"[VLLMClient Error] Model loading failed: {str(e)}")
                raise RuntimeError(f"Model loading failed: {str(e)}")

    def unload_model(self) -> None:
        if self.model is not None:
            print(f"[VLLMClient] Unloading model")
            try:
                del self.model
                del self.tokenizer
                self.model = None
                self.tokenizer = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[VLLMClient] Model unloaded successfully")
            except Exception as e:
                print(f"[VLLMClient Error] Model unloading failed: {str(e)}")

    def batch_get_probabilities(
        self,
        preprocessed_prompts: List[Union[str, Dict]],
        preprocessed_multi_modal_data: Optional[List[Optional[Dict[str, Any]]]] = None,
        target_phrase: str = "<answer>",
        max_total_tokens: int = 256,
        top_k: int = 5,
        temperature: float = 0.0,
        seed: int = 14,
        stop_sequences: Optional[List[str]] = None,
    ) -> List[Tuple[Optional[Dict[str, Any]], str]]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer not loaded")

        if preprocessed_multi_modal_data is None:
            preprocessed_multi_modal_data = [None] * len(preprocessed_prompts)
        elif len(preprocessed_multi_modal_data) != len(preprocessed_prompts):
            print(
                f"[VLLMClient Error] Mismatch in length between preprocessed_prompts ({len(preprocessed_prompts)}) and preprocessed_multi_modal_data ({len(preprocessed_multi_modal_data)}) during batch processing."
            )
            return [(None, "") for _ in preprocessed_prompts]

        inputs_for_generate: List[Union[str, Dict]] = []
        valid_to_original_idx_map: Dict[int, int] = {}
        for i, (p, mmd) in enumerate(zip(preprocessed_prompts, preprocessed_multi_modal_data)):
            if p is None:
                print(f"[VLLMClient Warn] Preprocessed prompt for sample {i} in the batch is empty, skipping generation for this sample.")
                continue

            current_input_obj: Union[str, Dict]
            if mmd is not None and isinstance(mmd, dict) and "image" in mmd and mmd["image"] is not None:
                # VLLM expects {"prompt": "...", "multi_modal_data": {"image": [PIL_Image]}}
                current_input_obj = {"prompt": p, "multi_modal_data": mmd}
            else:
                current_input_obj = p

            valid_to_original_idx_map[len(inputs_for_generate)] = i # Store original index
            inputs_for_generate.append(current_input_obj)
        
        if not inputs_for_generate:
            print("[VLLMClient Warn] All preprocessed prompts are invalid or empty, skipping batch generation.")
            return [(None, "") for _ in preprocessed_prompts]

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_total_tokens,
            logprobs=top_k,
            stop=stop_sequences,
            seed=seed,
        )

        outputs = []
        try:
            outputs = self.model.generate(inputs_for_generate, sampling_params)
        except Exception as e:
            print(f"[VLLMClient Error] Model batch generation failed: {e}")
            # If generation fails, return empty results for all original inputs to maintain index mapping.
            return [(None, "") for _ in preprocessed_prompts]

        # Initialize results list with same length as original input (preprocessed_prompts)
        final_results: List[Tuple[Optional[Dict[str, Any]], str]] = [(None, "") for _ in preprocessed_prompts]

        # Process each output and map it back to its original input position
        for i, output in enumerate(outputs):
            original_input_idx = valid_to_original_idx_map[i]
            gen = output.outputs[0]
            generated_text = gen.text
            chosen_token_ids = gen.token_ids
            generated_token_logprobs_list = gen.logprobs
            probabilities: Optional[Dict[str, Any]] = None

            # Attempt to find probabilities for "True" and "False"
            true_prob = 0.0
            false_prob = 0.0

            generated_text_lower = generated_text.lower()
            target_phrase_lower = target_phrase.lower()

            # Find the position of the target phrase in the generated text
            text_pos = generated_text_lower.rfind(target_phrase_lower)

            if text_pos != -1:
                end_of_target_pos = text_pos + len(target_phrase)

                current_pos = 0
                logprob_idx_after_target = -1

                for i, token_id in enumerate(chosen_token_ids):
                    token_text = self.tokenizer.decode([token_id])
                    current_pos += len(token_text)
                    if current_pos > end_of_target_pos:
                        logprob_idx_after_target = i
                        break
                if logprob_idx_after_target != -1 and logprob_idx_after_target < len(
                    generated_token_logprobs_list
                ):
                    next_token_logprobs_dict = generated_token_logprobs_list[
                        logprob_idx_after_target
                    ]

                    if next_token_logprobs_dict is not None:
                        probabilities = {
                            "token_probs": {},
                            "true_ratio": 0.5,  # Default to 0.5 if not found
                        }

                        for token_id, logprob_obj in next_token_logprobs_dict.items():
                            token_text = self.tokenizer.decode(
                                [token_id]
                            ).strip()  # .strip() to remove potential leading spaces
                            # prob = float(f"{math.exp(logprob_obj.logprob):.6f}")
                            prob = math.exp(logprob_obj.logprob)
                            if token_text in probabilities["token_probs"]:
                                probabilities["token_probs"][token_text] += prob
                            else:
                                probabilities["token_probs"][token_text] = prob

                        true_prob = probabilities["token_probs"].get("True", 0.0)
                        false_prob = probabilities["token_probs"].get("False", 0.0)

                        # Also check for " True", " False" for robustness
                        # true_prob = max(true_prob, probabilities["token_probs"].get(" True", 0.0))
                        # false_prob = max(false_prob, probabilities["token_probs"].get(" False", 0.0))

                        total_prob = true_prob + false_prob

                        if total_prob > 0:
                            probabilities["true_ratio"] = float(true_prob / total_prob)
                        else:
                            # If "True" and "False" not in top_k, but target phrase found,
                            # default to 0.5 or decide a policy. Here, we keep 0.5.
                            probabilities["true_ratio"] = 0.5
                    else:
                        # Logprobs dictionary for the next token is None, assign default 0.5
                        probabilities = {"token_probs": {}, "true_ratio": 0.5}
                else:
                    # Target phrase found, but no tokens follow it for logprob, assign default 0.5
                    probabilities = {"token_probs": {}, "true_ratio": 0.5}
            else:
                # Target phrase not found in generated text, assign default 0.5
                probabilities = {"token_probs": {}, "true_ratio": 0.5}

            final_results[original_input_idx] = (probabilities, generated_text)

        return final_results

    def __enter__(self) -> "VLLMLogitClient":
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.unload_model()
