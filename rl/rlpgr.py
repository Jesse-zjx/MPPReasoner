import re
import ast
from typing import List, Dict, Any, Optional
import numpy as np


PRINCIPLE_TERMS = {
    "electronic effect",
    "inductive effect",
    "conjugation",
    "hyperconjugation",
    "field effect",
    "steric effect",
    "steric hindrance",
    "heavy atom effect",
    "chelate effect",
    "electron donating",
    "electron withdrawing",
    "interaction",
    "van der waals",
    "pi-pi stacking",
    "pi-pi interaction",
    "hydrogen bond",
    "theory",
    "principle",
    "acid-base theory",
    "hsab",
    "hard soft acid base",
    "molecular orbital theory",
    "fmo",
    "frontier molecular orbital",
    "homo",
    "lumo",
    "aromaticity",
    "hückel's rule",
    "coordination theory",
    "hydrophobic",
    "hydrophilic",
    "hydrophobicity",
    "solubility",
    "permeability",
    "activity",
    "reactivity",
    "stability",
    "polarity",
    "pka",
    "logp",
    "bond energy",
}
COMPARATIVE_TERMS = {
    "compare",
    "contrast",
    "similar",
    "differ",
    "differentiating",
    "analogy",
    "analogous",
    "in contrast to",
    "similarly",
    "as with",
    "compared to",
    "ranking",
    "order",
    "case study",
    "example",
    "greater than",
    "less than",
    "unlike",
}
EVIDENCE_TERMS = {
    "group",
    "feature",
    "motif",
    "scaffold",
    "structure",
    "moiety",
    "substructure",
    "functional group",
    "nitro",
    "cyano",
    "aldehyde",
    "carboxyl",
    "hydroxyl",
    "alkoxy",
    "alkyl",
    "halo",
    "amino",
    "phenyl",
    "amide",
    "ester",
    "t-butyl",
    "isopropyl",
    "chiral center",
    "conformation",
    "fused ring",
    "side chain",
    "backbone",
    "pharmacophore",
}
WEIGHING_TERMS = {
    "weighing",
    "balancing",
    "overall",
    "dominant",
    "outweighs",
    "on balance",
    "considering",
    "based on the above",
    "therefore",
    "in conclusion",
    "ultimately",
    "the net effect is",
    "consequently",
    "taking all into account",
}
AFFIRMATIVE_CONFIRMATION_TERMS = {
    "affirmative",
    "prevails",
    "stronger case",
    "supports the property",
    "likely true",
    "is predicted to be true",
    "judged as true",
    "consequently is",
    "therefore has",
}
NEGATIVE_CONFIRMATION_TERMS = {
    "negative",
    "counter-argument",
    "more compelling",
    "lacks the property",
    "likely false",
    "is predicted to be false",
    "judged as false",
    "consequently is not",
    "therefore lacks",
}


def extract_think_content(response: str) -> Optional[str]:
    match = re.search(r"<think>(.*?)</think>", response, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def extract_prediction(response: str) -> Optional[str]:
    try:
        answer_match = re.search(
            r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE
        )
        if answer_match:
            prediction = answer_match.group(1).strip().lower()
            if prediction in ["true", "false"]:
                return prediction
    except Exception:
        return None
    return None


def _parse_ground_truth(ground_truth: Any) -> Optional[float]:
    if isinstance(ground_truth, (int, float)):
        return float(ground_truth)
    try:
        val = ast.literal_eval(str(ground_truth))
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, (list, np.ndarray)) and len(val) > 0:
            return float(val[0])
    except (ValueError, SyntaxError, IndexError):
        return None
    return None


def check_text_contains_any(text: str, terms: set) -> bool:
    return any(term in text.lower() for term in terms)


def compute_reward(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict] = None,
) -> Dict[str, Any]:
    lambda1 = 1.0
    r_ans = 0.0
    r_fmt = 0.0
    lambda2 = 0.25
    r_cons = 0.0
    r_comp = 0.0
    lambda3 = 0.25
    r_prin = 0.0
    r_struct = 0.0

    think_content = extract_think_content(solution_str)
    predicted_label_str = extract_prediction(solution_str)
    predicted_label_float = (
        None
        if predicted_label_str is None
        else (1.0 if predicted_label_str == "true" else 0.0)
    )
    true_label_float = _parse_ground_truth(ground_truth)
    if predicted_label_float is not None and true_label_float is not None:
        r_ans = predicted_label_float == true_label_float
    if think_content is not None and predicted_label_float is not None:
        r_fmt = 1.0
        if check_text_contains_any(think_content, WEIGHING_TERMS):
            if predicted_label_float == 1.0 and check_text_contains_any(
                think_content, AFFIRMATIVE_CONFIRMATION_TERMS
            ):
                r_cons = 1.0
            elif predicted_label_float == 0.0 and check_text_contains_any(
                think_content, NEGATIVE_CONFIRMATION_TERMS
            ):
                r_cons = 1.0
        if check_text_contains_any(think_content, COMPARATIVE_TERMS):
            r_comp = 1.0
        if check_text_contains_any(think_content, PRINCIPLE_TERMS):
            r_prin = 1.0
        if check_text_contains_any(think_content, EVIDENCE_TERMS):
            r_struct = 1.0

    total_reward = (
        lambda1 * (r_ans + r_fmt)
        + lambda2 * (r_cons + r_comp)
        + lambda3 * (r_prin + r_struct)
    )

    return {
        "score": total_reward,
        "r_ans": r_ans,
        "r_fmt": r_fmt,
        "r_cons": r_cons,
        "r_comp": r_comp,
        "r_prin": r_prin,
        "r_struct": r_struct,
    }
