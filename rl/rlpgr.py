import re
import ast
from typing import List, Dict, Any, Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem.rdMolChemicalFeatures import GetElementCounts
from rdkit.Chem.Fragments import *

STRUCTURAL_FEATURES = {
    "hydroxyl": {
        "keywords": ["hydroxyl", "oh group", "-oh", "alcohol", "hydroxy"],
        "verification": "verify_hydroxyl"
    },
    "carboxyl": {
        "keywords": ["carboxyl", "carboxylic acid", "cooh", "-cooh", "carboxylate"],
        "verification": "verify_carboxyl"
    },
    "amino": {
        "keywords": ["amino", "amine", "nh2", "-nh2", "primary amine", "secondary amine", "tertiary amine"],
        "verification": "verify_amino"
    },
    "nitro": {
        "keywords": ["nitro", "no2", "-no2", "nitro group"],
        "verification": "verify_nitro"
    },
    "ester": {
        "keywords": ["ester", "coo", "-coo-", "ester group", "ester linkage"],
        "verification": "verify_ester"
    },
    "amide": {
        "keywords": ["amide", "conh", "-conh-", "amide bond", "peptide bond"],
        "verification": "verify_amide"
    },
    "ether": {
        "keywords": ["ether", "-o-", "ether linkage", "alkoxy"],
        "verification": "verify_ether"
    },
    "ketone": {
        "keywords": ["ketone", "carbonyl", "c=o", "keto"],
        "verification": "verify_ketone"
    },
    "aldehyde": {
        "keywords": ["aldehyde", "cho", "-cho", "aldehyde group"],
        "verification": "verify_aldehyde"
    },
    "cyano": {
        "keywords": ["cyano", "nitrile", "cn", "-cn", "cyano group"],
        "verification": "verify_cyano"
    },
    "benzene": {
        "keywords": ["benzene", "phenyl", "aromatic ring", "benzene ring"],
        "verification": "verify_benzene"
    },
    "pyridine": {
        "keywords": ["pyridine", "pyridinyl", "pyridine ring"],
        "verification": "verify_pyridine"
    },
    "indole": {
        "keywords": ["indole", "indolyl", "tryptophan", "indole system"],
        "verification": "verify_indole"
    },
    "quinoline": {
        "keywords": ["quinoline", "quinolinyl", "quinoline system"],
        "verification": "verify_quinoline"
    },
    "naphthalene": {
        "keywords": ["naphthalene", "naphthyl", "naphthalene ring"],
        "verification": "verify_naphthalene"
    },
    "cyclohexane": {
        "keywords": ["cyclohexane", "cyclohexyl", "six-membered ring", "6-membered ring"],
        "verification": "verify_cyclohexane"
    },
    "cyclopentane": {
        "keywords": ["cyclopentane", "cyclopentyl", "five-membered ring", "5-membered ring"],
        "verification": "verify_cyclopentane"
    },
    "heterocycle": {
        "keywords": ["heterocycle", "heterocyclic", "heteroatom ring"],
        "verification": "verify_heterocycle"
    },
    
    "fluorine": {
        "keywords": ["fluorine", "fluoro", "f", "-f", "fluorinated"],
        "verification": "verify_fluorine"
    },
    "chlorine": {
        "keywords": ["chlorine", "chloro", "cl", "-cl", "chlorinated"],
        "verification": "verify_chlorine"
    },
    "bromine": {
        "keywords": ["bromine", "bromo", "br", "-br", "brominated"],
        "verification": "verify_bromine"
    },
    
    "chiral": {
        "keywords": ["chiral", "stereocenter", "asymmetric", "chiral center", "stereogenic"],
        "verification": "verify_chiral"
    },
    "branched": {
        "keywords": ["branched", "tert-butyl", "isopropyl", "tertiary", "quaternary"],
        "verification": "verify_branched"
    },
    "linear": {
        "keywords": ["linear", "straight chain", "unbranched", "n-alkyl"],
        "verification": "verify_linear"
    }
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
CHEMICAL_PRINCIPLES = {
    "lipinski": {
        "keywords": ["lipinski", "rule of five", "ro5", "drug-like", "drug likeness"],
        "verification": "verify_lipinski_rule"
    },
    "logp": {
        "keywords": ["logp", "log p", "lipophilicity", "lipophilic", "hydrophobicity", "hydrophobic", "partition coefficient"],
        "verification": "verify_logp_principle"
    },
    "hydrogen_bonding": {
        "keywords": ["hydrogen bond", "h-bond", "hbd", "hba", "donor", "acceptor", "polar interaction"],
        "verification": "verify_hydrogen_bonding"
    },
    "molecular_weight": {
        "keywords": ["molecular weight", "mw", "heavy", "bulky", "size", "steric bulk"],
        "verification": "verify_molecular_weight"
    },
    "aromaticity": {
        "keywords": ["aromatic", "benzene", "phenyl", "pi-pi", "π-π", "aromatic ring", "conjugated"],
        "verification": "verify_aromaticity"
    },
    "polarity": {
        "keywords": ["polar", "polarity", "dipole", "electronegativity"],
        "verification": "verify_polarity"
    },
    "steric_hindrance": {
        "keywords": ["steric", "hindrance", "crowded", "bulky group", "steric clash"],
        "verification": "verify_steric_hindrance"
    },
    "rotatable_bonds": {
        "keywords": ["flexible", "flexibility", "rotatable", "conformational"],
        "verification": "verify_flexibility"
    }
}


def get_molecule_properties(smiles: str) -> Dict[str, float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    
    return {
        "mw": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "polar_surface_area": rdMolDescriptors.CalcTPSA(mol),
        "num_rings": rdMolDescriptors.CalcNumRings(mol),
    }


def verify_lipinski_rule(reasoning: str, mol_props: Dict[str, float]) -> bool:
    if not mol_props:
        return False
    
    rule_violations = 0
    if mol_props.get("mw", 0) > 500:
        rule_violations += 1
    if mol_props.get("logp", 0) > 5:
        rule_violations += 1
    if mol_props.get("hbd", 0) > 5:
        rule_violations += 1
    if mol_props.get("hba", 0) > 10:
        rule_violations += 1
    reasoning_lower = reasoning.lower()
    if rule_violations >= 2:
        return any(term in reasoning_lower for term in ["violate", "violation", "exceed", "above", "beyond", "not comply", "fail"])
    else:
        return any(term in reasoning_lower for term in ["comply", "satisfy", "within", "pass", "meet", "acceptable"])


def verify_logp_principle(reasoning: str, mol_props: Dict[str, float]) -> bool:
    if not mol_props:
        return False
    
    logp = mol_props.get("logp", 0)
    reasoning_lower = reasoning.lower()
    if logp > 3:
        return any(term in reasoning_lower for term in ["hydrophobic", "lipophilic", "high logp", "lipid soluble"])
    elif logp < 1:
        return any(term in reasoning_lower for term in ["hydrophilic", "polar", "water soluble", "low logp"])
    else:
        return True


def verify_hydrogen_bonding(reasoning: str, mol_props: Dict[str, float]) -> bool:
    if not mol_props:
        return False
    
    hbd = mol_props.get("hbd", 0)
    hba = mol_props.get("hba", 0)
    reasoning_lower = reasoning.lower()
    
    has_hb_capability = hbd > 0 or hba > 0
    mentions_hb = any(term in reasoning_lower for term in ["hydrogen bond", "h-bond", "donor", "acceptor"])
    
    if mentions_hb and has_hb_capability:
        return True
    elif not mentions_hb and hbd + hba == 0:
        return True
    else:
        return False


def verify_molecular_weight(reasoning: str, mol_props: Dict[str, float]) -> bool:
    if not mol_props:
        return False
    
    mw = mol_props.get("mw", 0)
    reasoning_lower = reasoning.lower()
    
    if mw > 500:
        return any(term in reasoning_lower for term in ["large", "heavy", "high molecular weight", "bulky"])
    elif mw < 200:
        return any(term in reasoning_lower for term in ["small", "light", "low molecular weight"])
    else:
        return True


def verify_aromaticity(reasoning: str, mol_props: Dict[str, float]) -> bool:
    if not mol_props:
        return False
    
    aromatic_rings = mol_props.get("aromatic_rings", 0)
    reasoning_lower = reasoning.lower()
    
    has_aromatic = aromatic_rings > 0
    mentions_aromatic = any(term in reasoning_lower for term in ["aromatic", "benzene", "phenyl", "pi-pi", "π-π"])
    
    return (has_aromatic and mentions_aromatic) or (not has_aromatic and not mentions_aromatic)


def verify_polarity(reasoning: str, mol_props: Dict[str, float]) -> bool:
    if not mol_props:
        return False
    
    psa = mol_props.get("polar_surface_area", 0)
    reasoning_lower = reasoning.lower()
    
    if psa > 140:
        return any(term in reasoning_lower for term in ["polar", "high polarity", "hydrophilic"])
    elif psa < 60:
        return any(term in reasoning_lower for term in ["nonpolar", "low polarity", "hydrophobic"])
    else:
        return True


def verify_steric_hindrance(reasoning: str, mol_props: Dict[str, float]) -> bool:
    if not mol_props:
        return False
    
    heavy_atoms = mol_props.get("heavy_atoms", 0)
    reasoning_lower = reasoning.lower()
    
    if heavy_atoms > 30:
        return any(term in reasoning_lower for term in ["steric", "hindrance", "crowded", "bulky"])
    else:
        return True


def verify_flexibility(reasoning: str, mol_props: Dict[str, float]) -> bool:
    if not mol_props:
        return False
    
    rotatable_bonds = mol_props.get("rotatable_bonds", 0)
    reasoning_lower = reasoning.lower()
    
    if rotatable_bonds > 5:
        return any(term in reasoning_lower for term in ["flexible", "flexibility", "rotatable", "conformational freedom"])
    elif rotatable_bonds == 0:
        return any(term in reasoning_lower for term in ["rigid", "inflexible", "constrained"])
    else:
        return True
    

def create_mol_from_smiles(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            Chem.SanitizeMol(mol)
        return mol
    except:
        return None


def verify_hydroxyl(mol) -> bool:
    if mol is None:
        return False
    return fr_Al_OH(mol) > 0 or fr_Ar_OH(mol) > 0


def verify_carboxyl(mol) -> bool:
    if mol is None:
        return False
    return fr_COO(mol) > 0 or fr_COO2(mol) > 0


def verify_amino(mol) -> bool:
    if mol is None:
        return False
    return fr_NH2(mol) > 0 or fr_NH1(mol) > 0 or fr_NH0(mol) > 0


def verify_nitro(mol) -> bool:
    if mol is None:
        return False
    return fr_nitro(mol) > 0


def verify_ester(mol) -> bool:
    if mol is None:
        return False
    return fr_ester(mol) > 0


def verify_amide(mol) -> bool:
    if mol is None:
        return False
    return fr_amide(mol) > 0


def verify_ether(mol) -> bool:
    if mol is None:
        return False
    return fr_ether(mol) > 0


def verify_ketone(mol) -> bool:
    if mol is None:
        return False
    return fr_ketone(mol) > 0 or fr_ketone_Topliss(mol) > 0


def verify_aldehyde(mol) -> bool:
    if mol is None:
        return False
    return fr_aldehyde(mol) > 0


def verify_cyano(mol) -> bool:
    if mol is None:
        return False
    return fr_nitrile(mol) > 0


def verify_benzene(mol) -> bool:
    if mol is None:
        return False
    return fr_benzene(mol) > 0


def verify_pyridine(mol) -> bool:
    if mol is None:
        return False
    return fr_pyridine(mol) > 0


def verify_indole(mol) -> bool:
    if mol is None:
        return False
    return fr_indole(mol) > 0


def verify_quinoline(mol) -> bool:
    if mol is None:
        return False
    quinoline_pattern = Chem.MolFromSmarts('c1ccc2ncccc2c1')
    if quinoline_pattern is None:
        return False
    return len(mol.GetSubstructMatches(quinoline_pattern)) > 0


def verify_naphthalene(mol) -> bool:
    if mol is None:
        return False
    naphthalene_pattern = Chem.MolFromSmarts('c1ccc2ccccc2c1')
    if naphthalene_pattern is None:
        return False
    return len(mol.GetSubstructMatches(naphthalene_pattern)) > 0


def verify_cyclohexane(mol) -> bool:
    if mol is None:
        return False
    cyclohexane_pattern = Chem.MolFromSmarts('C1CCCCC1')
    if cyclohexane_pattern is None:
        return False
    return len(mol.GetSubstructMatches(cyclohexane_pattern)) > 0


def verify_cyclopentane(mol) -> bool:
    if mol is None:
        return False
    cyclopentane_pattern = Chem.MolFromSmarts('C1CCCC1')
    if cyclopentane_pattern is None:
        return False
    return len(mol.GetSubstructMatches(cyclopentane_pattern)) > 0


def verify_heterocycle(mol) -> bool:
    if mol is None:
        return False
    # 检查是否有包含杂原子的环
    for ring in mol.GetRingInfo().AtomRings():
        for atom_idx in ring:
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetSymbol() not in ['C', 'H']:
                return True
    return False


def verify_fluorine(mol) -> bool:
    if mol is None:
        return False
    return fr_halogen(mol) > 0 and any(atom.GetSymbol() == 'F' for atom in mol.GetAtoms())


def verify_chlorine(mol) -> bool:
    if mol is None:
        return False
    return fr_halogen(mol) > 0 and any(atom.GetSymbol() == 'Cl' for atom in mol.GetAtoms())


def verify_bromine(mol) -> bool:
    if mol is None:
        return False
    return fr_halogen(mol) > 0 and any(atom.GetSymbol() == 'Br' for atom in mol.GetAtoms())


def verify_chiral(mol) -> bool:
    if mol is None:
        return False
    return len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)) > 0


def verify_branched(mol) -> bool:
    if mol is None:
        return False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C' and atom.GetDegree() > 2 and not atom.IsInRing():
            return True
    return False


def verify_linear(mol) -> bool:
    if mol is None:
        return False
    carbon_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'C']
    if len(carbon_atoms) == 0:
        return False
    linear_carbons = [atom for atom in carbon_atoms if atom.GetDegree() <= 2]
    return len(linear_carbons) / len(carbon_atoms) > 0.7


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

def check_comparative_analysis(think_content: str, few_shot_examples: List[str]) -> float:
    think_lower = think_content.lower()
    for smiles in few_shot_examples:
        if smiles.lower() in think_lower:
            return 1.0
    return 0.0


def check_chemical_principle_application(reasoning: str, smiles: str) -> float:
    mol_props = get_molecule_properties(smiles)
    reasoning_lower = reasoning.lower()
    
    for principle_name, principle_info in CHEMICAL_PRINCIPLES.items():
        if any(keyword in reasoning_lower for keyword in principle_info["keywords"]):
            verification_func = globals().get(principle_info["verification"])
            if verification_func(reasoning, mol_props):
                return 1.0
    return 0.0


def check_molecular_structure_analysis(reasoning: str, smiles: str) -> float:
    mol = create_mol_from_smiles(smiles)
    if mol is None:
        return 0.0
    reasoning_lower = reasoning.lower()

    for feature_name, feature_info in STRUCTURAL_FEATURES.items():
        if any(keyword in reasoning_lower for keyword in feature_info["keywords"]):
            verification_func = globals().get(feature_info["verification"])
            if verification_func and verification_func(mol):
                return 1.0
    return 0.0


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

    # reasoning block and a parseable <answer> prediction.
    if think_content is not None and predicted_label_float is not None:
        r_fmt = 1.0

    # --- Correctness gating ---
    if predicted_label_float is None or true_label_float is None:
        # Malformed output: missing reasoning block or unparseable prediction.
        total_reward = -1.5
    elif predicted_label_float != true_label_float:
        # Incorrect prediction: no auxiliary reward is accumulated.
        total_reward = -1.0
    else:
        # Correct prediction: evaluate and add reasoning and chemistry layers.
        r_ans = 1.0
        if check_text_contains_any(think_content, WEIGHING_TERMS):
            if predicted_label_float == 1.0 and check_text_contains_any(
                think_content, AFFIRMATIVE_CONFIRMATION_TERMS
            ):
                r_cons = 1.0
            elif predicted_label_float == 0.0 and check_text_contains_any(
                think_content, NEGATIVE_CONFIRMATION_TERMS
            ):
                r_cons = 1.0
        r_comp = check_comparative_analysis(think_content, extra_info["few_shot_examples"])
        r_prin = check_chemical_principle_application(think_content, extra_info["smiles"])
        r_struct = check_molecular_structure_analysis(think_content, extra_info["smiles"])
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
