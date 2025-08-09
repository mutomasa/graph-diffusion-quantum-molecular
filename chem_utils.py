# chem_utils.py
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski

# 一部の RDKit ビルドでは MOL2 writer が含まれないため、存在チェックしてフォールバック
_Mol2Writer = getattr(Chem, "MolToMol2File", None)

def smiles_to_3d_mol(smi: str):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol, maxIters=400)
    center_mol(mol)
    return mol

def mol_to_sdf(mol: Chem.Mol, path: str):
    w = Chem.SDWriter(path); w.write(mol); w.close()

def mol_to_mol2(mol: Chem.Mol, path: str):
    if _Mol2Writer is None:
        raise RuntimeError("Mol2 writer is not available in this RDKit build.")
    _Mol2Writer(mol, path)

def basic_props(mol: Chem.Mol) -> dict:
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rot = Lipinski.NumRotatableBonds(mol)
    return {"MolWt": round(mw,2), "LogP": round(logp,2), "HBD": hbd, "HBA": hba, "RB": rot}

def center_mol(mol: Chem.Mol) -> None:
    """Translate conformer so that its centroid is at the origin.

    py3Dmol では座標が大きくオフセットしていると、表示が片側に寄ることがあるため、
    ここで中心化しておく。
    """
    try:
        conf = mol.GetConformer()
    except Exception:
        return
    import numpy as _np
    coords = _np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=float)
    centroid = coords.mean(axis=0)
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (p.x - centroid[0], p.y - centroid[1], p.z - centroid[2]))
