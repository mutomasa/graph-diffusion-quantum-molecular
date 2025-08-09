# generator.py
from typing import List, Tuple, Optional, Generator
import os
import subprocess
import tempfile
import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

class DiffusionSampler:
    """
    既存リポ（E(3)-equivariant diffusion / DiGress 等）の
    sample() を包む薄いアダプタ。ここは擬似実装。
    """
    def __init__(self, ckpt_path: str | None = None, device: str = "cpu"):
        """
        ckpt_path: E(3)-equivariant diffusion のチェックポイントパス
        device: "cpu" or "cuda"

        外部実装呼び出しの優先順:
        1) 環境変数 E3DM_SMILES_CMD（SMILES出力）/E3DM_COORDS_CMD（座標出力）
        2) third_party/e3_equivariant/ 以下のスクリプトを推測
        3) フォールバック（デモ用既知 SMILES）
        """
        self.ckpt_path = ckpt_path
        self.device = device
        self.repo_root = os.path.join(os.path.dirname(__file__), "third_party", "e3_equivariant")

    def sample_smiles(self, n: int = 8, cond: Optional[dict] = None) -> List[str]:
        # 1) 環境変数で明示されたコマンド
        cmd_tpl = os.environ.get("E3DM_SMILES_CMD")
        if cmd_tpl:
            out = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
            cmd = cmd_tpl.format(ckpt=self.ckpt_path or "", n=n, out=out, device=self.device)
            try:
                subprocess.run(cmd, shell=True, check=True)
                with open(out) as f:
                    smiles = [line.strip() for line in f if line.strip()]
                if smiles:
                    return smiles[:n]
            except Exception:
                pass

        # 2) third_party 推測（例: scripts/sample_smiles.py がある想定）
        cand = os.path.join(self.repo_root, "scripts", "sample_smiles.py")
        if os.path.isfile(cand):
            out = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
            cmd = [
                "python", cand,
                "--ckpt", self.ckpt_path or "",
                "--num", str(n),
                "--out", out,
                "--device", self.device,
            ]
            try:
                subprocess.run(cmd, check=True)
                with open(out) as f:
                    smiles = [line.strip() for line in f if line.strip()]
                if smiles:
                    return smiles[:n]
            except Exception:
                pass

        # 3) フォールバック（デモ用の既知 SMILES）
        fallback = ["CCO", "c1ccccc1O", "CCN(CC)CC", "CC(=O)O", "O=C(N)C", "CCOC(=O)C"]
        return fallback[:n]

    def sample_smiles_stream(self, n: int = 8, cond: Optional[dict] = None) -> Generator[str, None, None]:
        """Yield SMILES one-by-one for real-time visualization.

        Strategy:
        - If E3DM_SMILES_CMD is set, call it per-sample with --num 1 and yield line
        - Else, fall back to the local list, yielding gradually
        """
        cmd_tpl = os.environ.get("E3DM_SMILES_CMD")
        if cmd_tpl:
            for _ in range(n):
                out = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
                cmd = cmd_tpl.format(ckpt=self.ckpt_path or "", n=1, out=out, device=self.device)
                try:
                    subprocess.run(cmd, shell=True, check=True)
                    with open(out) as f:
                        for line in f:
                            smi = line.strip()
                            if smi:
                                yield smi
                                break
                except Exception:
                    continue
        else:
            for smi in ["CCO", "c1ccccc1O", "CCN(CC)CC", "CC(=O)O", "O=C(N)C", "CCOC(=O)C"][:n]:
                yield smi

    def sample_graph_with_coords(self) -> Tuple[list, list, np.ndarray]:
        """
        E(3)等の実モデルから座標を取得できる場合は、
        JSON 形式で {"atoms":[Z,...], "bonds":[[i,j,type],...], "pos":[[x,y,z],...]} を受け取る想定。
        """
        # 1) 環境変数で明示されたコマンド
        cmd_tpl = os.environ.get("E3DM_COORDS_CMD")
        if cmd_tpl:
            out = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
            cmd = cmd_tpl.format(ckpt=self.ckpt_path or "", out=out, device=self.device)
            try:
                subprocess.run(cmd, shell=True, check=True)
                with open(out) as f:
                    data = json.load(f)
                atoms = data.get("atoms", [])
                bonds = data.get("bonds", [])
                pos = np.array(data.get("pos", []), dtype=float)
                if len(atoms) and pos.size:
                    return atoms, bonds, pos
            except Exception:
                pass

        # 2) フォールバック：SMILES→3D埋め込み
        smi = "c1ccccc1O"  # フェノール
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
        atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
        bonds = []
        for b in mol.GetBonds():
            t = "SINGLE"
            if b.GetBondType().name == "DOUBLE":
                t = "DOUBLE"
            if b.GetIsAromatic():
                t = "AROM"
            bonds.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx(), t))
        pos = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=float)
        return atoms, bonds, pos
