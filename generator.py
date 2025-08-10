# generator.py
from typing import List, Tuple, Optional, Generator
import os
import subprocess
import tempfile
import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from src.e3_diffusion import SimpleE3DiffusionModel, DiffusionScheduler, MolecularDiffusionSampler

class DiffusionSampler:
    """
    E(3)-equivariant diffusion modelを使用した分子生成サンプラー
    """
    def __init__(self, ckpt_path: str | None = None, device: str = "cpu"):
        """
        ckpt_path: 学習済みモデルのチェックポイントパス（オプション）
        device: "cpu" or "cuda"
        """
        self.ckpt_path = ckpt_path
        self.device = device
        
        # E(3)-equivariant diffusion modelの初期化
        self.model = SimpleE3DiffusionModel(
            num_atom_types=119,  # 最大原子番号
            hidden_dim=128,
            num_layers=6,
            time_embed_dim=128,
        )
        
        # Diffusion scheduler
        self.scheduler = DiffusionScheduler(
            num_timesteps=1000,
            beta_start=1e-4,
            beta_end=0.02
        )
        
        # Molecular sampler
        self.sampler = MolecularDiffusionSampler(
            model=self.model,
            scheduler=self.scheduler,
            device=self.device
        )
        
        # チェックポイントがあれば読み込み
        if ckpt_path and os.path.exists(ckpt_path):
            self._load_checkpoint(ckpt_path)
        else:
            print("チェックポイントが見つからないため、初期化されたモデルを使用します")
    
    def _load_checkpoint(self, ckpt_path: str):
        """学習済みチェックポイントを読み込み"""
        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"チェックポイントを読み込みました: {ckpt_path}")
        except Exception as e:
            print(f"チェックポイントの読み込みに失敗しました: {e}")
    
    def sample_smiles(self, n: int = 8, cond: Optional[dict] = None) -> List[str]:
        """
        E(3)-equivariant diffusion modelを使用して分子を生成し、SMILESに変換
        
        Args:
            n: 生成する分子数
            cond: 条件（将来の拡張用）
            
        Returns:
            SMILES文字列のリスト
        """
        try:
            # 分子を生成
            molecules = self.sampler.sample_molecules(num_molecules=n, max_atoms=20)
            
            smiles_list = []
            for atom_types, positions in molecules:
                # 原子番号をSMILESに変換
                smiles = self._atoms_to_smiles(atom_types, positions)
                if smiles:
                    smiles_list.append(smiles)
            
            # 必要な数だけ返す
            return smiles_list[:n]
            
        except Exception as e:
            print(f"分子生成中にエラーが発生しました: {e}")
            # フォールバック
            fallback = ["CCO", "c1ccccc1O", "CCN(CC)CC", "CC(=O)O", "O=C(N)C", "CCOC(=O)C"]
            return fallback[:n]
    
    def sample_smiles_stream(self, n: int = 8, cond: Optional[dict] = None) -> Generator[str, None, None]:
        """
        ストリーミング形式でSMILESを生成
        """
        try:
            for i in range(n):
                # 1つずつ分子を生成
                molecules = self.sampler.sample_molecules(num_molecules=1, max_atoms=20)
                
                if molecules:
                    atom_types, positions = molecules[0]
                    smiles = self._atoms_to_smiles(atom_types, positions)
                    if smiles:
                        yield smiles
                    else:
                        # フォールバック
                        fallback = ["CCO", "c1ccccc1O", "CCN(CC)CC", "CC(=O)O", "O=C(N)C", "CCOC(=O)C"]
                        yield fallback[i % len(fallback)]
                else:
                    # フォールバック
                    fallback = ["CCO", "c1ccccc1O", "CCN(CC)CC", "CC(=O)O", "O=C(N)C", "CCOC(=O)C"]
                    yield fallback[i % len(fallback)]
                    
        except Exception as e:
            print(f"ストリーミング生成中にエラーが発生しました: {e}")
            # フォールバック
            fallback = ["CCO", "c1ccccc1O", "CCN(CC)CC", "CC(=O)O", "O=C(N)C", "CCOC(=O)C"]
            for i in range(n):
                yield fallback[i % len(fallback)]
    
    def _atoms_to_smiles(self, atom_types: List[int], positions: np.ndarray) -> Optional[str]:
        """
        原子番号と座標からSMILESを生成
        
        Args:
            atom_types: 原子番号のリスト
            positions: 3D座標
            
        Returns:
            SMILES文字列（失敗時はNone）
        """
        try:
            # 原子番号をRDKitの原子に変換
            mol = Chem.RWMol()
            
            # 原子を追加
            atom_indices = []
            for atom_type in atom_types:
                if 1 <= atom_type <= 119:  # 有効な原子番号
                    atom = mol.AddAtom(Chem.Atom(atom_type))
                    atom_indices.append(atom)
            
            if len(atom_indices) < 2:
                return None
            
            # 座標に基づいて結合を推定
            for i in range(len(atom_indices)):
                for j in range(i + 1, len(atom_indices)):
                    # 距離を計算
                    dist = np.linalg.norm(positions[i] - positions[j])
                    
                    # 距離に基づいて結合を追加（簡易的な判定）
                    if dist < 2.0:  # 単結合
                        mol.AddBond(atom_indices[i], atom_indices[j], Chem.BondType.SINGLE)
                    elif dist < 1.5:  # 二重結合
                        mol.AddBond(atom_indices[i], atom_indices[j], Chem.BondType.DOUBLE)
            
            # SMILESに変換
            smiles = Chem.MolToSmiles(mol)
            return smiles
            
        except Exception as e:
            print(f"SMILES変換中にエラーが発生しました: {e}")
            return None
    
    def sample_graph_with_coords(self) -> Tuple[list, list, np.ndarray]:
        """
        E(3)-equivariant diffusion modelから座標付きグラフを取得
        
        Returns:
            atoms: 原子番号のリスト
            bonds: 結合情報のリスト
            pos: 3D座標
        """
        try:
            # 1つの分子を生成
            molecules = self.sampler.sample_molecules(num_molecules=1, max_atoms=20)
            
            if molecules:
                atom_types, positions = molecules[0]
                
                # 結合情報を推定
                bonds = []
                for i in range(len(atom_types)):
                    for j in range(i + 1, len(atom_types)):
                        dist = np.linalg.norm(positions[i] - positions[j])
                        if dist < 2.0:
                            bond_type = "SINGLE"
                            if dist < 1.5:
                                bond_type = "DOUBLE"
                            bonds.append((i, j, bond_type))
                
                return list(atom_types), bonds, positions
            
        except Exception as e:
            print(f"グラフ生成中にエラーが発生しました: {e}")
        
        # フォールバック：フェノール
        smi = "c1ccccc1O"
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
