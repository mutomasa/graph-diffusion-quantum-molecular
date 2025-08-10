#!/usr/bin/env python3
"""
E(3)-equivariant diffusion modelのテストスクリプト
"""

import torch
import numpy as np
from src.e3_diffusion import SimpleE3DiffusionModel, DiffusionScheduler, MolecularDiffusionSampler
from generator import DiffusionSampler
from rdkit import Chem
from rdkit.Chem import AllChem

def test_e3_model():
    """E(3)-equivariant diffusion modelの基本テスト"""
    print("=== E(3)-equivariant diffusion model テスト ===")
    
    # デバイス設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")
    
    # モデル初期化
    model = SimpleE3DiffusionModel(
        num_atom_types=119,
        hidden_dim=64,  # テスト用に小さく
        num_layers=3,   # テスト用に小さく
        time_embed_dim=64,
    ).to(device)
    
    print(f"モデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    
    # テストデータ作成
    num_atoms = 5
    atom_types = torch.randint(1, 6, (num_atoms,)).to(device)  # H, C, N, O, F
    positions = torch.randn(num_atoms, 3).to(device) * 2.0
    
    # エッジ作成（完全グラフ）
    edges = []
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                edges.append([i, j])
    edge_index = torch.tensor(edges, dtype=torch.long).t().to(device)
    
    # 時間ステップ
    t = torch.tensor([500]).to(device)
    
    # フォワードパス
    try:
        from torch_geometric.data import Data
        atom_logits, pos_pred = model(
            Data(
                x=atom_types,
                pos=positions,
                edge_index=edge_index
            ),
            t
        )
        
        print(f"原子タイプ予測形状: {atom_logits.shape}")
        print(f"位置予測形状: {pos_pred.shape}")
        print("✅ フォワードパス成功")
        
    except Exception as e:
        print(f"❌ フォワードパス失敗: {e}")
        return False
    
    return True

def test_diffusion_scheduler():
    """Diffusion schedulerのテスト"""
    print("\n=== Diffusion Scheduler テスト ===")
    
    scheduler = DiffusionScheduler(
        num_timesteps=100,
        beta_start=1e-4,
        beta_end=0.02
    )
    
    # テストデータ
    x = torch.randn(5, 3)
    t = torch.tensor([50])
    
    # ノイズ追加
    noisy_x, noise = scheduler.add_noise(x, t)
    print(f"元データ形状: {x.shape}")
    print(f"ノイズ付きデータ形状: {noisy_x.shape}")
    print(f"ノイズ形状: {noise.shape}")
    
    # デノイズ
    denoised = scheduler.denoise_step(noisy_x, noise, t)
    print(f"デノイズ後形状: {denoised.shape}")
    print("✅ Diffusion Scheduler テスト成功")
    
    return True

def test_molecular_sampler():
    """Molecular samplerのテスト"""
    print("\n=== Molecular Sampler テスト ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")
    
    # モデルとスケジューラー
    model = SimpleE3DiffusionModel(
        num_atom_types=119,
        hidden_dim=32,  # テスト用に小さく
        num_layers=2,   # テスト用に小さく
        time_embed_dim=32,
    )
    
    scheduler = DiffusionScheduler(num_timesteps=50)  # テスト用に少なく
    
    sampler = MolecularDiffusionSampler(
        model=model,
        scheduler=scheduler,
        device=device
    )
    
    try:
        # 分子生成
        print("分子生成を開始...")
        molecules = sampler.sample_molecules(num_molecules=2, max_atoms=8)
        
        print(f"生成された分子数: {len(molecules)}")
        for i, (atom_types, positions) in enumerate(molecules):
            print(f"分子 {i+1}:")
            print(f"  原子数: {len(atom_types)}")
            print(f"  原子タイプ: {atom_types}")
            print(f"  座標形状: {positions.shape}")
        
        print("✅ Molecular Sampler テスト成功")
        return True
        
    except Exception as e:
        print(f"❌ Molecular Sampler テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_diffusion_sampler():
    """DiffusionSamplerクラスのテスト"""
    print("\n=== DiffusionSampler テスト ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # DiffusionSampler初期化
        sampler = DiffusionSampler(device=device)
        
        # SMILES生成
        smiles_list = sampler.sample_smiles(n=3)
        print(f"生成されたSMILES: {smiles_list}")
        
        # ストリーミング生成
        print("ストリーミング生成:")
        for i, smiles in enumerate(sampler.sample_smiles_stream(n=2)):
            print(f"  {i+1}: {smiles}")
        
        # グラフ生成
        atoms, bonds, pos = sampler.sample_graph_with_coords()
        print(f"グラフ生成:")
        print(f"  原子: {atoms}")
        print(f"  結合数: {len(bonds)}")
        print(f"  座標形状: {pos.shape}")
        
        print("✅ DiffusionSampler テスト成功")
        return True
        
    except Exception as e:
        print(f"❌ DiffusionSampler テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_smiles_conversion():
    """SMILES変換のテスト"""
    print("\n=== SMILES変換テスト ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sampler = DiffusionSampler(device=device)
    
    # テスト用の原子と座標
    atom_types = [6, 6, 8]  # C, C, O
    positions = np.array([
        [0.0, 0.0, 0.0],   # C1
        [1.5, 0.0, 0.0],   # C2
        [0.0, 1.5, 0.0],   # O
    ])
    
    try:
        smiles = sampler._atoms_to_smiles(atom_types, positions)
        print(f"原子タイプ: {atom_types}")
        print(f"座標: {positions}")
        print(f"生成されたSMILES: {smiles}")
        
        if smiles:
            # SMILESの妥当性チェック
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                print(f"✅ 有効なSMILES: {smiles}")
                # 分子式の計算（RDKitの正しいAPIを使用）
                try:
                    from rdkit.Chem import rdMolDescriptors
                    formula = rdMolDescriptors.CalcMolFormula(mol)
                    print(f"分子式: {formula}")
                except:
                    print("分子式計算は利用できません")
                return True
            else:
                print(f"❌ 無効なSMILES: {smiles}")
                return False
        else:
            print("❌ SMILES生成失敗")
            return False
            
    except Exception as e:
        print(f"❌ SMILES変換テスト失敗: {e}")
        return False

def main():
    """メインテスト関数"""
    print("E(3)-equivariant diffusion model テスト開始")
    print("=" * 50)
    
    tests = [
        test_e3_model,
        test_diffusion_scheduler,
        test_molecular_sampler,
        test_diffusion_sampler,
        test_smiles_conversion,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ テスト実行中にエラー: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("テスト結果サマリー:")
    passed = sum(results)
    total = len(results)
    print(f"成功: {passed}/{total}")
    
    if passed == total:
        print("🎉 すべてのテストが成功しました！")
    else:
        print("⚠️  一部のテストが失敗しました")
    
    return passed == total

if __name__ == "__main__":
    main()
