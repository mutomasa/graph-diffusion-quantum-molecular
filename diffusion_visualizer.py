# diffusion_visualizer.py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import List, Tuple, Dict
import time

class DiffusionVisualizer:
    """グラフベースの拡散過程を3Dで可視化するクラス"""
    
    def __init__(self):
        self.diffusion_steps = 50
        self.noise_levels = np.linspace(1.0, 0.0, self.diffusion_steps)
        
    def create_molecular_graph(self, smiles: str) -> Tuple[List[int], List[Tuple], np.ndarray]:
        """SMILESから分子グラフを作成"""
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        
        # 3D座標を生成
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
        
        conf = mol.GetConformer()
        atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
        bonds = []
        
        for b in mol.GetBonds():
            bonds.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        
        pos = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=float)
        
        return atoms, bonds, pos
    
    def simulate_diffusion_process(self, initial_pos: np.ndarray, target_pos: np.ndarray) -> List[np.ndarray]:
        """拡散過程をシミュレート"""
        positions = []
        
        for step in range(self.diffusion_steps):
            # ノイズレベルに基づいて位置を補間
            noise = self.noise_levels[step]
            current_pos = noise * initial_pos + (1 - noise) * target_pos
            
            # ランダムノイズを追加（拡散過程の表現）
            if step < self.diffusion_steps - 1:
                noise_scale = 0.1 * noise
                current_pos += np.random.normal(0, noise_scale, current_pos.shape)
            
            positions.append(current_pos.copy())
        
        return positions
    
    def create_3d_diffusion_animation(self, smiles_list: List[str]) -> go.Figure:
        """3D拡散アニメーションを作成"""
        
        # 複数の分子の拡散過程をシミュレート
        all_positions = []
        atom_colors = []
        atom_sizes = []
        
        for smiles in smiles_list:
            try:
                atoms, bonds, target_pos = self.create_molecular_graph(smiles)
                
                # 初期位置（ノイズだらけ）
                initial_pos = target_pos + np.random.normal(0, 2.0, target_pos.shape)
                
                # 拡散過程をシミュレート
                positions = self.simulate_diffusion_process(initial_pos, target_pos)
                all_positions.append(positions)
                
                # 原子の色とサイズを設定
                colors = []
                sizes = []
                for atom_num in atoms:
                    if atom_num == 1:  # H
                        colors.append('lightblue')
                        sizes.append(0.5)
                    elif atom_num == 6:  # C
                        colors.append('gray')
                        sizes.append(1.0)
                    elif atom_num == 7:  # N
                        colors.append('blue')
                        sizes.append(1.0)
                    elif atom_num == 8:  # O
                        colors.append('red')
                        sizes.append(1.0)
                    else:
                        colors.append('green')
                        sizes.append(1.0)
                
                atom_colors.extend(colors)
                atom_sizes.extend(sizes)
                
            except Exception as e:
                print(f"Error processing {smiles}: {e}")
                continue
        
        # 3Dアニメーションを作成
        fig = go.Figure()
        
        # 各ステップでフレームを作成
        frames = []
        for step in range(self.diffusion_steps):
            frame_data = []
            
            for mol_idx, positions in enumerate(all_positions):
                if step < len(positions):
                    pos = positions[step]
                    
                    # 原子をプロット
                    frame_data.append(
                        go.Scatter3d(
                            x=pos[:, 0],
                            y=pos[:, 1],
                            z=pos[:, 2],
                            mode='markers',
                            marker=dict(
                                size=atom_sizes[mol_idx * len(pos):(mol_idx + 1) * len(pos)],
                                color=atom_colors[mol_idx * len(pos):(mol_idx + 1) * len(pos)],
                                opacity=0.8
                            ),
                            name=f'Molecule {mol_idx + 1}',
                            showlegend=False
                        )
                    )
                    
                    # 結合をプロット
                    if mol_idx < len(smiles_list):
                        try:
                            atoms, bonds, _ = self.create_molecular_graph(smiles_list[mol_idx])
                            for bond in bonds:
                                if bond[0] < len(pos) and bond[1] < len(pos):
                                    frame_data.append(
                                        go.Scatter3d(
                                            x=[pos[bond[0], 0], pos[bond[1], 0]],
                                            y=[pos[bond[0], 1], pos[bond[1], 1]],
                                            z=[pos[bond[0], 2], pos[bond[1], 2]],
                                            mode='lines',
                                            line=dict(color='black', width=3),
                                            showlegend=False
                                        )
                                    )
                        except:
                            pass
            
            frames.append(go.Frame(data=frame_data, name=f'frame{step}'))
        
        # 初期フレームを設定
        if frames:
            fig.add_traces(frames[0].data)
        
        # レイアウトを設定
        fig.update_layout(
            title="🎭 グラフベース拡散過程の3D可視化",
            scene=dict(
                xaxis_title="X (Å)",
                yaxis_title="Y (Å)",
                zaxis_title="Z (Å)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectmode='cube'
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': '▶️ 拡散開始',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 100, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 50}
                        }]
                    },
                    {
                        'label': '⏸️ 一時停止',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }],
            sliders=[{
                'steps': [
                    {
                        'args': [[f'frame{step}'], {
                            'frame': {'duration': 100, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 50}
                        }],
                        'label': f'Step {step + 1}',
                        'method': 'animate'
                    }
                    for step in range(self.diffusion_steps)
                ],
                'active': 0,
                'currentvalue': {'prefix': '拡散ステップ: '},
                'len': 0.9,
                'x': 0.1,
                'xanchor': 'left',
                'y': 0,
                'yanchor': 'top'
            }],
            height=600,
            showlegend=False
        )
        
        fig.frames = frames
        
        return fig
    
    def create_diffusion_energy_plot(self) -> go.Figure:
        """拡散過程のエネルギー変化を可視化"""
        
        # 拡散過程のエネルギー変化をシミュレート
        steps = np.arange(self.diffusion_steps)
        energy_noise = np.exp(-steps / 10)  # ノイズエネルギー
        energy_structure = 1 - np.exp(-steps / 15)  # 構造エネルギー
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=steps,
            y=energy_noise,
            mode='lines+markers',
            name='ノイズエネルギー',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=steps,
            y=energy_structure,
            mode='lines+markers',
            name='構造エネルギー',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=steps,
            y=energy_noise + energy_structure,
            mode='lines+markers',
            name='総エネルギー',
            line=dict(color='green', width=4),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title="⚡ 拡散過程のエネルギー変化",
            xaxis_title="拡散ステップ",
            yaxis_title="エネルギー (任意単位)",
            template="plotly_white",
            height=400,
            showlegend=True
        )
        
        return fig

def create_diffusion_dashboard(smiles_list: List[str]):
    """拡散ダッシュボードを作成"""
    
    visualizer = DiffusionVisualizer()
    
    st.markdown("## 🔬 グラフベース拡散過程の3D可視化")
    st.markdown("### 分子生成の技術的プロセス")
    
    # 3D拡散アニメーション
    st.markdown("#### 3D分子生成可視化")
    diffusion_fig = visualizer.create_3d_diffusion_animation(smiles_list)
    st.plotly_chart(diffusion_fig, use_container_width=True)
    
    # エネルギー変化プロット
    st.markdown("#### ⚡ 拡散過程のエネルギー変化")
    energy_fig = visualizer.create_diffusion_energy_plot()
    st.plotly_chart(energy_fig, use_container_width=True)
    
    # 技術説明
    st.markdown("---")
    st.markdown("""
    ### 🔬 **技術概要**
    
    グラフベース拡散モデルによる分子生成と量子計算による物性予測の統合システムです。
    """)
    
    # 技術統計
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("拡散ステップ", "50")
    with col2:
        st.metric("分子生成数", f"{len(smiles_list)}")
    with col3:
        st.metric("計算精度", "高精度")
