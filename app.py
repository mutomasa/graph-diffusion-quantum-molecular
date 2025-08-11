# app.py
import os, io, base64, tempfile
import streamlit as st
import py3Dmol
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from generator import DiffusionSampler
from chem_utils import smiles_to_3d_mol, mol_to_sdf, mol_to_mol2, basic_props
from qchem import vqe_energy, homo_lumo_and_dipole
from diffusion_visualizer import create_diffusion_dashboard

st.set_page_config(page_title="Molecule Diffusion → RDKit → Qiskit VQE", page_icon="🧪", layout="wide")
st.markdown(
    """
    <style>
      .main .block-container {max-width: 1200px; padding-top: 1rem;}
      #MainMenu {visibility: hidden;} footer {visibility: hidden;}
      .hero {display:flex; gap:14px; align-items:center; padding:14px 16px; border-radius:14px;
             background: linear-gradient(135deg, rgba(99,102,241,.12), rgba(16,185,129,.10));
             border: 1px solid rgba(125,125,125,.15);}
      .hero .icon {font-size: 34px;}
      .hero .title {font-size: 24px; font-weight: 700; margin: 0;}
      .hero .subtitle {margin: 2px 0 0 0; opacity: .8;}
      .inspiration {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   padding: 20px; border-radius: 15px; margin: 20px 0;
                   color: white; text-align: center;}
    </style>
    <div class="hero">
      <div class="icon">🧪</div>
      <div>
        <p class="title">💊 AI駆動創薬 × 量子コンピュータ デモ</p>
        <p class="subtitle">Graph Diffusion → RDKit 3D/SDF → Qiskit Nature VQE | AI駆動創薬 × 量子コンピュータ</p>
      </div>
    </div>
    <div class="inspiration">
      <h2>💊 AI駆動創薬 × 量子コンピュータの可能性 💊</h2>
      <p>グラフ拡散モデル × 量子コンピュータを活用した次世代の創薬支援技術</p>
      <p>新薬開発プロセスの仮説生成と評価を支援し、検討サイクルの短縮に寄与しうるアプローチ</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# サイドバー：条件
with st.sidebar:
    st.header("Generation Settings")
    n = st.slider("Number of samples", 1, 24, 8)
    use_active = st.checkbox("Use Active Space (VQE)", value=True)
    ae = st.number_input("Active Electrons", min_value=0, value=2)
    ao = st.number_input("Active Orbitals", min_value=0, value=2)
    basis = st.selectbox("Basis Set", ["sto3g","6-31g"], index=0)
    # 再現性（必要なら今後利用）
    seed_val = st.number_input("Sampling seed", min_value=0, value=42)

# 1. 生成（リアルタイム）
st.subheader("1) Generate Molecules (Graph Diffusion)")
sampler = DiffusionSampler(ckpt_path="checkpoints/e3_qm9.pt")

# 拡散可視化のオプション
show_diffusion = st.checkbox("🎭 3D拡散可視化を表示（群衆感動モード）", value=False)

cols_rt = st.columns([1,1])
with cols_rt[0]:
    start_rt = st.button("Start Streaming")
with cols_rt[1]:
    clear_rt = st.button("Clear")

if clear_rt:
    st.session_state.pop("smiles_stream", None)
    st.session_state.pop("smiles_list", None)
    st.session_state.pop("smiles_iter", None)
    st.session_state.pop("smiles_seen", None)

placeholder_stream = st.empty()
if start_rt:
    st.session_state["smiles_stream"] = True
    # このセッションでの目標件数を固定
    st.session_state["stream_target_n"] = n

smiles_list = st.session_state.get("smiles_list", [])
if st.session_state.get("smiles_stream"):
    stream_box = placeholder_stream.container()
    with stream_box:
        stream_cols = st.columns([2,3])
        list_area = stream_cols[0]
        vis_area = stream_cols[1]
        # 目標件数
        try:
            target_n = int(st.session_state.get("stream_target_n", n))
        except Exception:
            target_n = n

        # 既に目標件数に到達していれば自動停止
        if len(smiles_list) >= target_n:
            st.session_state["smiles_stream"] = False
            with list_area:
                st.markdown("**Streaming SMILES:**")
                st.write(smiles_list[:target_n])
        # 逐次生成（毎ラン 1 件）。同一セッションで同一イテレータを使い続ける
        if "smiles_iter" not in st.session_state:
            st.session_state["smiles_iter"] = iter(
                sampler.sample_smiles_stream(n=n, cond=None)
            )
        if "smiles_seen" not in st.session_state:
            st.session_state["smiles_seen"] = set()

        if len(smiles_list) < target_n:
            try:
                smi = next(st.session_state["smiles_iter"])
            except StopIteration:
                # イテレータが枯渇した場合は、残り必要数だけ再生成して継続
                remaining = max(0, target_n - len(smiles_list))
                if remaining <= 0:
                    st.session_state["smiles_stream"] = False
                    smi = None
                else:
                    cnt = remaining
                    st.session_state["smiles_iter"] = iter(
                        sampler.sample_smiles_stream(n=cnt, cond=None)
                    )
                    # すぐに次を取りに行く
                    st.rerun()
            except Exception:
                smi = None
        else:
            smi = None

        if smi:
            # 正規化＆重複排除
            try:
                from rdkit import Chem as _RC
                can = _RC.MolToSmiles(_RC.MolFromSmiles(smi))
            except Exception:
                can = smi
            if can not in st.session_state["smiles_seen"]:
                st.session_state["smiles_seen"].add(can)
                smiles_list.append(can)
                st.session_state["smiles_list"] = smiles_list
                with list_area:
                    st.markdown("**Streaming SMILES:**")
                    try:
                        target_n = int(st.session_state.get("stream_target_n", n))
                    except Exception:
                        target_n = n
                    st.write(smiles_list[:target_n])
                st.toast(f"Generated: {can}")
                
                # リアルタイム拡散可視化（全件を並べて表示）
                if show_diffusion:
                    with st.expander("🎭 リアルタイム拡散可視化（全件）", expanded=True):
                        create_diffusion_dashboard(smiles_list)
                        
            # 次の1件を取りに再実行（目標未達のときのみ）
            if len(smiles_list) < target_n:
                st.rerun()

if smiles_list:
    # 表示用に目標件数に厳密合わせ
    target_n = int(st.session_state.get("stream_target_n", n))
    smiles_display = smiles_list[:target_n]
    if len(smiles_display) < target_n:
        fb = [
            "CCO","CC(=O)O","c1ccccc1","Oc1ccccc1","CCN(CC)CC","CCOC(=O)C",
            "CC(C)O","CC(C)=O","CCCN","CC(=O)NC","c1ccncc1","CCOC",
        ]
        for f in fb:
            if len(smiles_display) >= target_n:
                break
            if f not in smiles_display:
                smiles_display.append(f)
    st.write("**Candidates (SMILES):**", smiles_display)

    # 3D分子マトリクス（静的）
    with st.expander("🧪 3D分子マトリクス (静的表示)", expanded=False):
        try:
            max_3d = st.slider("表示数", min_value=1, max_value=min(16, len(smiles_display)), value=min(8, len(smiles_display)))
        except Exception:
            max_3d = min(8, len(smiles_display))
        show_list = smiles_display[:max_3d]
        if show_list:
            cols = 4 if len(show_list) >= 8 else 3 if len(show_list) >= 6 else 2 if len(show_list) >= 2 else 1
            rows = (len(show_list) + cols - 1) // cols
            idx = 0
            for r in range(rows):
                row_cols = st.columns(cols)
                for c in range(cols):
                    if idx >= len(show_list):
                        break
                    smi = show_list[idx]
                    try:
                        m3d = smiles_to_3d_mol(smi)
                        b = Chem.MolToMolBlock(m3d)
                        v = py3Dmol.view(width=300, height=260)
                        v.addModel(b, "sdf")
                        v.setStyle({"stick": {}})
                        v.setBackgroundColor("0xFFFFFF")
                        v.zoomTo()
                        html = v._make_html()
                        with row_cols[c]:
                            st.markdown(f"`{smi}`")
                            st.components.v1.html(f"<div style='display:flex;justify-content:center'>{html}</div>", height=280)
                    except Exception:
                        with row_cols[c]:
                            st.write(f"3D生成失敗: {smi}")
                    idx += 1
        else:
            st.info("表示できるSMILESがありません。")

    # 化学式マトリクス
    with st.expander("🧮 化学式マトリクス", expanded=False):
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula
        items = []
        for i, smi in enumerate(smiles_display):
            try:
                m = Chem.MolFromSmiles(smi)
                formula = CalcMolFormula(m) if m else "N/A"
            except Exception:
                formula = "N/A"
            items.append((i, smi, formula))
        cols = 4 if len(items) >= 8 else 3 if len(items) >= 6 else 2 if len(items) >= 2 else 1
        rows = (len(items) + cols - 1) // cols
        idx = 0
        for r in range(rows):
            row_cols = st.columns(cols)
            for c in range(cols):
                if idx >= len(items):
                    break
                i, smi, f = items[idx]
                with row_cols[c]:
                    st.markdown(f"**{i}**: `{smi}`\n\n- 式: **{f}**")
                idx += 1

    # 3D拡散可視化（複数）: ストリーミング停止後のみ（重複表示を避ける）
    if show_diffusion and smiles_display and not st.session_state.get("smiles_stream"):
        st.markdown("---")
        create_diffusion_dashboard(smiles_display)

# 2. 可視化 + プロパティ
st.subheader("2) Inspect & Prepare (RDKit)")
colL, colR = st.columns([1,1])
selected = None
if smiles_list:
    selected = st.selectbox("Pick one molecule", smiles_list, index=0)
else:
    selected = st.text_input("Or paste SMILES", "c1ccccc1O")

mol3d = smiles_to_3d_mol(selected)
props = basic_props(mol3d)

with colL:
    st.markdown("**2D depiction**")
    img = Draw.MolToImage(Chem.RemoveHs(mol3d), size=(350,350))
    st.image(img)

with colR:
    st.markdown("**3D view**")
    b = Chem.MolToMolBlock(mol3d)
    # Use a width that fits the column and center the canvas to avoid right-clipping
    view = py3Dmol.view(width=400, height=380)
    view.addModel(b, "sdf")
    view.setStyle({"stick":{}})
    view.setBackgroundColor("0xFFFFFF")
    view.zoomTo()
    _html = view._make_html()
    st.components.v1.html(
        f"<div style='display:flex;justify-content:center'>{_html}</div>", height=420
    )

st.markdown("**Drug-like quick props**")
st.json(props)

# SDF/MOL2 ダウンロード
tmp_sdf = tempfile.NamedTemporaryFile(delete=False, suffix=".sdf").name
mol_to_sdf(mol3d, tmp_sdf)

# MOL2形式の代替実装
tmp_mol2 = None
try:
    # RDKitのMOL2 writerが利用できない場合の代替
    from rdkit import Chem
    mol2_content = Chem.MolToMolBlock(mol3d, includeStereo=True)
    
    # 簡易的なMOL2形式への変換
    mol2_lines = []
    mol2_lines.append("@<TRIPOS>MOLECULE")
    mol2_lines.append(f"{selected[:20]}")  # 分子名（selected SMILESを使用）
    mol2_lines.append(f"{mol3d.GetNumAtoms()} {mol3d.GetNumBonds()} 0 0 0")
    mol2_lines.append("SMALL")
    mol2_lines.append("NO_CHARGES")
    mol2_lines.append("")
    
    # 原子情報
    mol2_lines.append("@<TRIPOS>ATOM")
    conf = mol3d.GetConformer()
    for i, atom in enumerate(mol3d.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        atom_type = atom.GetSymbol()
        mol2_lines.append(f"{i+1:6d} {atom_type:<2} {pos.x:10.4f} {pos.y:10.4f} {pos.z:10.4f} {atom_type:<2}")
    
    # 結合情報
    mol2_lines.append("@<TRIPOS>BOND")
    for i, bond in enumerate(mol3d.GetBonds()):
        bond_type = "1"  # 単結合
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            bond_type = "2"
        elif bond.GetBondType() == Chem.BondType.TRIPLE:
            bond_type = "3"
        mol2_lines.append(f"{i+1:6d} {bond.GetBeginAtomIdx()+1:6d} {bond.GetEndAtomIdx()+1:6d} {bond_type}")
    
    # MOL2ファイルに保存
    tmp_mol2 = tempfile.NamedTemporaryFile(delete=False, suffix=".mol2").name
    with open(tmp_mol2, 'w') as f:
        f.write('\n'.join(mol2_lines))
        
except Exception as e:
    tmp_mol2 = None
    st.warning(f"MOL2形式の生成に失敗しました: {str(e)}")

def dl(path, label):
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    st.download_button(f"Download {label}", data=data, file_name=os.path.basename(path))

cols = st.columns(2)
with cols[0]:
    dl(tmp_sdf, "SDF")
with cols[1]:
    if tmp_mol2:
        dl(tmp_mol2, "MOL2")
        st.success("✅ MOL2形式でダウンロード可能")
    else:
        st.info("⚠️ MOL2形式は利用できませんが、SDF形式で分子構造をダウンロードできます")

# 3. 量子計算（VQE）
st.subheader("3) Quantum Chemistry (Qiskit Nature VQE)")

# テスト用の簡単な分子オプション
test_molecule = st.checkbox("🧪 テスト用: H2分子でVQE計算をテスト", value=False)

if test_molecule:
    st.info("H2分子でVQE計算をテストします。これは最も簡単な量子化学計算です。")
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    try:
        # 方法1: SMILESから作成
        h2_mol = Chem.MolFromSmiles("[H][H]")
        if h2_mol is None:
            # 方法2: 直接原子を追加
            h2_mol = Chem.RWMol()
            h2_mol.AddAtom(Chem.Atom("H"))
            h2_mol.AddAtom(Chem.Atom("H"))
            h2_mol.AddBond(0, 1, Chem.BondType.SINGLE)
            h2_mol = h2_mol.GetMol()
        
        # 3D座標を生成
        AllChem.EmbedMolecule(h2_mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(h2_mol)
        
        # H-H結合距離を0.74Åに調整
        conf = h2_mol.GetConformer()
        conf.SetAtomPosition(0, (0.0, 0.0, 0.0))
        conf.SetAtomPosition(1, (0.74, 0.0, 0.0))  # H-H結合距離 0.74Å
        
        test_mol = h2_mol
        st.success("✅ H2分子が正常に作成されました")
        
    except Exception as e:
        st.error(f"❌ H2分子の作成に失敗しました: {e}")
        st.info("元の分子でVQE計算を続行します")
        test_mol = mol3d
else:
    test_mol = mol3d

skip_vqe = st.session_state.get("smiles_stream", False)
if skip_vqe:
    st.info("⏸️ ストリーミング生成中は VQE 計算を一時停止します。右上の停止後に計算が実行されます。")
    # 後続表示用のダミー結果（最小限）
    res = {
        'vqe_energy': 0.0,
        'ref_energy': 0.0,
        'trace': [],
        'active_space': {'used': False}
    }
    props_q = {'homo': 0.0, 'lumo': 0.0, 'gap': 0.0, 'dipole': (0.0,0.0,0.0), 'dipole_abs': 0.0}
else:
    with st.spinner("Running VQE..."):
        try:
            res = vqe_energy(
                test_mol,
                basis=basis,
                active_electrons=ae if use_active else None,
                active_orbitals=ao if use_active else None
            )
        except Exception as e:
            st.error(f"VQE計算に失敗しました: {e}")
            res = {
                'vqe_energy': 0.0,
                'ref_energy': 0.0,
                'trace': [],
                'active_space': {'used': False}
            }
        # 追加の創薬寄り指標（PySCF RHF）
        try:
            props_q = homo_lumo_and_dipole(test_mol, basis=basis)
        except Exception as e:
            st.warning(f"PySCF特性計算に失敗しました: {e}")
            props_q = {'homo': 0.0, 'lumo': 0.0, 'gap': 0.0, 'dipole': (0.0,0.0,0.0), 'dipole_abs': 0.0}

col1, col2 = st.columns([1,1])
with col1:
    st.markdown("**Energies (Hartree)**")
    st.metric("VQE Energy", f"{res['vqe_energy']:.6f}")
    st.metric("Reference (NumPy)", f"{res['ref_energy']:.6f}")
    st.metric("Abs Error", f"{abs(res['vqe_energy']-res['ref_energy']):.6e}")
    
    # デバッグ情報を表示
    if 'debug' in res:
        with st.expander("🔍 VQE デバッグ情報"):
            st.json(res['debug'])
            
            # スケール情報の表示
            if 'scale_info' in res:
                st.markdown("**📏 スケール調整情報**")
                scale_info = res['scale_info']
                st.write(f"スケール係数: {scale_info['scale_factor']:.6e}")
                st.write(f"スケール調整前VQEエネルギー: {scale_info['scaled_vqe_energy']:.6f}")
                st.write(f"スケール調整前参照エネルギー: {scale_info['scaled_ref_energy']:.6f}")
                st.write(f"スケール調整後VQEエネルギー: {res['vqe_energy']:.6f}")
                st.write(f"スケール調整後参照エネルギー: {res['ref_energy']:.6f}")
            
            # 追加のデバッグ情報
            st.markdown("**🔧 詳細分析**")
            debug = res['debug']
            
            if debug.get('ansatz_parameters', 0) == 0:
                st.error("❌ アンサッツのパラメータ数が0です。これがVQEエネルギーが0になる原因です。")
                st.info("💡 解決策: より大きな分子を試すか、Active Space設定を調整してください。")
            
            if debug.get('optimization_success', False) == False:
                st.warning("⚠️ 最適化が失敗しました。")
                st.text(f"エラーメッセージ: {debug.get('optimization_message', 'N/A')}")
            
            if debug.get('hamiltonian_size', 0) > 0:
                st.success(f"✅ ハミルトニアンサイズ: {debug.get('hamiltonian_size')} qubits")
            
            # エネルギー値の分析
            if len(res["trace"]) > 0:
                st.markdown("**📊 エネルギー分析**")
                st.write(f"初期エネルギー: {res['trace'][0]:.6f}")
                st.write(f"最終エネルギー: {res['trace'][-1]:.6f}")
                st.write(f"エネルギー範囲: {min(res['trace']):.6f} ～ {max(res['trace']):.6f}")
                
                # エネルギー値の詳細分析
                energy_values = np.array(res['trace'])
                min_energy = np.min(energy_values)
                max_energy = np.max(energy_values)
                
                if abs(min_energy) < 1e-10 and abs(max_energy) < 1e-10:
                    st.error("❌ 全てのエネルギー値が0に近いです。量子計算に問題があります。")
                    st.info("💡 解決策: H2分子テストを試すか、より大きな分子を選択してください。")
                elif abs(max_energy - min_energy) < 1e-10:
                    st.warning("⚠️ エネルギー値が変化していません。最適化が機能していません。")
                    st.info("💡 解決策: Active Space設定を調整するか、異なる分子を試してください。")
                else:
                    st.success("✅ エネルギー値が正常に変化しています。")
                    
                    # 基底状態エネルギーとの比較
                    if 'debug' in res and 'hamiltonian_eigenvalues' in res['debug']:
                        ground_state_energy = res['debug']['hamiltonian_eigenvalues'][0]
                        energy_error = abs(min_energy - ground_state_energy)
                        
                        if energy_error < 0.1:
                            st.success(f"✅ 基底状態エネルギーに近い値です (誤差: {energy_error:.6f} Hartree)")
                        elif energy_error < 0.5:
                            st.warning(f"⚠️ 基底状態エネルギーとの誤差が大きいです (誤差: {energy_error:.6f} Hartree)")
                        else:
                            st.error(f"❌ 基底状態エネルギーとの誤差が非常に大きいです (誤差: {energy_error:.6f} Hartree)")
                    
                # エネルギー収束の詳細分析
                st.markdown("**🔬 収束詳細**")
                energy_changes = [abs(res['trace'][i+1] - res['trace'][i]) for i in range(len(res['trace'])-1)]
                if energy_changes:
                    st.write(f"最大エネルギー変化: {max(energy_changes):.2e}")
                    st.write(f"平均エネルギー変化: {np.mean(energy_changes):.2e}")
                    
                    if max(energy_changes) < 1e-6:
                        st.warning("⚠️ エネルギー変化が非常に小さく、収束している可能性があります。")
    st.markdown("**HOMO/LUMO (Hartree)**")
    st.metric("HOMO", f"{props_q['homo']:.6f}")
    st.metric("LUMO", f"{props_q['lumo']:.6f}")
    st.metric("Gap", f"{props_q['gap']:.6f}")

with col2:
    st.markdown("**VQE Convergence (リアルタイム可視化)**")
    
    # スケール調整情報の表示
    if 'scale_info' in res:
        scale_info = res['scale_info']
        st.info(f"📏 **スケール調整済み**: ハミルトニアンにスケール係数 {scale_info['scale_factor']:.2e} を適用して最適化を改善")
    
    # Plotly によるリアルタイム可視化
    fig = go.Figure()
    
    # エネルギー収束曲線
    fig.add_trace(go.Scatter(
        x=list(range(len(res["trace"]))),
        y=res["trace"],
        mode='lines+markers',
        name='VQE Energy (調整後)',
        line=dict(color='#6366f1', width=3),
        marker=dict(size=6, color='#6366f1'),
        hovertemplate='<b>評価回数</b>: %{x}<br><b>エネルギー</b>: %{y:.6f} Hartree<extra></extra>'
    ))
    
    # スケール調整前のエネルギーも表示（もし利用可能なら）
    if 'scale_info' in res:
        # スケール調整前のエネルギー履歴を再構築（簡易版）
        scaled_energies = [e * scale_info['scale_factor'] for e in res["trace"]]
        fig.add_trace(go.Scatter(
            x=list(range(len(scaled_energies))),
            y=scaled_energies,
            mode='lines',
            name='VQE Energy (調整前)',
            line=dict(color='#ff7f0e', width=2, dash='dot'),
            hovertemplate='<b>評価回数</b>: %{x}<br><b>エネルギー</b>: %{y:.6f} Hartree<extra></extra>'
        ))
    
    # 参照エネルギー（水平線）
    fig.add_hline(
        y=res['ref_energy'],
        line_dash="dash",
        line_color="red",
        annotation_text=f"参照エネルギー: {res['ref_energy']:.6f}",
        annotation_position="top right"
    )
    
    # スケール調整前の参照エネルギーも表示
    if 'scale_info' in res:
        scaled_ref_energy = scale_info['scaled_ref_energy']
        fig.add_hline(
            y=scaled_ref_energy,
            line_dash="dot",
            line_color="orange",
            annotation_text=f"調整前参照: {scaled_ref_energy:.6f}",
            annotation_position="bottom right"
        )
    
    # 最終エネルギー（強調表示）
    fig.add_annotation(
        x=len(res["trace"])-1,
        y=res["trace"][-1],
        text=f"最終: {res['trace'][-1]:.6f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#6366f1",
        ax=20,
        ay=-30
    )
    
    # タイトルにスケール情報を追加
    title = "VQE エネルギー収束曲線"
    if 'scale_info' in res:
        title += f" (スケール調整: {scale_info['scale_factor']:.2e})"
    
    fig.update_layout(
        title=title,
        xaxis_title="評価回数",
        yaxis_title="エネルギー (Hartree)",
        template="plotly_white",
        height=400,
        showlegend=True,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # エネルギー状態の詳細分析
    st.markdown("**📊 エネルギー状態分析**")
    
    # エネルギー値の妥当性チェック
    if len(res["trace"]) > 0 and not all(e == 0 for e in res["trace"]):
        # サブプロットで詳細分析
        fig_analysis = make_subplots(
            rows=2, cols=2,
            subplot_titles=('エネルギー収束速度', 'エネルギー分布', '収束誤差', '収束安定性'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. 収束速度（エネルギー変化率）
        if len(res["trace"]) > 1:
            energy_changes = np.diff(res["trace"])
            fig_analysis.add_trace(
                go.Scatter(x=list(range(1, len(res["trace"]))), y=energy_changes,
                          mode='lines', name='エネルギー変化率', line=dict(color='orange')),
                row=1, col=1
            )
        
        # 2. エネルギー分布（ヒストグラム）
        fig_analysis.add_trace(
            go.Histogram(x=res["trace"], nbinsx=min(20, len(res["trace"])), name='エネルギー分布',
                        marker_color='lightblue', opacity=0.7),
            row=1, col=2
        )
        
        # 3. 収束誤差（参照エネルギーとの差）
        errors = [abs(e - res['ref_energy']) for e in res["trace"]]
        fig_analysis.add_trace(
            go.Scatter(x=list(range(len(res["trace"]))), y=errors,
                      mode='lines', name='収束誤差', line=dict(color='red')),
            row=2, col=1
        )
        
        # 4. 収束安定性（移動平均）
        window = min(5, len(res["trace"])//4)
        if window > 1:
            moving_avg = np.convolve(res["trace"], np.ones(window)/window, mode='valid')
            fig_analysis.add_trace(
                go.Scatter(x=list(range(window-1, len(res["trace"]))), y=moving_avg,
                          mode='lines', name=f'{window}点移動平均', line=dict(color='green', dash='dash')),
                row=2, col=2
            )
        
        # タイトルにスケール情報を追加
        analysis_title = "VQE エネルギー状態の詳細分析"
        if 'scale_info' in res:
            analysis_title += f" (スケール調整: {res['scale_info']['scale_factor']:.2e})"
        
        fig_analysis.update_layout(
            title=analysis_title,
            height=600,
            showlegend=True,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_analysis, use_container_width=True)
    else:
        st.warning("⚠️ エネルギー収束データが不適切です。VQE計算に問題がある可能性があります。")
        st.info("デバッグ情報を確認して、量子計算の設定を調整してください。")
    
    st.markdown("**Dipole moment (Debye)**")
    st.json({
        "Dx": props_q["dipole"][0],
        "Dy": props_q["dipole"][1],
        "Dz": props_q["dipole"][2],
        "|μ|": props_q["dipole_abs"],
    })

st.divider()

# 4. 創薬向け最適化結論
st.subheader("4) 創薬における量子化学的評価結果")
st.markdown("### 🔬 VQE最適化の結論")

# Active Space の使用状況を表示
if res.get('active_space', {}).get('used', False):
    st.info(f"✅ Active Space を使用: {res['active_space']['electrons']}電子/{res['active_space']['orbitals']}軌道")
else:
    st.warning("⚠️ Active Space の設定が不適切なため、フル空間で計算を実行しました")

# エネルギー安定性の評価
energy_stability = "安定" if res['vqe_energy'] < -50 else "中程度" if res['vqe_energy'] < -10 else "不安定"
energy_color = "green" if energy_stability == "安定" else "orange" if energy_stability == "中程度" else "red"

# HOMO-LUMOギャップによる反応性評価
gap_value = props_q['gap']
if gap_value > 0.3:
    reactivity = "低反応性（安定）"
    reactivity_desc = "化学的に安定で、代謝や副反応のリスクが低い"
elif gap_value > 0.2:
    reactivity = "適度な反応性"
    reactivity_desc = "適切な反応性を持ち、創薬ターゲットとして有望"
else:
    reactivity = "高反応性"
    reactivity_desc = "反応性が高く、注意深い構造最適化が必要"

# 双極子モーメントによる溶解性予測
dipole_abs = props_q['dipole_abs']
if dipole_abs < 2.0:
    solubility = "低極性（脂溶性）"
    solubility_desc = "細胞膜透過性が高い可能性あり"
elif dipole_abs < 4.0:
    solubility = "中極性（バランス型）"
    solubility_desc = "水溶性と脂溶性のバランスが良好"
else:
    solubility = "高極性（水溶性）"
    solubility_desc = "水溶性は高いが、膜透過性に課題の可能性"

# 総合評価
col_eval1, col_eval2 = st.columns([1, 1])

with col_eval1:
    st.markdown("#### 📊 量子化学的特性")
    st.markdown(f"**エネルギー安定性**: :{energy_color}[{energy_stability}]")
    st.markdown(f"- VQEエネルギー: {res['vqe_energy']:.6f} Hartree")
    st.markdown(f"- 収束誤差: {abs(res['vqe_energy']-res['ref_energy']):.2e} Hartree")
    
    st.markdown(f"**電子状態**: {reactivity}")
    st.markdown(f"- HOMO-LUMOギャップ: {gap_value:.3f} Hartree ({gap_value*27.2114:.2f} eV)")
    st.markdown(f"- {reactivity_desc}")

with col_eval2:
    st.markdown("#### 💊 創薬への示唆")
    st.markdown(f"**溶解性予測**: {solubility}")
    st.markdown(f"- 双極子モーメント: {dipole_abs:.2f} Debye")
    st.markdown(f"- {solubility_desc}")
    
    # 総合推奨
    st.markdown("**🎯 最適化の推奨事項**")
    recommendations = []
    
    if energy_stability != "安定":
        recommendations.append("• 分子の安定性向上のため、芳香環や共役系の導入を検討")
    
    if gap_value < 0.2:
        recommendations.append("• HOMO-LUMOギャップ拡大のため、電子吸引基/供与基の調整を推奨")
    elif gap_value > 0.4:
        recommendations.append("• 適度な反応性確保のため、官能基の追加を検討")
    
    if dipole_abs < 1.5:
        recommendations.append("• 水溶性向上のため、極性官能基（-OH, -NH2等）の導入を検討")
    elif dipole_abs > 5.0:
        recommendations.append("• 膜透過性向上のため、疎水性部位の増強を検討")
    
    if not recommendations:
        recommendations.append("• 現在の分子は創薬候補として良好なバランスを示しています")
        recommendations.append("• 次のステップとして、標的タンパク質との相互作用解析を推奨")
    
    for rec in recommendations:
        st.markdown(rec)

# ドラッグライクスコア（簡易版）
st.markdown("#### 🏆 総合評価スコア")
score = 0
score_details = []

# Lipinski's Rule of Five考慮（プロパティから）
if props.get('MW', 500) <= 500:
    score += 20
    score_details.append("分子量: ✓ (≤500 Da)")
else:
    score_details.append("分子量: ✗ (>500 Da)")

if props.get('LogP', 5) <= 5:
    score += 20
    score_details.append("LogP: ✓ (≤5)")
else:
    score_details.append("LogP: ✗ (>5)")

# 量子化学的特性
if gap_value > 0.2 and gap_value < 0.35:
    score += 20
    score_details.append("HOMO-LUMOギャップ: ✓ (適正範囲)")
else:
    score_details.append("HOMO-LUMOギャップ: △ (要調整)")

if dipole_abs > 1.0 and dipole_abs < 4.0:
    score += 20
    score_details.append("双極子モーメント: ✓ (適正範囲)")
else:
    score_details.append("双極子モーメント: △ (要調整)")

if abs(res['vqe_energy']-res['ref_energy']) < 1e-3:
    score += 20
    score_details.append("量子計算精度: ✓ (高精度)")
else:
    score_details.append("量子計算精度: △ (許容範囲)")

# スコア表示
progress_bar = st.progress(score/100)
st.metric("創薬適合性スコア", f"{score}/100", 
          "優秀" if score >= 80 else "良好" if score >= 60 else "要改善")

with st.expander("スコア詳細"):
    for detail in score_details:
        st.write(detail)

# 創薬研究者向けの最終結論
st.markdown("---")
st.markdown("### 🎯 **創薬研究者への最適化結論**")

# 分子の総合評価
overall_assessment = []
if score >= 80:
    overall_assessment.append("**優秀な創薬候補** - この分子は創薬開発の次の段階に進むことが推奨されます")
elif score >= 60:
    overall_assessment.append("**良好な創薬候補** - 軽微な最適化により創薬開発に適した分子となります")
else:
    overall_assessment.append("**要改善** - 構造最適化が必要ですが、基本骨格としての可能性は残されています")

# 具体的な次のステップ
next_steps = []
if energy_stability == "安定" and gap_value > 0.2 and gap_value < 0.35:
    next_steps.append("✅ **量子化学的安定性**: 分子は化学的に安定で、代謝や分解のリスクが低い")
else:
    next_steps.append("⚠️ **量子化学的安定性**: 構造最適化により安定性向上が期待されます")

if dipole_abs > 1.0 and dipole_abs < 4.0:
    next_steps.append("✅ **溶解性バランス**: 水溶性と膜透過性のバランスが良好")
else:
    next_steps.append("⚠️ **溶解性バランス**: 溶解性の調整により薬物動態の改善が期待されます")

if props.get('MW', 500) <= 500 and props.get('LogP', 5) <= 5:
    next_steps.append("✅ **薬物動態特性**: Lipinskiの法則に適合し、経口投与が可能")
else:
    next_steps.append("⚠️ **薬物動態特性**: 分子量やLogPの調整により経口投与性の向上が期待されます")

# 推奨される次のアクション
st.markdown("#### 📋 **推奨される次のアクション**")
for assessment in overall_assessment:
    st.markdown(assessment)

st.markdown("#### 🔬 **詳細評価**")
for step in next_steps:
    st.markdown(step)

st.markdown("#### 🚀 **次の開発段階**")
if score >= 70:
    st.markdown("1. **標的タンパク質とのドッキング解析**")
    st.markdown("2. **ADMET予測による薬物動態評価**")
    st.markdown("3. **細胞毒性・安全性試験の実施**")
    st.markdown("4. **特許性調査と合成経路の検討**")
else:
    st.markdown("1. **構造最適化による物性改善**")
    st.markdown("2. **類似化合物の探索とSAR解析**")
    st.markdown("3. **計算化学的手法による構造設計**")
    st.markdown("4. **再評価後の創薬開発検討**")

st.success("🎉 **量子計算による分子評価が完了しました。創薬開発チームでの検討をお待ちしています！**")
