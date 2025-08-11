# 💊 AI駆動創薬 × 量子コンピュータ デモ

## 概要

このプロジェクトは、グラフベース拡散モデルと量子コンピュータを組み合わせた革新的な創薬支援システムです。AIによる分子生成と量子計算による高精度な物性予測により、新薬開発プロセスを大幅に加速します。

## 🎯 主な機能

### 1. グラフベース分子生成
- **E(3)等変性拡散モデル**: 3D分子構造の自然な生成
- **リアルタイム可視化**: 分子生成過程の3Dアニメーション
- **化学的妥当性**: 既存の分子パターンを学習した高品質生成

### 2. 量子化学計算 (VQE)
- **基底状態エネルギー**: 変分量子固有値ソルバーによる高精度計算
- **HOMO-LUMOギャップ**: 分子の反応性と安定性の評価
- **双極子モーメント**: 溶解性と膜透過性の予測

### 3. 創薬支援機能
- **薬物動態予測**: Lipinskiの法則に基づく評価
- **創薬適合性スコア**: 総合的な創薬候補評価
- **最適化推奨事項**: 構造改善の具体的提案

## 🚀 クイックスタート

### 1. 環境セットアップ

```bash
# リポジトリのクローン（例）
git clone <repository-url>
cd graph-diffusion-quantum-molecular

# 依存関係のインストール
uv sync
# または
pip install -e .
```

### 2. Streamlitアプリケーションの起動

```bash
streamlit run app.py
```

ブラウザで `http://localhost:8501` にアクセスしてアプリケーションを使用できます。

## 📱 Streamlitアプリケーション操作ガイド

### メイン画面の構成（現状のUI）

アプリケーションは以下のセクションで構成されています：

#### 1. サイドバー（Generation Settings）
- **Number of samples**: 生成目標数（1〜24）
- **Sampling seed**: 乱数シード（任意）
- **Use Active Space (VQE)**: VQEでアクティブ空間を使用
- **Active Electrons / Active Orbitals**: アクティブ空間の設定
- **Basis Set**: `sto3g` / `6-31g`

#### 2. 分子生成（ストリーミング）
- ボタン: **Start Streaming / Clear**
- オプション: **🎭 3D拡散可視化を表示（群衆感動モード）**
- 生成中は `Streaming SMILES:` が1件ずつ増加し、`Number of samples` に到達すると自動停止
- 重複は自動除外。足りない場合は安全なフォールバックで補完
- 停止後に `Candidates (SMILES)` がちょうど n 件で表示

#### 3. 可視化
- **🧪 3D分子マトリクス (静的表示)**: 生成分子を格子状に py3Dmol で表示（表示数スライダ）
- **🧮 化学式マトリクス**: 各分子の化学式を格子表示
- オプション: 拡散過程のリアルタイム可視化（複数分子をグリッドでアニメーション）

#### 4. Inspect & Prepare (RDKit)
- 一覧から1分子選択（`Pick one molecule`）または SMILES 入力
- **2D depiction / 3D view**、基本プロパティ表示
- **Download**: SDF（常時）/ MOL2（環境によりフォールバック）

#### 5. Quantum Chemistry (Qiskit Nature VQE)
- テスト用: **H2分子でVQE計算をテスト**（簡易検証）
- ストリーミング中は VQE を一時停止（停止後に実行）
- 指標: VQE Energy / Reference / Abs Error、HOMO/LUMO/GAP、Dipole、収束曲線

#### 5. 創薬適合性評価セクション
- **Lipinskiの法則**: 薬物動態の評価
- **創薬適合性スコア**: 総合評価（0-100点）
- **改善提案**: 構造最適化の推奨事項

### 操作手順（現状のフロー）

##### ステップ1: 起動
```bash
streamlit run app.py
```
ブラウザで `http://localhost:8501` を開く。

##### ステップ2: サイドバー設定
- Number of samples を設定（例: 8）
- Sampling seed（任意）
- VQE 用に Use Active Space / Active Electrons / Active Orbitals / Basis Set を必要に応じて設定

##### ステップ3: 分子生成（ストリーミング）
- 「Start Streaming」をクリック
- `Streaming SMILES` が1件ずつ増え、n 件に到達すると自動停止
- 必要に応じて「Clear」でリセット

##### ステップ4: 候補の可視化と確認
- `Candidates (SMILES)` に n 件表示
- 「🧪 3D分子マトリクス (静的表示)」「🧮 化学式マトリクス」で一覧確認
- 拡散アニメはチェックボックスで有効化

##### ステップ5: 詳細確認とエクスポート（RDKit）
- `Pick one molecule` で1つ選択（または SMILES を入力）
- 2D/3D 表示、基本プロパティを確認
- SDF ダウンロード（MOL2 は環境によりフォールバック）

##### ステップ6: 量子化学計算（VQE）
- ストリーミング中は VQE が一時停止（停止後に自動実行）
- テスト用の H2 設定で疎通確認も可能
- 結果: VQE エネルギー、参照値、誤差、HOMO/LUMO/GAP、Dipole、収束曲線

##### ステップ6: 創薬適合性評価の確認
1. **Lipinskiの法則の確認**
   - **分子量**: ≤ 500 Da（推奨）
   - **LogP**: ≤ 5（推奨）
   - **水素結合供与体**: ≤ 5（推奨）
   - **水素結合受容体**: ≤ 10（推奨）

2. **創薬適合性スコアの確認**
   - **総合スコア**: 0-100点での評価
   - **詳細分析**: 各項目の詳細な評価
   - **改善提案**: 構造最適化の推奨事項

3. **結果の解釈**
   - スコアが高いほど創薬候補として有望
   - 改善提案に基づいて構造最適化を検討

#### 高度な機能の使用方法

##### バッチ処理の実行
1. **複数分子の一括生成**
   - 分子生成数を増やして（例：20個）一括生成
   - 生成された分子を一覧で確認

2. **一括量子計算**
   - 複数の分子を選択して一括計算
   - 結果を比較して最適な分子を特定

3. **結果の保存**
   - 生成結果をCSV形式で保存
   - 計算結果をJSON形式で保存

##### カスタム分子の分析
1. **SMILES入力**
   - 既存の分子のSMILES文字列を入力
   - 「カスタム分子分析」ボタンをクリック

2. **3D構造生成**
   - 入力されたSMILESから3D構造を生成
   - RDKitによる構造最適化を実行

3. **量子計算実行**
   - カスタム分子に対して量子化学計算を実行
   - 創薬適合性評価を実行

##### リアルタイム生成の使用
1. **ストリーミング生成の開始**
   - 「リアルタイム生成」ボタンをクリック
   - 分子が1つずつ生成される

2. **生成過程の観察**
   - 各分子の生成過程をリアルタイムで確認
   - 生成された分子を即座に評価

3. **生成の停止**
   - 十分な数の分子が生成されたら停止
   - 結果を保存して後で分析



#### 最適な使用パターン

##### 初心者向けパターン
1. **最初の設定**
   - 分子生成数: 5個
   - 最大原子数: 15個
   - デバイス: CPU

2. **基本的な流れ**
   - 分子生成 → 3D確認 → 量子計算 → 評価

3. **結果の解釈**
   - 創薬適合性スコア60点以上を目標
   - Lipinskiの法則を満たす分子を優先

##### 上級者向けパターン
1. **効率的な探索**
   - 分子生成数: 20個
   - 最大原子数: 25個
   - デバイス: CUDA（GPU使用）

2. **詳細分析**
   - バッチ処理で複数分子を一括評価
   - 結果を保存して後で比較分析
   - 改善提案に基づく構造最適化

3. **カスタマイズ**
   - 特定の分子骨格をターゲットにした生成
   - 条件付き生成の活用
   - 学習済みモデルの使用

## 🔬 E(3)-equivariant Diffusionモデル詳細解説

### 論文ベース実装について

この実装は、**2022年の「E(3)-Equivariant Diffusion for Molecule Generation in 3D」論文**（University of Amsterdam / Qualcomm AI Research）をベースにしています。

#### 論文との対応関係

**基本コンセプトの一致:**
- ✅ **E(3)等価性**: 回転・並進に対して不変な分子生成
- ✅ **3D座標生成**: 原子種と3D座標の同時生成
- ✅ **拡散モデル**: ノイズから秩序への変換過程
- ✅ **EGNN統合**: E(3)等価なGNNを拡散モデルに組み込み

**論文の主要技術:**
- E(3)等価なGNN（EGNN）を拡散モデルに組み込み
- 原子種と3D座標を同時に生成
- メッセージパッシングによる等価性の保持

#### 実装の違いと簡略化

**論文の完全実装 vs 今回の実装:**

| 要素 | 論文（完全版） | 今回の実装（簡略版） |
|------|----------------|---------------------|
| **E(3)等価性** | 完全なEGNN | 簡略化されたメッセージパッシング |
| **テンソル積** | 複雑な球面調和関数 | 相対位置ベースの簡易実装 |
| **拡散スケジュール** | 高度なノイズスケジュール | 線形ノイズスケジュール |
| **学習済みモデル** | 大規模データセットで学習 | 初期化されたモデル |
| **計算複雑度** | 高（完全な等価性） | 中（簡略化された等価性） |

#### 簡略化の理由

**実装の簡略化を行った理由:**
1. **学習済みモデルの不在**: 論文の完全実装には大規模な学習データが必要
2. **計算リソース**: 完全なEGNNは計算コストが高い
3. **デモンストレーション目的**: 概念の理解と動作確認が主目的
4. **拡張性**: 段階的に機能を追加できる基盤として

#### 今後の拡張可能性

**完全実装への道筋:**
```python
# 将来的な完全実装例
class FullE3DiffusionModel(nn.Module):
    def __init__(self):
        # 完全なEGNNレイヤー
        self.egnn_layers = nn.ModuleList([
            EGNNLayer(irreps_in, irreps_out)
            for _ in range(num_layers)
        ])
        
        # 球面調和関数ベースのテンソル積
        self.tensor_products = nn.ModuleList([
            o3.TensorProduct(irreps_in, irreps_out, instructions)
            for _ in range(num_layers)
        ])
```

### 基本原理

E(3)-equivariant diffusionモデルは、分子の3D構造を直接生成する革新的なAI技術です。

#### 1. E(3)等変性とは

E(3)等変性とは、3次元ユークリッド空間での回転・並進に対して不変な性質を指します。

```
E(3) = SO(3) × ℝ³
- SO(3): 3次元回転群
- ℝ³: 3次元並進群
```

#### 2. 分子生成における重要性

**従来の問題点:**
- SMILES文字列ベースの生成では3D情報が失われる
- 後処理での3D構造生成が必要
- 物理的制約が考慮されない

**E(3)等変性の利点:**
- 3D座標を直接生成
- 物理的制約を自然に満たす
- 回転・並進に対して不変な表現

### 技術的実装

#### 1. アーキテクチャ概要

```python
class SimpleE3DiffusionModel(nn.Module):
    def __init__(self, num_atom_types=119, hidden_dim=128, num_layers=6):
        # 時間埋め込み
        self.time_embed = nn.Sequential(...)
        
        # 原子タイプ埋め込み
        self.atom_embed = nn.Embedding(num_atom_types, hidden_dim)
        
        # E(3)等変性レイヤー（論文のEGNNを簡略化）
        self.layers = nn.ModuleList([
            SimpleE3Layer(hidden_dim, hidden_dim) 
            for _ in range(num_layers)
        ])
        
        # 出力ヘッド
        self.atom_type_head = nn.Linear(hidden_dim, num_atom_types)
        self.position_head = nn.Linear(hidden_dim, 3)
```

#### 2. メッセージパッシングレイヤー（論文のEGNNを簡略化）

```python
class SimpleE3Layer(MessagePassing):
    def message(self, x_i, x_j, pos_i, pos_j):
        # 相対位置の計算（E(3)等価性の保持）
        rel_pos = pos_j - pos_i
        
        # ノード特徴と相対位置の結合
        message_input = torch.cat([x_i, x_j, rel_pos], dim=-1)
        
        # メッセージ計算（論文の複雑なテンソル積を簡略化）
        return self.message_mlp(message_input)
```

#### 3. 拡散プロセス

**ノイズ追加過程:**
```
q(x_t | x_0) = N(x_t; √α_t x_0, (1-α_t)I)
```

**デノイズ過程:**
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

### 分子生成の流れ

#### 1. 初期化
```python
# 純粋なノイズから開始
atom_types = torch.randint(1, 6, (num_atoms,))  # ランダム原子タイプ
positions = torch.randn(num_atoms, 3) * 2.0     # ランダム3D座標
```

#### 2. 反復デノイズ
```python
for t in reversed(range(num_timesteps)):
    # モデルによる予測
    atom_logits, pos_pred = model(data, t)
    
    # デノイズステップ
    positions = scheduler.denoise_step(positions, pos_pred, t)
    atom_types = torch.argmax(atom_logits, dim=-1)
```

#### 3. 後処理
```python
# 原子間距離に基づく結合推定
for i, j in atom_pairs:
    dist = ||positions[i] - positions[j]||
    if dist < bond_threshold:
        add_bond(i, j)
```

### 学習済みモデルの使用

#### 1. チェックポイントの読み込み

```python
# 学習済みモデルの読み込み
sampler = DiffusionSampler(
    ckpt_path="path/to/checkpoint.pt",
    device="cuda"
)
```

#### 2. 条件付き生成

```python
# 特定の条件での分子生成
molecules = sampler.sample_smiles(
    n=10, 
    cond={"target_property": "drug_likeness"}
)
```

### 性能評価

#### 1. 生成品質指標

- **化学的妥当性**: 有効なSMILESの割合
- **多様性**: 生成分子の構造的多様性
- **新規性**: 既存データベースとの重複率

#### 2. 計算効率

- **生成速度**: 分子/秒
- **メモリ使用量**: GPU/CPUメモリ消費
- **スケーラビリティ**: 大規模分子への対応

### 複数サンプル生成のメリット（E(3)等変性拡散モデル）

- **多様性の確保**: 拡散は確率的・多峰性。複数生成で異なる骨格・置換・結合様式・互変異性体・コンフォーマを広く回収できる。
- **ベストオブN選択**: 薬物様性・合成容易性・毒性・VQEエネルギーなどの指標で上位のみ採用し、ヒット確率を向上。
- **不確実性推定**: 物性の分散や valid/unique/novel 率からモデルの信頼区間やリスクを見積もれる。
- **失敗耐性**: 単発生成が無効構造でも、複数生成で有効分子が得られる確率が上がる。
- **化学空間カバレッジ**: クラスタリングで代表構造を抽出し、探索の偏りを低減。下流評価（ドッキング/量子計算）に多様な候補を供給。
- **最適化・パレート探索**: 多目的（活性・溶解性・合成性など）のトレードオフ前面でパレートフロント候補を得やすい。
- **GPU効率の向上**: バッチ生成でスループットと GPU 使用率が上がり、同時間あたりの有効候補数を増やせる。
- **収束/異常解析**: 複数トレースを比較することで拡散過程の失敗モードを診断し、しきい値や前処理のチューニングに役立つ。

### 応用例

#### 1. 創薬支援
- **リード化合物発見**: 新規化合物の生成
- **構造最適化**: 既存化合物の改良
- **標的特異性**: 特定タンパク質への結合分子設計

#### 2. 材料科学
- **有機材料**: 有機EL、太陽電池材料
- **触媒設計**: 効率的な触媒分子の生成
- **ポリマー**: 機能性ポリマーの設計

### 参考文献

**ベース論文:**
- **"E(3)-Equivariant Diffusion for Molecule Generation in 3D"** (2022)
  - University of Amsterdam / Qualcomm AI Research
  - E(3)等価なGNN（EGNN）を拡散モデルに組み込み、原子種＋3D座標を同時生成
  - [GitHub](https://github.com/qualcomm-ai-research/e3-equivariant-diffusion) / [論文](https://arxiv.org/abs/2203.17003)

**関連研究:**
- "Equivariant Graph Neural Networks" (ICML 2020)
- "Geometric Deep Learning: Going beyond Euclidean data" (IEEE Signal Processing Magazine 2017)
- "Diffusion Models: A Comprehensive Survey of Methods and Applications" (arXiv 2022)

## 🔬 技術詳細

### グラフベース拡散モデル

#### 基本原理
グラフベース拡散モデルは、分子をグラフ構造（原子をノード、結合をエッジ）として表現し、ノイズから秩序への変換過程を学習します。

#### 創薬でのメリット
1. **自然な構造表現**
   - 分子の本質的な構造を直接的に扱える
   - 化学結合の情報を保持
   - 3D等変性により回転・並進に対して不変

2. **高品質な分子生成**
   - 既存薬物に類似した構造を生成
   - 新規性と妥当性のバランス
   - 標的特異性を考慮した設計

3. **創薬プロセスの加速**
   ```
   従来: 化合物ライブラリ → スクリーニング → 数ヶ月〜数年
   グラフ拡散: AI生成 → 即座評価 → 数日〜数週間
   ```

#### 技術的優位性
| 手法 | グラフ拡散 | 従来のVAE/GAN |
|------|------------|---------------|
| 構造表現 | グラフ（自然） | 文字列/配列 |
| 3D情報 | 直接処理 | 後処理必要 |
| 化学妥当性 | 高 | 中〜低 |
| 生成品質 | 高 | 中 |

### 量子コンピュータ (VQE)

#### VQEアルゴリズムの原理
変分量子固有値ソルバー（Variational Quantum Eigensolver）は、量子コンピュータを用いて分子の基底状態エネルギーを計算するアルゴリズムです。

#### 創薬での重要性
1. **高精度エネルギー計算**
   - 古典計算では困難な大規模分子の計算
   - 電子相関効果の正確な取り扱い
   - 基底状態と励起状態のエネルギー差

2. **分子特性の予測**
   - **HOMO-LUMOギャップ**: 分子の反応性と安定性
   - **双極子モーメント**: 溶解性と膜透過性
   - **電子密度分布**: 反応部位の予測

3. **創薬設計への応用**
   - 薬物-標的タンパク質相互作用の予測
   - 代謝安定性の評価
   - 副作用リスクの評価

#### 技術的実装
```python
# VQE計算の流れ
1. 分子構造 → ハミルトニアン構築
2. 量子回路（アンサッツ）設計
3. パラメータ最適化
4. 基底状態エネルギー計算
5. 分子特性の導出
```

## 💊 創薬プロセスでの活用

### 1. リード化合物発見
- **AI生成**: グラフ拡散モデルによる新規化合物生成
- **量子評価**: VQEによる高精度物性予測
- **スクリーニング**: 創薬適合性スコアによる選別

### 2. 構造最適化
- **官能基置換**: 最適な位置の自動提案
- **骨格修飾**: 新しい分子骨格の生成
- **物性改善**: 溶解性・膜透過性の向上

### 3. 新規ターゲット対応
- **未知タンパク質**: 構造情報からの結合分子設計
- **変異対応**: 耐性獲得への対応
- **個別化医療**: 患者固有の分子設計

## 🚀 実用的メリット

### 計算効率
- **並列処理**: 複数分子の同時生成・評価
- **GPU活用**: 深層学習フレームワークとの親和性
- **スケーラビリティ**: 大規模分子ライブラリ構築

### コスト削減
- **実験コスト**: 合成実験の削減
- **時間短縮**: 開発期間の大幅短縮
- **リスク低減**: 失敗リスクの事前評価

### 精度向上
- **量子精度**: 古典計算を超える精度
- **AI学習**: 大量データからの学習
- **統合評価**: 複数指標の総合判断

## 📊 創薬適合性評価

### 評価指標
1. **Lipinskiの法則**
   - 分子量 ≤ 500 Da
   - LogP ≤ 5
   - 水素結合供与体 ≤ 5
   - 水素結合受容体 ≤ 10

2. **量子化学的特性**
   - 基底状態エネルギー
   - HOMO-LUMOギャップ
   - 双極子モーメント

3. **創薬適合性スコア**
   - 総合評価（0-100点）
   - 詳細分析
   - 改善提案

### 最適化推奨事項
- **エネルギー安定性**: 芳香環や共役系の導入
- **反応性調整**: HOMO-LUMOギャップの最適化
- **溶解性改善**: 極性官能基の導入
- **膜透過性**: 疎水性部位の調整

## 🔧 技術的実装

### 必要なライブラリ
```bash
pip install streamlit plotly rdkit qiskit qiskit-nature pyscf numpy scipy torch torch-geometric e3nn
```

### 主要コンポーネント
- `src/e3_diffusion.py`: E(3)-equivariant diffusionモデルの実装
- `generator.py`: 分子生成サンプラー
- `qchem.py`: VQE計算と量子化学計算
- `diffusion_visualizer.py`: 3D可視化機能
- `app.py`: Streamlitアプリケーション

### 実行方法
```bash
# 依存関係のインストール
uv sync

# テストの実行
python test_e3_diffusion.py

# アプリケーションの起動
streamlit run app.py
```

## 🧪 テストと検証

### テストスイート
```bash
# 全テストの実行
python test_e3_diffusion.py

# 個別テスト
python -c "from test_e3_diffusion import test_e3_model; test_e3_model()"
```

### テスト内容
- ✅ E(3)-equivariant diffusion model
- ✅ Diffusion Scheduler
- ✅ Molecular Sampler
- ✅ DiffusionSampler
- ✅ SMILES変換

## 🎯 今後の展望

### 短期目標
- **精度向上**: より高精度なVQE計算
- **速度改善**: 量子計算の高速化
- **機能拡張**: より多くの分子特性予測

### 中期目標
- **大規模化**: より大きな分子の処理
- **統合化**: 他の創薬ツールとの連携
- **実用化**: 製薬企業での実証実験

### 長期目標
- **創薬革命**: 創薬プロセスの根本的変革
- **個別化医療**: 患者固有の薬物設計
- **難治性疾患**: 新たな治療法の開発

## 📚 参考文献

1. **グラフ拡散モデル**
   - "E(3)-Equivariant Diffusion for 3D Molecular Generation"
   - "Graph Diffusion Models for 3D Molecular Generation"

2. **VQEアルゴリズム**
   - "Variational Quantum Eigensolver: A Review"
   - "Quantum Computing for Quantum Chemistry"

3. **創薬応用**
   - "AI in Drug Discovery: Current Trends and Future Directions"
   - "Quantum Computing Applications in Pharmaceutical Research"

## 🤝 貢献

このプロジェクトへの貢献を歓迎します。以下の方法で参加できます：

1. **バグ報告**: Issuesでの問題報告
2. **機能提案**: 新機能の提案
3. **コード改善**: Pull Requestでの改善
4. **ドキュメント**: ドキュメントの改善

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

---

**💊 AI駆動創薬 × 量子コンピュータで、創薬の未来を切り開こう！**
