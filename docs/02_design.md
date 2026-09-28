# CMN-C1-021 OptimizationSolverAgent — 設計仕様書

## 1. 概要

自然言語で記述された最適化問題を **構造化スペック** に変換し、決定論的に OSS ソルバ
（PuLP / CBC）のモデルを組み立てて解き、結果を自然言語で説明する Cat 1 汎用エージェント。

対応問題クラス: **線形計画（LP）** と **混合整数計画（MILP）**。
変数は連続 / 整数 / バイナリ、制約は線形（`<=`, `>=`, `==`）、目的は最小化 / 最大化。

> **設計原則:** LLM は **構造化スペック（JSON）のみ** を出力し、コードは生成・実行しない。
> ソルバモデルは決定論的ビルダーが構築する。これにより任意コード実行 / サンドボックス脱出の
> リスクを排除し、サンドボックス基盤への依存をなくす（旧 decision.md §7 ブロッカー解消）。

### 対象スコープ（2026-06-03 確定）

**位置づけ: 小〜中規模の汎用 LP/MILP ソルバ。**

| 区分 | 内容 |
|---|---|
| ✅ 対応 | 資源配分 / 生産計画(product-mix) / 配合・ブレンド / 小規模な割当・シフト(binary) / 小規模ナップサック・パッキング。`max_variables=200` / `max_constraints=500` の範囲 |
| ❌ 対象外 | **非線形・二次計画** / **大規模 VRP（経路変数が爆発）** / 条件付き(if-then)制約の自動線形化 / 多段の動的意思決定(ReAct) / 確率・ロバスト最適化。これらは派生テンプレ or 別パターンで対応 |

> 「OptimizationSolver」という汎用名だが、本テンプレは **線形・小中規模に限定**。大規模 VRP は
> 列生成等が必要で本テンプレ単体では成立しない（看板倒れを避けるため明記）。

### 入力モードの位置づけ（論点5 の確定方針、2026-06-03）

- **構造化 JSON 入力（`problem_spec`）を "正" とする**（確定的・検証可能）。アプリ / config が
  構造を持つ呼び出しはこちらを使う。
- **自然言語入力（`problem_description`）は補助（ベストエフォート）**。LLM 定式化には誤り
  （制約抜け等）のリスクがあり、本標準テンプレでは精度を保証しない。
- 実 LLM での定式化精度は **STG invoke で検証してからリリース**する（テストは fake LLM 中心のため）。
- 重要意思決定向けに「抽出した制約・目的を要約提示 → 人間が確認してから求解」する確認ゲートが
  必要な場合は **派生テンプレ** で追加する（標準テンプレには含めない）。

---

## 2. レイヤーアーキテクチャ

> **アーキテクチャ:** 本テンプレートは **L1 base**（**L1 フレームワーク `framework/`**）を
> **直接継承**する（旧 Level 2 → Level 3 継承モデルは撤廃）。L1 base = `AgentBaseGraph`。

```
L1 base: framework/                    ← 直接継承（AgentBaseGraph / FunctionNode / TrustLevel）
CMN-C1-021                             ← 本テンプレート（L1 base を直接継承）
```

- 継承構造: `OptimizationSolverAgent(OptimizationSolverGraph)` /
  `OptimizationSolverGraph(AgentBaseGraph)`。
- 実行は L1 `AgentBaseGraph.invoke(state, config)` を継承して使う（**production エントリは
  `invoke()`。`run()` は使わない** — `run()` は存在せず、呼ぶと AttributeError になり
  S-1 trust ゲートも通らない）。`self.graph` 合成も不可（独自クラスは 5 層セキュリティを迂回する）。

```python
# src/graph/graph.py — グラフ定義（ノード登録 + エッジ配線）
class OptimizationSolverGraph(AgentBaseGraph):
    def register_nodes(self): ...   # constraint_parse → solver_dispatch → solution_format
    def add_edges(self): ...

# src/nodes/constraint_parse.py — entry ノードが S-1 を宣言（BaseNode.__call__ が自動強制）
class ConstraintParseNode(BaseNode):
    required_trust_level = TrustLevel.VERIFIED_EXTERNAL      # S-1（ADR-006）

# src/agent.py — Agent は Graph を継承し、S-2〜S-4 のフックを実装（S-1 は __call__ が担う）
class OptimizationSolverAgent(OptimizationSolverGraph):
    REQUIRED_TRUST_LEVEL = TrustLevel.VERIFIED_EXTERNAL     # 有効値: ANONYMOUS/VERIFIED_EXTERNAL/INTERNAL
    # __pre_invoke__ は廃止（ADR-006）— framework は呼ばない dead code
    def _security_gate_input(self, state, config=None): ...  # S-2
    def _security_gate_output(self, state, config=None): ... # S-3
    def _emit_trace_event(self, state, config=None): ...     # S-4
```

---

## 3. ノードフロー

### AGENTIC STAR Marketplace entry adapter

リポジトリ直下の `cli.py` は `MarketplaceGraph(OptimizationSolverAgent)` を公開する。
Marketplace runner は自然言語の文字列を 1 本渡すため、adapter は compile 前に
`config["llm"]` へ credential-free な lazy Azure wrapper を注入する。runner のライフサイクルは
`agent_cls(config)` → `compile()` → `provision_secrets(provider)` の順なので、wrapper の構築時と
compile 時には secret を解決しない。`complete()` の呼び出しごとに allowlist provider から
`AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_DEPLOYMENT` を `require()` し、
新しい `AzureOpenAIClient` を構築する。接続値と client は wrapper にキャッシュしない。

`provision_secrets()` は3つの Azure 接続値を起動時に `require()` して fail closed とする。
`EnvProvider` が利用できる SDK ではそれを、AgentCore 1.0.0 / 1.0.1 のように利用できない SDK では
runner provider を delegate とし、どちらも3キーだけを公開する allowlist provider で包む。
process environment を adapter から直接読むことはない。

`invoke()` は runner の `input_context` をマージせず破棄し、空の dict として組み直す。
`conversation_history` は S-2 の検査対象外であるため State に入れない。また Marketplace は
`AWAITING_HUMAN` を resume できないので、context がある場合だけ `hitl_allowed=False` の copy を渡す。

#### tenant / org scope を実装しない理由

この adapter は tenant / org scope を実装せず、`src/entry_adapter/` も import しない。

1. この repository の entry-adapter part は v0.2.0 で、後発 part の `SCOPE_KEYS` /
   `STG_DEFAULT_SCOPE_ENVS` API を持たず、固定2キー形の `STG_DEFAULT_TENANT_ENV` /
   `STG_DEFAULT_ORG_ENV` しかない。part は変更禁止であり、新 API による実装経路がない。
2. `src/nodes/` と `src/graph/` の全コードパスに tenant / org を読む処理がない。
3. standalone entry の `src/api/server.py` も同じ設計判断を明記している:

   > this template has NO scope model. OptimizationSolverAgent is a generic,
   > use-case-agnostic Cat 1 solver — it holds no tenant/org resources and there
   > is no code path that reads a per-request tenant/org scope. Enforcing
   > authenticate_scope() would gate on a scope nothing consumes (STG-default
   > theater), so only the auth core is wired.

Marketplace runner が trust を stamp するため adapter で入口認証を重複実装せず、各ノードの
`VERIFIED_EXTERNAL` 要求を既存の S-1 として維持する。この境界で fail closed にする対象は、
実際に自然言語定式化が消費する Azure 接続値である。

固定 3 ノード線形パイプライン:

```
constraint_parse → solver_dispatch → solution_format
```

### 各ノードの責務

| ノード | クラス | 責務 |
|---|---|---|
| `constraint_parse` | `ConstraintParseNode` | 入力検証（空 / 長さ上限）+ LLM で `problem_spec`（JSON）抽出 + 構造妥当性検証。問題サイズ上限（`max_variables`/`max_constraints`）超過は `PROBLEM_TOO_LARGE` |
| `solver_dispatch` | `SolverDispatchNode` | `problem_spec` → PuLP モデルを決定論的に構築 → CBC で求解（`solver_timeout_s`）。`solver_status`（OPTIMAL/INFEASIBLE/UNBOUNDED/TIMEOUT）+ `solution`（JSON）を生成 |
| `solution_format` | `SolutionFormatNode` | 決定論サマリを先頭に組み立て、LLM の説明文を「参考」節として後ろに足す（INFEASIBLE 等も説明）+ S-3 出力ゲート（mask + credential ブロック） |

### この 3 ノード分割の根拠

- **構造化（LLM）/ 求解（決定論）/ 説明（LLM）の分離** = Single-responsibility。
  LLM の非決定性をソルバ実行から隔離し、求解部を完全に検証可能にする。
- 実装単位（ConstraintParser / SolverDispatch / SolutionFormatter）と 1:1 対応。

---

## 4. State スキーマ（flat TypedDict、msgpack serializable）

primitives + JSON 文字列のみ。Pydantic / dataclass / 任意 Python オブジェクトは保存しない
（msgpack 非互換 + LangGraph checkpoint 安全性）。
リスト / dict は **JSON 文字列** として保持する（PB-2 スカラ安全性）。

```python
class OptimizationState(TypedDict, total=False):
    # ── Input ──
    problem_description: Optional[str]   # 自然言語の最適化問題（max 10,000 chars）
    session_id: Optional[str]

    # ── ConstraintParseNode 出力 ──
    problem_spec: Optional[str]          # JSON: {sense, variables[], constraints[], objective{}}
    spec_valid: Optional[bool]           # 構造妥当性チェック結果
    variable_count: Optional[int]
    constraint_count: Optional[int]

    # ── SolverDispatchNode 出力 ──
    solver_status: Optional[str]         # OPTIMAL | INFEASIBLE | UNBOUNDED | TIMEOUT | UNDEFINED
    solution: Optional[str]              # JSON: {"x1": 4.0, "x2": 3.0}
    objective_value: Optional[float]     # 目的関数値（OPTIMAL 時）
    solver_backend: Optional[str]        # 使用ソルバ識別（"PULP_CBC_CMD"）

    # ── SolutionFormatNode 出力 ──
    formatted_output: Optional[str]      # S-3 通過後の最終応答（自然言語）

    # ── エラー伝播 ──
    error: Optional[str]                 # 真値で AgentBaseGraph._invoke_impl() が break
    error_code: Optional[str]            # EMPTY_INPUT / PROBLEM_TOO_LARGE / SPEC_PARSE_FAILED / S2_INJECTION_DETECTED / S3_BLOCKED / SOLVER_FAILED
    error_message: Optional[str]

    # ── 一時（直接 execute() 経路のみ。出力 state に残さない） ──
    invocation_context: Optional[object]
```

### `problem_spec` の JSON 契約

```json
{
  "sense": "maximize",
  "variables": [
    {"name": "x1", "type": "continuous", "low": 0, "up": null},
    {"name": "x2", "type": "integer",    "low": 0, "up": 10}
  ],
  "objective": {"x1": 1.0, "x2": 1.0},
  "constraints": [
    {"coeffs": {"x1": 1.0}, "op": "<=", "rhs": 4.0},
    {"coeffs": {"x2": 1.0}, "op": "<=", "rhs": 3.0}
  ]
}
```

- `type`: `continuous` | `integer` | `binary`
- `op`: `<=` | `>=` | `==`
- 係数は線形のみ（変数名 → 係数の dict）。係数・境界・rhs は数値必須（非数値は `SPEC_PARSE_FAILED`）。
- **`low` 省略時のデフォルト = `0`（非負）**。LP 標準形の慣例に倣う。`up` 省略 = 上限なし。
  > ✅ **決定（2026-06-03、案A）**: 省略時は **非負（0）**。負値が必須な変数（在庫バランス /
  > ネットワークフロー等）は `low` を明示する運用（下限なし(None) 既定は採らない）。

---

## 5. Config 仕様（`config/agent.yaml`）

```yaml
template_id: CMN-C1-021
agent_name: OptimizationSolverAgent
config:
  # ── 問題サイズ上限（暴走防止） ──
  max_variables: 200
  max_constraints: 500
  max_input_chars: 10000
  # ── ソルバ設定 ──
  solver_timeout_s: 10            # CBC 求解タイムアウト
  # ── LLM ──
  # クライアントは config["llm"] に BaseLLM として**注入**される（同系テンプレート共通の注入シーム）。
  # model / temperature / max_tokens は注入クライアントの持ち物で、テンプレの
  # config では選択できない（テンプレにモデル ID も具体クライアントも書かない）。
  # 定式化の再現性が要る場合は、注入側が temperature を低く設定したクライアントを渡す。
# ── セキュリティ ──
category: Cat 1                    # Cat 一貫性ゲート（template_id C1 / category / README が一致）
required_trust_level: VERIFIED_EXTERNAL   # 有効値: ANONYMOUS / VERIFIED_EXTERNAL / INTERNAL のみ
s3_gate_enabled: true             # ❗ 絶対 false にしない
mask_patterns: []                 # 派生で PII ラベルを追加
audit_log_fields:
  - template_id
  - session_id
  - variable_count
  - constraint_count
  - solver_status
# NOTE: problem_description / solution の本文は NEVER ログしない（メタのみ）
```

---

## 6. 求解とタイムアウト・実行不能ポリシー

- **タイムアウト:** PuLP `PULP_CBC_CMD(timeLimit=solver_timeout_s, msg=0)`。常に timeLimit を
  設定して solve するため、CBC が `NotSolved`(0) を返したら **暫定解の有無に依存せず `TIMEOUT`**
  にマップする（`UNDEFINED` と区別する）。暫定解があれば目的関数値を併記する。
- **`binary` 変数の `low`/`up`:** PuLP は `LpBinary` の境界を**無条件に `0..1` へ上書き**する
  ため、`binary, low=2` のような spec は黙って握り潰され、**指定下限に違反する解を OPTIMAL として
  返して**しまう。`{0,1}` との共通部分をこちらで解決する（空なら `INFEASIBLE`、1 点なら
  `Integer` で固定、それ以外は通常の binary）。
- **変数の定義域が空（`low > up`）:** ソルバを呼ばずに `INFEASIBLE` を返す。CBC に渡すと
  求解ではなく**ソルバ実行そのものが失敗**し、正常な求解結果であるべきものが `SOLVER_FAILED`
  になってしまう（実 LLM が「x は 5 以上かつ 4 以下」を制約 2 本ではなく `low=5`/`up=4` と
  定式化して踏んだ経路）。
- **実行不能 / 非有界:** `INFEASIBLE` / `UNBOUNDED` をエラーにせず、`solver_status` として
  正常に返し、SolutionFormatNode が「制約を満たす解が存在しない」等を自然言語で説明する
  （graceful degradation — 求解結果はビジネス的に有意な出力）。
- **問題サイズ超過:** ConstraintParseNode が `PROBLEM_TOO_LARGE` で早期終了。

---

## 7. マルチ呼び出し分離（状態非共有）

本エージェントは **ステートレス**（会話履歴を持たない単発求解）。`problem_description` ごとに
独立した state で実行され、`session_id` は監査トレース用途のみ。ノード間共有サービス（リポジトリ）は
持たない。

---

## 8. セキュリティ設計

### S-1 — Trust チェック（`required_trust_level` / `BaseNode.__call__()`）

**ADR-006 準拠。** S-1 は `BaseNode.__call__()` が
**自動・非オーバーライドで強制**する。各ノードが `required_trust_level` クラス変数を宣言し、
`__call__()` が `state["caller_trust_level"]`（`AgentBaseGraph.invoke()` が `InvocationContext`
から書き込む）と照合、不足なら `execute()` に到達せず ERROR（`S1_TRUST_DENIED`）で停止する。

- entry ノード `ConstraintParseNode.required_trust_level = TrustLevel.VERIFIED_EXTERNAL`。
- **`__pre_invoke__` は廃止**（framework は呼ばない dead code）。テンプレ側に S-1 コードは書かない。
- 有効な TrustLevel は **`ANONYMOUS` / `VERIFIED_EXTERNAL` / `INTERNAL` の 3 値のみ**
  （`AUTHENTICATED` / `VERIFIED_INTERNAL` は import 時 AttributeError）。
- context 不在 → `caller_trust_level = ANONYMOUS` → entry ノードが拒否（**default-deny**）。

### S-2 — 入力ゲート（`_security_gate_input`）

`problem_description` **および `problem_spec`**（構造化スペック直接入力経路）に credential
パターン（`detect_credentials`）が混入していれば `S2_INJECTION_DETECTED` で停止。
ConstraintParseNode 内でも両経路で credential 走査を行い多層防御（構造化 spec 直接投入でも
S-2 をバイパスできない）。

### S-3 — 出力ゲート（`_security_gate_output`）— **実装済み**

`formatted_output`（無ければ生 LLM 出力）に対し `apply_masks` で PII マスク → `detect_credentials`
で credential 残存チェック。検出時は `S3_BLOCKED` で `formatted_output=None`。
`s3_gate_enabled: true` は非バイパス。SolutionFormatNode 内にも S-3 を埋め込み多層防御。

### S-4 — 監査トレース（`_emit_trace_event`）

`template_id` / `session_id` / `variable_count` / `constraint_count` / `solver_status` /
`has_error` / `error_code` のみを emit。**problem_description / solution の本文は NEVER 記録しない。**

### S-5 — 認証情報検出（自動）

`__init_subclass__` による import 時 JWT / API キー検出（framework 提供）。

---

## 9. エラー伝播戦略

`AgentBaseGraph._invoke_impl()`（invoke から呼ばれるグラフ実行フック）は各ノード後に
`state["error"]` が真なら break する。各ノードは entry で `state.get("error")` を確認し、
立っていれば即 return（後段スキップ）。

### エラーコード一覧

| error_code | 発生ノード | 意味 | fatal |
|---|---|---|---|
| `S1_TRUST_DENIED` | BaseNode.__call__ | caller_trust_level < required_trust_level（S-1）| ✅ |
| `EMPTY_INPUT` | constraint_parse | 入力が空 | ✅ |
| `PROBLEM_TOO_LARGE` | constraint_parse | 変数 / 制約が上限超過 | ✅ |
| `SPEC_PARSE_FAILED` | constraint_parse | LLM 出力が JSON 契約に適合しない（数値が非有限 `NaN`/`Infinity`、変数名の重複を含む。`json.loads()` は既定で `NaN`/`Infinity` を受理するため有限性まで検査する）| ✅ |
| `S2_INJECTION_DETECTED` | S-2 / constraint_parse | 入力に credential | ✅ |
| `SOLVER_FAILED` | solver_dispatch | モデル構築 / 求解の例外（理由は**例外の型名のみ**。PuLP/CBC の例外文はソルバ実行ファイルの絶対パスを含むため State に入れない）| ✅ |
| `LLM_NOT_CONFIGURED` | constraint_parse | 自然言語経路で `config["llm"]` が未注入（決定論代替が無いため縮退しない）| ✅ |
| `LLM_CALL_FAILED` | constraint_parse | 定式化の LLM 呼び出しが失敗（理由は**例外の型名のみ**）| ✅ |
| `S3_BLOCKED` | S-3 / solution_format | 出力に credential 残存 | ✅ |

### Graceful degradation

- `INFEASIBLE` / `UNBOUNDED` / `TIMEOUT` は **エラーではなく `solver_status`** として返し、
  自然言語で説明する。

---

## 10. 派生テンプレートのオーバーライド面

- `config/agent.yaml`: 問題サイズ上限 / タイムアウト / プロンプト差分。
- 業種特化（VRP / シフト等）は Cat 2 派生として system prompt と検証ルールを差し替え。
- `mask_patterns` に業種固有 PII ラベルを追加。

---

## 11. 呼び出しモード

### Production エントリは `invoke()`（`run()` は使わない）

```python
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

agent = OptimizationSolverAgent(config=CONFIG)
cfg = {"configurable": {"invocation_context": InvocationContext(
    caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)}}
out = agent.invoke({"problem_spec": "...", "session_id": "s1"}, cfg)
```

### invoke ライフサイクル

`_security_gate_input`（S-2）→ `_invoke_impl`（3 ノード。各ノード `__call__` が **S-1** を強制）→
`_security_gate_output`（S-3）→ `_emit_trace_event`（S-4）の順。`__pre_invoke__` は廃止（ADR-006）。

---

## 12. テスト計画（概要 — 詳細は `docs/03_test_spec.md`）

### Unit Tests（目標 8+ TC）
- ConstraintParseNode: 入力検証 / spec 抽出 / PROBLEM_TOO_LARGE / SPEC_PARSE_FAILED
- SolverDispatchNode: OPTIMAL（既知最適解）/ INFEASIBLE / INTEGER 変数 / SOLVER_FAILED
- SolutionFormatNode: 自然言語化 / S-3 マスク / S3_BLOCKED

### Integration Tests
- E2E: 既知の LP / MILP で最適値一致、S-1 lifecycle、INFEASIBLE 説明。

### Proof-of-Boundary Tests（目標 6+ PB）
- PB-1 trace 発火 / PB-2 state スカラ / PB-3 ソルバ実サービス境界（実 CBC 求解）/
  PB-4 import isolation / PB-5 checkpoint safety / PB-6 invoke 順序 / ノード契約（execute のみ）。

---

## 13. Definition of Done（設計 → 実装フェーズのゲート）

- [x] 提案内容の確定
- [x] `docs/02_design.md` コミット（L1 base 直接継承を明記）
- [x] 汚染スキャフォールド（別テンプレートの残骸）を削除しクリーン baseline 復元

---

## 14. 設計判断サマリ

| # | 判断 | 根拠 |
|---|---|---|
| 1 | LLM はコード生成せず構造化スペックのみ | 任意コード実行排除 → サンドボックス不要（旧 §7 ブロッカー解消） |
| 2 | ソルバは PuLP（CBC 同梱） | pure-pip・OSS・ライセンス不要・slim CI で動作 |
| 3 | INFEASIBLE/UNBOUNDED/TIMEOUT は非エラー | 求解結果はビジネス的に有意 → graceful degradation |
| 4 | 3 ノード（parse / solve / format） | 実装単位と 1:1、LLM 非決定性を求解から隔離 |
| 5 | Agent ⊃ Graph 継承、`invoke()` 継承（`run()` は使わない） | `run()` は存在せず、呼ぶと AttributeError かつ S-1 trust ゲートを迂回する |
| 6 | S-1 は `BaseNode.__call__()` が強制（`__pre_invoke__` 廃止）、TrustLevel=VERIFIED_EXTERNAL | ADR-006。有効な TrustLevel は 3 値のみ |
| 7 | LLM は `config["llm"]` 注入シームのみ（独自コンストラクタ引数を持たない）| `BaseGraph.__init__(self, config=None)` が framework の実契約で、呼び出し側は独自引数を渡さない＝永久に未配線になる |
| 8 | 生成パラメータ（model / temperature / max_tokens）はテンプレに書かない | 同系テンプレート共通の方針。モデル ID をテンプレに書かない。定式化の再現性は注入クライアント側の設定責務 |
| 9 | 自然言語経路の LLM 失敗は**失敗**、整形段の LLM 失敗は**縮退** | 自由文 → 線形計画係数の定式化に決定論代替は無い。一方、求解結果の説明は決定論サマリが常に存在する |
| 10 | `formatted_output` は**決定論サマリが先頭・LLM の説明文はその後**（「参考」節として見出しで分離）| 説明文は LLM が書く散文でソルバの数値を再述・改竄しうる（レビューで「最適値は 999」を返す LLM を刺し、正しい 7.0 より先に読まれることを実測）。先に読まれるものを決定論的な事実にし、食い違ったときどちらが正かを出力上に明示する。説明文自体は読み手の理解を助けるので捨てない |
| 11 | Marketplace adapter は scope を持たず、Azure 3接続値を起動時 fail closed にする | tenant/org を消費するコードパスがなく、v0.2.0 part に新 scope API もない。自然言語経路に必須の Azure 接続値だけを allowlist provider 経由で供給する |

---

## 15. 既知の未解決事項（Open Items）

- 非線形 / 二次計画は **スコープ外**（線形 LP/MILP のみ）。将来 Cat 2 派生で検討。
- 超大規模（数千変数）の性能は `max_variables` で抑制。実運用チューニングは派生で。
- ~~変数下限 `low` 省略時のデフォルト~~ → **2026-06-03 決定: 案A（非負=0 既定）**。負値が必須な
  変数は `low` 明示（§4 参照）。
- 大規模 VRP（組合せ最適化）は純 LP/MILP 定式化だと変数数が爆発しうる。`max_variables=200`
  の範囲で扱える小〜中規模に限定する。大規模対応は列生成等を持つ派生で別途。

---

## 16. 変更履歴

| 日付 | 変更 |
|---|---|
| 2026-09-01 | AGENTIC STAR Marketplace adapter の lazy Azure LLM、入力 context 再構築、Azure 接続値 fail-closed、および scope 非実装判断を追記 |
| 2026-06-02 | 初稿（設計フェーズ）。Approach B（構造化スペック + プロセス内 PuLP）確定 |
