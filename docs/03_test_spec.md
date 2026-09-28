# CMN-C1-021 OptimizationSolverAgent — テスト仕様書

## 1. テスト構成

| 種別 | ファイル | 件数 | 目的 |
|---|---|---|---|
| Unit | `tests/unit/test_constraint_parse.py` | 12 | 入力検証 / LLM 抽出 / 構造化入力 / サイズ上限 / S-2(直接spec) / 非数値早期検出 / エラー伝播 |
| Unit | `tests/unit/test_solver_dispatch.py` | 8 | 実 CBC 求解（OPTIMAL/INFEASIBLE/整数）/ SOLVER_FAILED / 非エラー実行不能 |
| Unit | `tests/unit/test_solution_format.py` | 6 | 自然言語化 / 決定論サマリ / S-3 マスク・ブロック / LLM 失敗時非致命 |
| Integration | `tests/integration/test_optimization_agent.py` | 8 | E2E 求解(invoke) / 実行不能説明 / S-1(VERIFIED_EXTERNAL/INTERNAL/ANONYMOUS) / S-2(spec) / trace / 独立実行 |
| Proof-of-Boundary | `tests/proof_of_boundary/test_pb_placeholder.py` | 16 | PB-1〜PB-7 + ノード契約（execute のみ）+ execute シグネチャ + 全ノード required_trust_level 宣言 |

**合計 50 テスト**（`pytest tests/`、unit 26 + integration 8 + PB 16）。stub-only な placeholder は廃止。
CI は `tests/` と `tests/proof_of_boundary/` を別途実行。

実行コマンド:
```bash
python -m pytest tests/ -v
python -m pytest tests/proof_of_boundary/ -v
```

LLM（`framework.llm.LLMClient`）と framework.* は `tests/conftest.py` のスタブで解決。
ソルバ（PuLP/CBC）は **実ライブラリを実行**（モックしない）。E2E は構造化 `problem_spec` を
直接入力して LLM スタブ非依存で決定論的に検証する。

---

## 2. Framework Compliance TC（framework 契約の必須項目）

| TC-ID | 対応テスト | 期待結果 |
|---|---|---|
| TC-01 State 契約 | PB-2 `test_pb2_state_values_are_primitives` | 出力 state は str/int/float/bool/None のみ |
| TC-02 S-1 拒否 | `test_s1_absent_context_default_deny` / `test_s1_insufficient_trust_rejected`（integration）| 不正 trust は `S1_TRUST_DENIED` で停止（ADR-006: 例外送出でなく ERROR）|
| TC-03 JWT/Credential 検出 | PB-5 `test_pb5_checkpoint_safety` / `test_credential_in_input_blocked` | state に credential 残存なし |
| TC-04 InvocationContext は configurable 経由 | PB-5 / S-1 テスト | context は config 経由、出力 state に残さない |
| TC-05 AuditSink (`_emit_trace_event`) | PB-1 / `test_invoke_lifecycle_trace_emitted` | 全 invoke で 1 イベント発火 |
| TC-06 `_security_gate_input` 非バイパス | `test_credential_in_input_blocked` / S-2 | credential 入力で停止 |
| TC-07 `_security_gate_output` 非バイパス | `test_credential_in_llm_explanation_blocked` | credential 出力で S3_BLOCKED |
| TC-08 `required_trust_level` 強制 | `test_s1_insufficient_trust_rejected` / `test_s1_internal_trust_allowed` | VERIFIED_EXTERNAL 未満は拒否、INTERNAL は許可 |

---

## 3. 機能 TC（OptimizationSolverAgent 固有）

| TC | 内容 | 期待 |
|---|---|---|
| TC-F1 | 既知 LP（max x1+x2 / x1≤4,x2≤3）| OPTIMAL, 目的値 7.0, x1=4,x2=3 |
| TC-F2 | 整数変数（max x / 2x≤7, integer）| OPTIMAL, x=3 |
| TC-F3 | 実行不能（x≥5 かつ x≤4）| INFEASIBLE（エラーにしない）, 自然言語説明 |
| TC-F4 | 自然言語入力 → LLM 構造化（fake LLM）| spec_valid, 変数/制約数一致 |
| TC-F5 | JSON フェンス除去 | ```json``` 包みでも抽出成功 |
| TC-F6 | 構造化 problem_spec 直接入力 | LLM スキップで検証通過 |
| TC-F7 | 空入力 / 過大入力 / 未知変数参照 | EMPTY_INPUT / PROBLEM_TOO_LARGE / SPEC_PARSE_FAILED |
| TC-F8 | PII マスク（mynumber）| `[REDACTED:mynumber]` に置換 |
| TC-F9 | LLM 説明失敗 | 決定論サマリは残り非致命 |

---

## 4. Proof-of-Boundary（docs/02_design.md §12）

| PB-ID | 境界 | 期待結果 |
|---|---|---|
| PB-1 | BaseNode → EventEmitter | `_emit_trace_event` が 1 回発火（silent failure なし）|
| PB-2 | State serialization | 出力 state はスカラ primitive のみ |
| PB-3 | SolverDispatch → 外部ソルバ（実 CBC）| OPTIMAL, backend=PULP_CBC_CMD, 実値一致 |
| PB-4 | Import isolation | src/ に `agenticstar` import なし（AST scan 0 件）|
| PB-5 | Checkpoint safety | 出力に invocation_context / `sk-` なし |
| PB-6 | Invoke 実行順序 | pre→in→run→out→trace |
| PB-7（固有）| 任意コード実行なし | coeffs のコード文字列は float 変換で SOLVER_FAILED、未実行 |
| ノード契約 | ノードは execute() のみ | `_invoke_impl` を実装しない（全3ノード）|
| ノード契約（F-02）| execute シグネチャ | `execute(self, state)` を全ノードで検証（v1.0.0 の BaseNode.__call__ は state 1 引数のみで呼ぶ。`test_execute_method_signature`）|

---

## 5. 実行結果

| 環境 | 結果 |
|---|---|
| ローカル（Python 3.11.13 + pulp 2.8.0 / CBC）| ✅ 50 passed（tests/ 全体）/ 16 passed（PB）。check スクリプト3本 PASS |
| CI（push pipeline, Python 3.11）| ✅ 全8ジョブ green（gate 系7 + run-tests 50/16）|

**レビュー所見トレーサビリティ**: F-02（execute シグネチャ契約）= `test_execute_method_signature`（PB）。
F-04（`SPEC_PARSE_FAILED` はリトライせず即時終了）= `test_invalid_llm_json` / `test_non_numeric_*`（unit）+ `docs/07 §7`。
F-01（自然言語入力の精度非保証）= `docs/07 §7` に明記。
| CI（Python 3.11 + `.[dev]`）| 実装フェーズのパイプラインで確認 |

> ローカル CBC 求解確認済み（`status Optimal`）。CI（python:3.11-slim）でも PuLP 同梱 CBC が動作する想定。

---

## 6. 変更履歴

| 日付 | 変更 |
|---|---|
| 2026-06-02 | 初稿（テストフェーズ）。39 TC + ローカル 100% PASS |

## 実 LLM 1 回転検証

モックだけで固めた実装は実 LLM を繋いだ瞬間に動かないことがあるため、注入シームへ実 LLM クライアントを刺して 1 回 invoke し、**中身のある出力が出ること**を確認している（`gitproject-dev-flow.md` §14-1-bis）。

| 項目 | 内容 |
|---|---|
| 実施日 | 2026-08-28 |
| 使用 LLM | Azure OpenAI のモデルを `BaseLLM` 実装として注入 |
| 注入シーム | `config[llm]` |
| 実 LLM 呼び出し | 2 回 |
| 出力 | ['<text output: 229字>'] が非空。縮退理由なし |

本番の LLM は注入されたクライアントで、これは検証用シーム。テンプレの `src/` は変更していない（一時テストを生成→実行→削除する方式）。CI には入れていない（外部モデルの鍵が要るため）。

**出力の並び（H3 裁定 = 案②）**: この 1 回転の実出力でも、決定論サマリ
（`## 求解結果: OPTIMAL` / 目的関数値 7.0 / x1=4.0, x2=3.0）が先頭に来て、LLM の説明文が
`## 参考: 言語モデルによる説明` 節として後ろに付くことを目視で確認した。回帰は
`tests/unit/test_solution_format.py` の 3 件（矛盾する数値を返す LLM で順序をリテラル比較 /
参考節の見出しと注記 / 縮退時に参考節が付かないこと）。

### 業務入力での追加確認（2026-08-28）

汎用ペイロードは「invoke 経路が生きている」までしか示さないため、**この AG の中核能力
（自然言語 → ソルバ仕様の定式化）** が実 LLM で成立するかを、手計算で答えが分かる問題で
別途確認した（注入シームは同じ `config["llm"]`。repo 外で CLI を起動しソースを読ませない）。

| ケース | 入力 | 実 LLM の定式化 | 結果 | 判定 |
|---|---|---|---|---|
| 生産計画 LP | 利益 300/500 円、機械時間 2h/4h・上限 40h、材料 1kg/1kg・上限 12kg の利益最大化 | 変数 2・制約 2（機械時間 / 材料）| `OPTIMAL` / 目的関数値 **5200.0** / A=4.0, B=8.0 | ✅ 手計算の最適解と一致 |
| 上下限のみの LP | 「x1 は最大 4、x2 は最大 3、合計を最大化」 | 変数 2（`up` に畳む）・制約 0 | `OPTIMAL` / 目的関数値 **7.0** | ✅ 一致（制約を変数境界に畳むのは妥当な等価変換）|
| 実行不能 | 「x は 5 以上かつ 4 以下」 | 変数 1（`low=5, up=4`）・制約 0 | `INFEASIBLE`（success として説明）| ✅ 修正後。**修正前は `SOLVER_FAILED`**（下記）|

**この検証で出た欠陥と是正:** 3 件目で実 LLM は矛盾する上下限を **変数境界**（`low=5, up=4`）に
畳んだ。この spec を CBC に渡すと求解ではなく**ソルバ実行そのものが失敗**し、正常な求解結果で
あるべき `INFEASIBLE` が `SOLVER_FAILED` になり、さらに例外文に含まれる**ソルバ実行ファイルの
サーバ側絶対パス**が `error` / `error_message` として呼び出し元まで返っていた。
→ `solver_service.solve()` が空の定義域を求解前に `INFEASIBLE` として返すようにし、
`SolverDispatchNode` の予期せぬ例外は**型名のみ**を理由に載せるよう是正した
（回帰: `tests/unit/test_solver_dispatch.py` の空定義域 2 件 + 例外文非永続化 1 件）。
