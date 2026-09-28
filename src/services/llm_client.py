"""LLM クライアント抽象と `config["llm"]` 注入シーム。

本 AG が LLM を使うのは 2 箇所だけで、性質が違う:

- `ConstraintParseNode` — 自然言語の問題文 → ソルバ仕様（JSON）への定式化。
  **決定論代替が無い**（自由文から線形計画の係数を導く規則は書けない）ため、
  LLM 未設定は縮退でなく失敗として扱う（`LLM_NOT_CONFIGURED`）。
  なお構造化 `problem_spec` を直接渡す経路は LLM を一切使わない。
- `SolutionFormatNode` — 求解結果の平易な説明文。**決定論サマリが常に存在する**ため、
  LLM 失敗は縮退（説明文なし）で success を維持する。

🔴 旧実装は外側 Graph に `llm_client=` という独自コンストラクタ引数を持っていた。
注入シームは `config["llm"]` だけであるべきで、実際そのとおりの結果になっていた —
framework の実契約（`BaseGraph.__init__(self, config=None)`）に従う `src/api/server.py` は
`OptimizationSolverAgent(config=_config)` としか呼ばないため引数は永久に未設定で、
既定ビルドでは自然言語経路が必ず失敗していた。

プロトコルの差:
- 注入されるのは `BaseLLM`（`complete(messages: list) -> dict`）
- 本 AG のノードが呼ぶのは `generate(prompt: str) -> str`

`BaseLLMAdapter` がこの差を吸収する。**src/ に置く**（テスト側にだけ置くと本番経路に
アダプタが存在せず、テストだけが緑になる — 旧実装がその状態だった）。

モデル ID も具体クライアントもここには書かない。生成パラメータ
（model / temperature / max_tokens）は**注入されるクライアントが保持する**もので、
テンプレ側の config では選択できない。定式化の再現性が要る場合は、注入側で
temperature を低く設定したクライアントを渡すこと。
"""

from __future__ import annotations

from typing import Any


class LLMNotConfiguredError(RuntimeError):
    """LLM が注入されていない状態で LLM 呼び出しが要求された。"""


def _declares(obj: Any, name: str) -> bool:
    """`name` を **実体として持つ**か（クラス階層 or インスタンス属性）。

    `getattr()` を使わないのは 2 つの理由から:
    - SDK の実クライアント（`AnthropicClient` 等）は `__getattr__` で内部 LangChain
      クライアントへ委譲する。委譲先の別シグネチャのメソッドを「自前実装」と誤判定
      すると、文字列 prompt を渡した瞬間に落ちる。
    - `getattr()` は property を **実行してしまう**。property が例外を投げる型を
      渡されると、判定しただけで agent の構築が落ちる。

    インスタンス属性（`self.generate = ...` で組み立てるクライアント）も実体なので
    `__dict__` を併せて見る。
    """
    if name in getattr(obj, "__dict__", {}):
        return True
    return any(name in klass.__dict__ for klass in type(obj).__mro__)


class LLMClient:
    """未注入を表す番兵。`generate()` は必ず失敗する。

    握り潰さず送出するのは、自然言語経路に決定論代替が無いため
    （`ConstraintParseNode` が `LLM_NOT_CONFIGURED` として伝播する）。
    `SolutionFormatNode` 側はこの例外も縮退として扱う。
    """

    def generate(self, prompt: str) -> str:
        raise LLMNotConfiguredError(
            "No LLM is configured. Inject one via config={'llm': <BaseLLM>} "
            "(the platform / entry adapter supplies it at deploy time)."
        )


class BaseLLMAdapter:
    """`BaseLLM.complete(messages) -> dict` を本 AG の `generate(prompt) -> str` へ合わせる。"""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def generate_with_model(self, prompt: str) -> tuple[str, str]:
        """`(text, model)` を返す。model は **応答が名乗ったモデル名**。"""
        response = self._llm.complete([{"role": "user", "content": str(prompt)}])
        return _content_of(response), _model_of(response)

    def generate(self, prompt: str) -> str:
        return self.generate_with_model(prompt)[0]


def _content_of(response: Any) -> str:
    content = response.get("content") if isinstance(response, dict) else response
    if content is None:
        # `{"content": None}` を str() にかけると文字列 "None" が説明文として
        # 利用者に返る。空応答は空文字へ正規化する。
        return ""
    return content if isinstance(content, str) else str(content)


def _model_of(response: Any) -> str:
    """**応答が名乗るモデル**を返す（不明なら空文字）。

    `BaseLLM` の抽象契約は complete / stream / bind_tools だけで public な `model` は
    無い。config の model 文字列は注入クライアントの選択を決めないので、監査に載せる
    モデル名は canonical response の `model` から採る。
    """
    if isinstance(response, dict):
        name = response.get("model")
        if isinstance(name, str) and name:
            return name
    return ""


def resolve_llm_client(injected: Any | None) -> Any:
    """`config["llm"]` の値を本 AG のクライアント契約へ解決する。

    🔴 **`isinstance(BaseLLM)` を duck typing より先に置く。** SDK の実クライアントは
    `__getattr__` 委譲を持つため、`getattr(x, "generate")` がバッチ用の別シグネチャを
    拾ってしまう（`_declares` の docstring 参照）。
    """
    if injected is None:
        return LLMClient()
    try:
        from shared.services.llm.base_llm import BaseLLM

        if isinstance(injected, BaseLLM):
            return BaseLLMAdapter(injected)
    except ImportError:  # pragma: no cover — SDK 不在環境
        pass
    # BaseLLM を継承していない型は、**canonical な `complete` を優先**する。
    # `generate` を先に見ると、LangChain 系のバッチ API（`generate(prompts)`）を
    # 持つクライアントで文字列 prompt を渡して落ちる。`complete(messages) -> dict`
    # が一次契約なので、両方持つ曖昧な型はそちらへ寄せる。
    if _declares(injected, "complete"):
        return BaseLLMAdapter(injected)
    # `complete` を持たない型は、generate 系を**自前で定義している**ものだけ使う。
    for name in ("generate_with_model", "generate"):
        if _declares(injected, name):
            return injected
    return LLMClient()


def generate_text(client: Any, prompt: str) -> tuple[str, str]:
    """`(text, model)` を返す。model が分からないクライアントでは空文字。

    空応答は空文字へ正規化する（`None` を `str()` して "None" を本文に混ぜない）。
    """
    if _declares(client, "generate_with_model"):
        text, model = client.generate_with_model(prompt)
        return ("" if text is None else str(text)), str(model or "")
    raw = client.generate(prompt)
    return ("" if raw is None else str(raw)), ""
