# CMN-C1-021 — OptimizationSolverAgent

> **Category**: Cat 1 (generic capability)
> **Industry**: CMN

## Overview

OptimizationSolverAgent solves linear and mixed-integer linear programs (LP / MILP). It accepts
either a structured problem specification or a natural-language description, builds the model
deterministically with PuLP, solves it with CBC, and returns the solver status, the variable
assignment, the objective value and a readable summary.

It suits work where the model is small to medium and the answer has to be reproducible: a
planner exploring resource-allocation or production-mix trade-offs, an application that hands
the agent a structured LP and needs the same answer every time, or an analyst framing an
assignment, shift or knapsack problem as linear constraints.

The division of labour between the language model and the deterministic layer is the point of
the design. The model does one thing only: it transcribes a natural-language problem into a
`problem_spec` — a JSON object of variables, an objective and constraints. Everything that
decides the answer is computed:

- **Generated code is never executed.** The model returns data, not Python. There is no sandbox
  to escape because there is nothing to run.
- **Every specification is validated before the model is built.** Sense, variable names and
  types, bounds, objective and constraint coefficients must all be well-formed and numeric;
  unknown variable references, duplicate names and non-finite values are rejected with
  `SPEC_PARSE_FAILED` rather than passed to the solver.
- **Problem size and solver runtime are bounded.** `max_variables`, `max_constraints`,
  `max_input_chars` and `solver_timeout_s` cap the workload; an oversized problem comes back as
  `PROBLEM_TOO_LARGE`.
- **The deterministic summary comes first.** The formatted output opens with the computed
  result — status, objective value, variable assignment — and any model-written prose follows it
  under a separate heading. Prose can restate a number wrongly; whichever a reader sees first
  should be the computed one.

A structured `problem_spec` needs no language model at all. That path runs end to end without
one — only the explanatory prose is omitted.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Integration

Provide:

- an AgentCore runtime;
- for the natural-language path, a `BaseLLM` implementation injected through `config["llm"]` —
  no provider client is bundled and no model is named in this template. Without one, a
  natural-language call ends as `status=error` / `LLM_NOT_CONFIGURED`: turning free text into
  linear coefficients has no deterministic substitute, so the agent refuses rather than guesses.
  A structured `problem_spec` is unaffected;
- caller authentication in front of the entry adapter.

Runtime parameters read from `config/config.yaml` (defaults in parentheses):

| Key | Meaning |
|---|---|
| `max_variables` (`200`) | Maximum number of variables accepted |
| `max_constraints` (`500`) | Maximum number of constraints accepted |
| `max_input_chars` (`10000`) | Maximum length accepted for the problem text |
| `solver_timeout_s` (`10`) | CBC time limit, in seconds |
| `s3_gate_enabled` (`true`) | Output security gate; not intended to be disabled |
| `mask_patterns` (`[]`) | Output masks applied by the output gate — add labels in a derivative |
| `audit_log_fields` | Field names recorded in the audit event (metadata only) |

Model id, temperature and `max_tokens` belong to the injected client, not to this configuration.
Inject a low-temperature client if reproducible formulation matters.

The template does not solve non-linear, quadratic, stochastic or robust programs, does not run
column generation, and does not fetch data or maintain a knowledge base. Objective and
constraints are linear; variable types are `continuous`, `integer` and `binary`; constraint
operators are `<=`, `>=` and `==`.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent is designed to run on the platform, which supplies the entry adapter, the secret provider and the language-model client. Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Supplied by the platform's own package registry at build time. It is not published on public PyPI, so `pip install -e .` alone does not fetch it. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run on AGENTIC STAR, and the agent does not fall back to a degraded
mode for the parts that need it. The standalone server in `src/api/server.py` does start without
the platform — the manifest declares no required secrets, so nothing fails at start-up — but it
constructs no language-model client. An authenticated `/invoke` carrying only natural-language
text therefore returns `LLM_NOT_CONFIGURED` instead of producing a guessed formulation, while an
authenticated call carrying a structured `problem_spec` runs to completion with the explanatory
prose omitted. Refusing to guess is deliberate — a plausible-looking formulation of the wrong
problem is worse than an error. The test suite runs without a platform connection.

## Input

`POST /invoke` takes a JSON body with `input`, `session_id` and `input_context`. A structured
specification is supplied as a JSON **string** in `input_context.problem_spec`; when it is
absent, the natural-language text in `input` is formulated by the injected model instead.

Callers must be authenticated. The standalone entry adapter starts every request at `ANONYMOUS`
unless upstream middleware stamps a verified trust level; when `INVOKE_AUTH_TOKEN` is set on the
server environment, a matching `Authorization: Bearer` header promotes the caller to
`VERIFIED_EXTERNAL`. A caller that is neither middleware-verified nor token-authenticated is
rejected with **401** — there is no anonymous path into the graph, and the entry node requires
`VERIFIED_EXTERNAL` regardless.

```bash
curl -X POST http://127.0.0.1:8000/invoke \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${INVOKE_AUTH_TOKEN}" \
  -d '{
    "input": "",
    "session_id": "sample-optimization",
    "input_context": {
      "problem_spec": "{\"sense\":\"maximize\",\"variables\":[{\"name\":\"x1\",\"type\":\"continuous\",\"low\":0},{\"name\":\"x2\",\"type\":\"continuous\",\"low\":0}],\"objective\":{\"x1\":1.0,\"x2\":1.0},\"constraints\":[{\"coeffs\":{\"x1\":1.0},\"op\":\"<=\",\"rhs\":4.0},{\"coeffs\":{\"x2\":1.0},\"op\":\"<=\",\"rhs\":3.0}]}"
    }
  }'
```

A variable whose `low` is omitted defaults to `0`, following the LP standard form; an omitted
`up` means unbounded above. State `low` explicitly for variables that may go negative, such as
inventory-balance or network-flow variables.

Input fails closed with stable error codes: `EMPTY_INPUT` (no text and no specification),
`PROBLEM_TOO_LARGE` (over `max_input_chars`, `max_variables` or `max_constraints`),
`SPEC_PARSE_FAILED` (malformed or internally inconsistent specification) and
`S2_INJECTION_DETECTED` (a credential pattern in the problem text or in the specification).

## Output

The standard AgentCore envelope is preserved. `output` carries the formatted result; the same
content and the intermediate fields are also surfaced individually — `formatted_output`,
`solver_status`, `objective_value`, `solution`, `solver_backend`, `variable_count`,
`constraint_count`, `spec_valid`, alongside `status`, `error`, `error_code`, `error_message`,
`trace_id`, `correlation_id`, `session_id`, `node_history` and `error_log`.

`solution` is a JSON string mapping variable names to their values.

```json
{
  "status": "success",
  "solver_status": "OPTIMAL",
  "solution": "{\"x1\": 4.0, \"x2\": 3.0}",
  "objective_value": 7.0,
  "solver_backend": "PULP_CBC_CMD",
  "output": "## 求解結果: OPTIMAL\n最適解が見つかりました\n- 目的関数値: 7.0\n- 変数割当:\n    - x1 = 4.0\n    - x2 = 3.0"
}
```

`solver_status` is one of `OPTIMAL`, `INFEASIBLE`, `UNBOUNDED`, `TIMEOUT` or `UNDEFINED`. None of
these is an error: a problem with no feasible solution is a real, useful answer, so it comes back
as a successful invocation carrying the explanation. Only a failure of the agent itself —
`SOLVER_FAILED`, `LLM_NOT_CONFIGURED`, `LLM_CALL_FAILED`, `S3_BLOCKED` — ends as `status=error`.

The audit event records field names and counts only — `template_id`, `session_id`,
`variable_count`, `constraint_count`, `solver_status`. The problem text and the solution are
never written to the audit trail.

Values above are illustrative; real output depends on the problem supplied.

## Security and Limitations

The entry node requires verified-external trust, so an anonymous caller is stopped before any
node body runs. Both input routes — the natural-language text **and** a directly supplied
`problem_spec` — are scanned for credential patterns, so the structured path cannot be used to
walk around the input gate. On the way out, the output gate masks the configured patterns and
blocks the response with `S3_BLOCKED` if a credential pattern survives masking; the solver
payload is cleared along with the text, so a blocked response leaks nothing through the
numeric fields.

Solver behaviour is guarded where PuLP would otherwise fail silently:

- a `binary` variable whose declared bounds exclude `{0, 1}` is not quietly rewritten to `0..1` —
  the intersection is resolved first, so a specification that cannot be satisfied returns
  `INFEASIBLE` instead of an "optimal" solution that violates the stated bound;
- a variable with an empty domain (`low > up`) returns `INFEASIBLE` without invoking the solver,
  which would otherwise fail as an execution error rather than a solver result;
- CBC is always called with a time limit, so a `NotSolved` result maps to `TIMEOUT` independently
  of whether an incumbent solution exists.

Known limits: formulation from free text is model-assisted and does not guarantee that the
intended constraints and objective were captured — pass a structured `problem_spec` when it
matters. Problem size is capped at `max_variables` / `max_constraints`, which rules out large
vehicle-routing and similar combinatorial models whose variable counts explode; those need a
derivative with column generation or an equivalent method.

Output is machine-generated and assistive. It does not substitute for expert review, and any
decision that matters should be checked against the original problem by a person.

## Release

- Version: 1.0.0 in the agent manifest (`config/agent.yaml`); the Python package metadata in
  `pyproject.toml` still reads `0.1.0`
- Category: single generic capability (Cat 1)
- Generation mode: `llm` for the natural-language route; a structured `problem_spec` is solved
  deterministically
- HITL: not enabled
- Persistence: none — no checkpointer is configured
- Solver: PuLP 2.8.0 with the bundled CBC backend; no external solver licence required
- External API clients: deployment-injected; none bundled

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection, using a fake language model. Running the agent itself
does not: the standalone server in `src/api/server.py` loads `config/config.yaml` and passes it
through unchanged, and it deliberately does **not** supply `config["llm"]` — so a
natural-language `/invoke` against a bare `uvicorn src.api.server:app` returns
`LLM_NOT_CONFIGURED` by design, while a structured `problem_spec` solves normally. A real
language model is supplied by the platform entry adapter, or by a caller that injects a
`BaseLLM` into the configuration.

## Documentation

`docs/02_design.md` carries the node flow, the state schema, the solver and timeout policy and
the security design. `docs/03_test_spec.md` records the test specification and the verification
results, including a live-model run.

## Project Structure

```
src/          agent implementation (nodes, services, schemas)
tests/        unit, integration and boundary tests
config/       agent configuration
docs/         design and test specifications
```

See `docs/02_design.md` for the design and `docs/03_test_spec.md` for the test specification.

## Customising

1. Adjust `config/` for your own environment and policies.
2. Replace the knowledge sources and sample data with your own.
3. Review the node implementations under `src/nodes/` for domain-specific logic.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.

---
