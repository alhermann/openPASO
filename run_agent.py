#!/usr/bin/env python3
"""Run one simulation task with an AI model of your choice.

This is the simplest way to use openPASO. You need two things:
an OpenRouter API key, and a description of what you want to simulate.

    cp .env.example .env        # then paste your key into .env
    python run_agent.py "Solve the Poisson equation on a unit square
                         and check the convergence rate"

The model reads your sentence, picks a solver, writes the input file,
runs the solver, and checks the answer. Everything it does is printed
as it happens, so you can watch and stop it at any time with Ctrl-C.

If you would rather use openPASO from Claude Code, Claude Desktop or
Cursor, you do not need this file at all. See "Start here" in README.md.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
OPENROUTER_URL = "https://openrouter.ai/api/v1"


def load_env_file(path: Path) -> None:
    """Read KEY=value lines from .env into the environment.

    Kept deliberately small so that openPASO needs no extra package just to
    read one file. A value already set in the shell always wins, because an
    explicit choice should not be overridden by a file.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if value and key not in os.environ:
            os.environ[key] = value


def explain_and_exit(problem: str, fix: str) -> None:
    print(f"\n  openPASO cannot start.\n\n  Problem: {problem}\n  Fix:     {fix}\n",
          file=sys.stderr)
    raise SystemExit(1)


def build_model(model_id: str, api_key: str, temperature: float):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        base_url=OPENROUTER_URL,
        api_key=api_key,
        model=model_id,
        temperature=temperature,
        timeout=600,
    )


async def run(task: str, *, model_id: str, api_key: str, workdir: Path | None,
              temperature: float, step_limit: int) -> int:
    # LangGraph 1.0 moved this factory and renamed its prompt argument.
    try:
        from langchain.agents import create_agent
        prompt_kwarg = "system_prompt"
    except ImportError:
        from langgraph.prebuilt import create_react_agent as create_agent
        prompt_kwarg = "prompt"

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "src"))
    from core.instructions import INSTRUCTIONS
    from langgraph_eval.agent import openpaso_mcp_tools_session

    print(f"  model    {model_id}", flush=True)
    print(f"  task     {task}", flush=True)
    print("  starting the openPASO server ...\n", flush=True)

    async with openpaso_mcp_tools_session(workdir, surface="all") as tools:
        print(f"  {len(tools)} solver tools ready\n" + "─" * 72, flush=True)
        agent = create_agent(build_model(model_id, api_key, temperature),
                             tools=tools, **{prompt_kwarg: INSTRUCTIONS})
        final = None
        async for step in agent.astream(
                {"messages": [("user", task)]},
                {"recursion_limit": step_limit},
                stream_mode="values"):
            message = step["messages"][-1]
            final = message
            kind = getattr(message, "type", "")
            for call in getattr(message, "tool_calls", None) or []:
                print(f"  → {call['name']}({_short(call.get('args'))})", flush=True)
            if kind == "tool":
                # The solver's own answer goes to the model, not to the screen;
                # a raw tool payload is long and is not written for a reader.
                lines = str(getattr(message, "content", "")).count("\n") + 1
                print(f"    ← {message.name}: {lines} line(s)", flush=True)
                continue
            text = getattr(message, "content", "")
            if text and kind == "ai" and not getattr(message, "tool_calls", None):
                print(f"\n{text}\n", flush=True)
        print("─" * 72)
        print("  done" if final is not None else "  the model returned nothing")
    return 0


def _short(args, width: int = 90) -> str:
    text = ", ".join(f"{k}={v!r}" for k, v in (args or {}).items())
    return text if len(text) <= width else text[:width - 3] + "..."


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run one simulation task with openPASO and an AI model.",
        epilog='example:  python run_agent.py "Solve Poisson on a unit square '
               'and verify the convergence rate"')
    ap.add_argument("task", nargs="*", help="what you want to simulate, in normal words")
    ap.add_argument("--model", help="model id; overrides OPENPASO_MODEL from .env")
    ap.add_argument("--workdir", type=Path,
                    help="folder for this run's files (default: the current folder)")
    ap.add_argument("--temperature", type=float, default=0.2,
                    help="0.0 is the most repeatable, 1.0 the most varied (default 0.2)")
    ap.add_argument("--step-limit", type=int, default=200,
                    help="stop after this many agent steps (default 200)")
    args = ap.parse_args()

    load_env_file(REPO / ".env")

    task = " ".join(args.task).strip()
    if not task:
        explain_and_exit("you did not say what to simulate",
                         'put the task in quotes, for example:\n'
                         '           python run_agent.py "Solve Poisson on a unit square"')

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        explain_and_exit(
            "OPENROUTER_API_KEY is empty",
            "run  cp .env.example .env  and paste your key from\n"
            "           https://openrouter.ai/keys  into the .env file")

    model_id = (args.model or os.environ.get("OPENPASO_MODEL", "")).strip()
    if not model_id:
        explain_and_exit(
            "no model chosen",
            "set OPENPASO_MODEL in .env, or pass --model.\n"
            "           Pick one from https://openrouter.ai/models")

    try:
        return asyncio.run(run(task, model_id=model_id, api_key=api_key,
                               workdir=args.workdir,
                               temperature=args.temperature,
                               step_limit=args.step_limit))
    except KeyboardInterrupt:
        print("\n  stopped by you")
        return 130
    except ModuleNotFoundError as missing:
        explain_and_exit(
            f"a Python package is missing: {missing.name}",
            "install the agent packages with\n"
            "           pip install -r langgraph_eval/requirements-langgraph.txt")
    except BaseException as error:               # noqa: BLE001 - reported below
        explain_and_exit(*_diagnose(error))
    return 1


def _diagnose(error: BaseException) -> tuple[str, str]:
    """Turn whatever went wrong into one sentence and one instruction.

    Exceptions arrive wrapped in an ExceptionGroup, because the MCP session
    and the agent run in the same task group, so the text is searched rather
    than the type matched.
    """
    def unwrap(exc, depth=0):
        """Every message in the chain: groups, causes and contexts alike."""
        if exc is None or depth > 8:
            return []
        found = [f"{exc.__class__.__name__}: {exc}"]
        for sub in getattr(exc, "exceptions", ()):       # ExceptionGroup
            found += unwrap(sub, depth + 1)
        found += unwrap(exc.__cause__, depth + 1)
        found += unwrap(exc.__context__, depth + 1)
        return found

    messages = unwrap(error)
    text = " | ".join(dict.fromkeys(messages))
    low = text.lower()
    if "401" in low or "no auth" in low or "invalid api key" in low:
        return ("OpenRouter rejected the key",
                "check OPENROUTER_API_KEY in your .env file against\n"
                "           https://openrouter.ai/keys")
    if "402" in low or "credit" in low or "quota" in low:
        return ("your OpenRouter account is out of credit",
                "add credit at https://openrouter.ai/credits")
    if "404" in low and "model" in low:
        return (f"OpenRouter does not know the model id",
                "copy an id exactly from https://openrouter.ai/models\n"
                "           and put it in OPENPASO_MODEL in your .env file")
    if "does not support tool" in low or "tool use" in low:
        return ("the chosen model cannot use tools",
                "pick a model listed with tool support on\n"
                "           https://openrouter.ai/models")
    if "no interpreter found" in low:
        return ("the openPASO server has no Python to run in",
                "create it with  python3 -m venv .venv && .venv/bin/pip install -e .\n"
                "           or set OPENPASO_PYTHON in your .env file")
    specific = [m for m in messages if "TaskGroup" not in m]
    return ((specific[-1] if specific else text)[:400] or error.__class__.__name__,
            "if this is not clear, open an issue and paste this message")


if __name__ == "__main__":
    raise SystemExit(main())
