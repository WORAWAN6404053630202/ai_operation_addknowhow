#/Users/w.worawan/Downloads/ai-operation-microservice3_v2ori/code/main.py
"""
Local entrypoint for running PersonaSupervisor in CLI mode

Enterprise CLI baseline:
- Deterministic greet on session start via supervisor.handle(user_input="")
- Shows minimal debug state each turn (last_action / fsm_state / pending_slot)
- Provides 'reset' to wipe current session state (avoid stale state confusion)
"""

import logging
import re
import threading
import uuid
from rich.console import Console

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
# Suppress verbose background-thread logs (reranker/embedding model loading)
# that would appear mid-prompt and corrupt the Rich terminal display.
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.WARNING)
logging.getLogger("utils.reranker").setLevel(logging.WARNING)
from rich.markup import escape as rich_escape
from rich.prompt import Prompt

from model.state_manager import StateManager
from model.conversation_state import ConversationState
from model.persona_supervisor import PersonaSupervisor
from service.local_vector_store import get_retriever

console = Console()

_BARE_URL_RE = re.compile(r'(?<![(\[])https?://\S+')

def _format_reply(text: str) -> str:
    """
    Prepare bot reply for Rich console.print():
    1. Escape all Rich markup characters in the text
    2. Convert bare URLs to Rich OSC-8 hyperlinks [link=URL]URL[/link]
    Result: compact display (no Markdown blank-line spacing) + clickable URLs.
    """
    if not text:
        return text
    escaped = rich_escape(text)
    def _replace(m: re.Match) -> str:
        url = m.group(0).rstrip(".,;:)\"'")
        # After rich_escape, brackets are escaped — match on original URL text
        escaped_url = rich_escape(url)
        return f"[link={url}]{escaped_url}[/link]"
    return _BARE_URL_RE.sub(_replace, escaped)


def create_initial_state(session_id: str) -> ConversationState:
    return ConversationState(
        session_id=session_id,
        persona_id="practical",
        context={},
    )


def _debug_line(state: ConversationState) -> str:
    ctx = state.context or {}
    fsm = (ctx.get("fsm_state") or "").strip() or "-"
    last_action = (getattr(state, "last_action", None) or "-").strip() if isinstance(getattr(state, "last_action", None), str) else (getattr(state, "last_action", None) or "-")
    pending = ctx.get("pending_slot")
    if isinstance(pending, dict):
        key = pending.get("key") or "-"
        opts = pending.get("options") or []
        try:
            nopts = len(opts)
        except Exception:
            nopts = "?"
        pending_txt = f"{key}({nopts})"
    else:
        pending_txt = "-"

    did_greet = bool(ctx.get("did_greet"))
    greet_streak = ctx.get("greet_streak", "-")
    return f"[dim]DEBUG[/dim] last_action={last_action} | fsm={fsm} | pending_slot={pending_txt} | did_greet={did_greet} | greet_streak={greet_streak}"


def main():
    console.rule("[bold cyan]Restbiz Persona AI Local CLI[/bold cyan]")

    session_id = str(uuid.uuid4())[:8]
    console.print(f"[dim]Session ID:[/dim] {session_id}")

    state_manager = StateManager()

    retriever = get_retriever(fail_if_empty=True)
    supervisor = PersonaSupervisor(retriever=retriever)

    import conf as _conf
    if getattr(_conf, "RERANKER_ENABLED", False):
        _rr_model = getattr(_conf, "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
        def _preload_reranker() -> None:
            try:
                from utils.reranker import _get_reranker
                _get_reranker(_rr_model)
            except Exception as _e:
                logging.getLogger(__name__).warning("[Startup] Reranker preload failed: %s", _e)
        threading.Thread(target=_preload_reranker, daemon=True, name="reranker-preload").start()

    state = state_manager.load(session_id)
    if state is None:
        state = create_initial_state(session_id)
        state_manager.save(session_id, state)
        console.print("[dim]State:[/dim] new session created")
    else:
        console.print("[yellow]Loaded existing session state (unexpected for new session_id).[/yellow]")

    # Deterministic greet on startup
    state, greet = supervisor.handle(state=state, user_input="")
    state_manager.save(session_id, state)

    if greet:
        console.print(f"\n[bold magenta]Assistant[/bold magenta]:")
        console.print(_format_reply(greet))
        console.print("")
    console.print(_debug_line(state))
    console.print("[green]System ready. Type 'exit' to quit. Type 'reset' to restart this session.[/green]\n")

    while True:
        try:
            user_input = Prompt.ask("[bold blue]You[/bold blue]")

            cmd = user_input.strip().lower()
            if cmd in {"exit", "quit"}:
                console.print("\n[bold yellow]Session ended.[/bold yellow]")
                state_manager.delete(session_id)
                break

            if cmd == "reset":
                console.print("\n[bold yellow]Resetting session state...[/bold yellow]")
                state = create_initial_state(session_id)
                state_manager.save(session_id, state)
                state, greet = supervisor.handle(state=state, user_input="")
                state_manager.save(session_id, state)
                if greet:
                    console.print(f"\n[bold magenta]Assistant[/bold magenta]:")
                    console.print(_format_reply(greet))
                    console.print("")
                console.print(_debug_line(state))
                console.print("")
                continue

            state, reply = supervisor.handle(state=state, user_input=user_input)
            state_manager.save(session_id, state)

            console.print(f"\n[bold magenta]Assistant[/bold magenta]:")
            console.print(_format_reply(reply))
            console.print("")
            console.print(_debug_line(state))
            console.print("")

        except KeyboardInterrupt:
            console.print("\n[red]Interrupted by user.[/red]")
            # keep state file for postmortem; do not delete automatically
            break
        except Exception as e:
            import traceback
            console.print(f"[red]Unexpected error:[/red] {e}")
            console.print(f"[red]{traceback.format_exc()}[/red]")


if __name__ == "__main__":
    main()