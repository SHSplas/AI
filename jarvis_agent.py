"""
JARVIS AGENT - Local Autonomous AI Editor
Reads your project files, talks offline, and can run local commands.

HOW IT WORKS:
  You type a task in plain English
  Agent uses your local PyTorch model
  It answers offline and can run or queue PowerShell commands
  File edit blocks still get confirmation plus backups

COMMANDS:
  /files           list project text/code files
  /models          list saved checkpoints
  /status          show live training/log status
  /read <file>     show a file's contents
  /run <file>      run a Python file and see output
  /exec <command>  run a PowerShell command now
  /queue <command> queue a PowerShell command for jarvis_executor.py
  /undo            restore last backup
  /clear           clear conversation history
  /quit            exit
"""

# ── Auto-install dependencies ─────────────────────────────────────────────────
import subprocess
import sys


def _install(pkg: str):
    print(f"  Installing {pkg}...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
        check=True,
    )


REQUIRED = {"colorama": "colorama", "torch": "torch"}
for import_name, pip_name in REQUIRED.items():
    try:
        __import__(import_name)
    except ImportError:
        _install(pip_name)

# ── Imports ───────────────────────────────────────────────────────────────────
import json
import os
import re
import shutil
import textwrap
from datetime import datetime
from pathlib import Path

from colorama import Fore, Style, init as colorama_init
from jarvis_executor import execute_powershell_command

colorama_init(autoreset=True)
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except AttributeError:
    pass

# ── Config ────────────────────────────────────────────────────────────────────
# Local model checkpoint. You can override with JARVIS_LOCAL_CKPT.
LOCAL_MODEL_CKPT = os.environ.get(
    "JARVIS_LOCAL_CKPT",
    str(Path("Models") / "cpu_gpt_jarvis_v6_guarded_best.pth"),
)
MAX_TOKENS     = 4056
BACKUP_DIR     = Path(".jarvis_backups")
MAX_FILE_BYTES = 80_000   # skip files larger than this (likely binary / huge)
MAX_DISCOVERED_FILES = 120
COMMAND_QUEUE_FILE = Path("command_queue.txt")
SPEAK_REPLIES = os.environ.get("JARVIS_SPEAK", "").strip().lower() in {"1", "true", "yes", "on"}

# Extensions the agent will read and edit
CODE_EXTENSIONS = {
    ".py", ".txt", ".json", ".md", ".csv", ".cfg", ".ini", ".yaml", ".yml", ".toml"
}

# Never read these (binary / too large / secrets)
SKIP_FILENAMES = {
    ".jarvis_key",
}
SKIP_EXTENSIONS = {".pth", ".pt", ".bin", ".pkl", ".exe", ".dll", ".zip", ".rar"}
SKIP_DIR_NAMES = {
    ".git", ".jarvis_backups", "__pycache__", ".pytest_cache",
    "Models", "metrics", "temp", "venv", ".venv", "node_modules",
}

# ── Colours ───────────────────────────────────────────────────────────────────
def c(text, color): return f"{color}{text}{Style.RESET_ALL}"
def green(t):  return c(t, Fore.GREEN)
def yellow(t): return c(t, Fore.YELLOW)
def red(t):    return c(t, Fore.RED)
def cyan(t):   return c(t, Fore.CYAN)
def bold(t):   return c(t, Style.BRIGHT)


# ── File Discovery ────────────────────────────────────────────────────────────
def discover_files(root: Path) -> list[Path]:
    files = []
    for f in sorted(root.rglob("*")):
        if any(part in SKIP_DIR_NAMES for part in f.relative_to(root).parts[:-1]):
            continue
        if not f.is_file():
            continue
        if f.name in SKIP_FILENAMES:
            continue
        if f.suffix.lower() in SKIP_EXTENSIONS:
            continue
        if f.suffix.lower() not in CODE_EXTENSIONS:
            continue
        try:
            if f.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        files.append(f)
        if len(files) >= MAX_DISCOVERED_FILES:
            break
    return files


def display_path(path: Path, root: Path) -> str:
    """Return a stable project-relative path for UI and prompts."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def resolve_user_path(root: Path, user_path: str) -> Path:
    """Resolve a user supplied path while blocking writes outside the project."""
    candidate = (root / user_path.strip().strip('"')).resolve()
    root_resolved = root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"Path is outside the project folder: {user_path}") from exc
    return candidate


def read_file_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"[Could not read file: {e}]"


def build_file_context(root: Path, files: list[Path]) -> str:
    """Build a string of all project files for the system prompt."""
    parts = []
    for f in files:
        content = read_file_safe(f)
        parts.append(f"### FILE: {display_path(f, root)}\n```\n{content}\n```")
    return "\n\n".join(parts)


# ── Backup ────────────────────────────────────────────────────────────────────
_last_backup: dict[str, str] = {}   # filename → original content


def backup_file(path: Path):
    """Save file content in memory and in .jarvis_backups/"""
    BACKUP_DIR.mkdir(exist_ok=True)
    content = read_file_safe(path)
    _last_backup[path.name] = content
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"{path.stem}_{ts}{path.suffix}"
    backup_path.write_text(content, encoding="utf-8")


def undo_last_backup():
    if not _last_backup:
        print(yellow("  Nothing to undo yet."))
        return
    for name, content in _last_backup.items():
        target = Path(name)
        target.write_text(content, encoding="utf-8")
        print(green(f"  ✓ Restored {name}"))
    _last_backup.clear()


# ── Parse local model file edits ──────────────────────────────────────────────
EDIT_RE = re.compile(
    r'<file\s+name=["\']([^"\']+)["\']\s*>(.*?)</file>',
    re.DOTALL | re.IGNORECASE,
)


def parse_edits(response_text: str) -> list[tuple[str, str]]:
    """Extract (filename, new_content) pairs from the model response."""
    edits = []
    for match in EDIT_RE.finditer(response_text):
        fname = match.group(1).strip()
        content = match.group(2).strip()
        # Strip wrapping code fences if the model added them
        content = re.sub(r'^```[a-z]*\n?', '', content)
        content = re.sub(r'\n?```$', '', content)
        edits.append((fname, content))
    return edits


def apply_edits(edits: list[tuple[str, str]], root: Path):
    if not edits:
        return
    print(f"\n{bold('━━  Proposed file edits  ━━')}")
    for fname, content in edits:
        lines = content.count('\n') + 1
        print(f"  • {cyan(fname)}  ({lines} lines)")

    confirm = input(f"\n{yellow('Apply these edits? (Y/N): ')}").strip().upper()
    if confirm != "Y":
        print(red("  Edits cancelled."))
        return

    for fname, content in edits:
        target = root / fname
        backup_file(target) if target.exists() else None
        target.write_text(content, encoding="utf-8")
        print(green(f"  ✓ Saved {fname}"))

    print(green("\n  All edits applied. Use /undo to revert."))


# ── Run a Python file ─────────────────────────────────────────────────────────
def run_python(filepath: Path, timeout: int = 30) -> str:
    try:
        result = subprocess.run(
            [sys.executable, str(filepath)],
            capture_output=True, text=True, timeout=timeout,
            cwd=filepath.parent,
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        parts = []
        if out:
            parts.append(f"STDOUT:\n{out[:2000]}")
        if err:
            parts.append(f"STDERR:\n{err[:1000]}")
        return "\n".join(parts) if parts else "(ran with no output)"
    except subprocess.TimeoutExpired:
        return "Timed out after 30 seconds."
    except Exception as e:
        return f"Error running file: {e}"


def run_executor_command(command: str) -> str:
    """Run a PowerShell command through the same function used by jarvis_executor.py."""
    result = execute_powershell_command(command)
    parts = []
    status = "success" if result["success"] else "failed"
    parts.append(f"Command {status}: {command}")
    if result["output"].strip():
        parts.append("Output:\n" + result["output"].strip())
    if result["error"].strip():
        parts.append("Error:\n" + result["error"].strip())
    return "\n".join(parts)


def queue_executor_command(command: str) -> None:
    """Append a PowerShell command for jarvis_executor.py to run."""
    command = command.strip()
    if not command:
        raise ValueError("No command provided.")
    with COMMAND_QUEUE_FILE.open("a", encoding="utf-8") as f:
        f.write(command + "\n")


def speak_text(text: str) -> None:
    """Speak a short reply using pyttsx3 when available."""
    try:
        import pyttsx3
    except ImportError:
        print(yellow("  Voice output needs pyttsx3. Install with: pip install pyttsx3"))
        return

    spoken = EDIT_RE.sub("", text)
    spoken = re.sub(r"`([^`]+)`", r"\1", spoken)
    spoken = re.sub(r"\s+", " ", spoken).strip()
    if not spoken:
        return
    try:
        engine = pyttsx3.init()
        engine.say(spoken[:1000])
        engine.runAndWait()
    except Exception as e:
        print(yellow(f"  Voice output failed: {e}"))


def listen_for_command(timeout: int = 6, phrase_time_limit: int = 12) -> str | None:
    """Listen once from the microphone using SpeechRecognition when available."""
    try:
        import speech_recognition as sr
    except ImportError:
        print(yellow("  Voice input needs SpeechRecognition and a microphone backend."))
        print(yellow("  Try: pip install SpeechRecognition pyaudio"))
        return None

    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print(yellow("  Listening..."))
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        text = recognizer.recognize_google(audio)
        print(green(f"  Heard: {text}"))
        return text.strip()
    except Exception as e:
        print(red(f"  Voice input failed: {e}"))
        return None


def command_from_plain_english(text: str) -> str | None:
    """Small offline intent layer for common execution requests."""
    low = text.lower().strip()
    if low.startswith(("execute ", "run command ", "powershell ")):
        return re.sub(r"^(execute|run command|powershell)\s+", "", text, flags=re.I).strip()
    if low in {"open notepad", "start notepad", "launch notepad"}:
        return "Start-Process notepad.exe"
    if low in {"open calculator", "start calculator", "launch calculator", "open calc"}:
        return "Start-Process calc.exe"
    if low in {"open google", "start google", "launch google"}:
        return 'Start-Process "https://www.google.com"'
    if low in {"open colab", "open google colab", "start colab", "launch colab"}:
        return 'Start-Process "https://colab.research.google.com"'
    if low in {
        "start cpu training",
        "run cpu training",
        "train on cpu",
        "start local training",
        "run local training",
    }:
        return 'powershell -NoProfile -ExecutionPolicy Bypass -File ".\\train_jarvis_from_scratch.ps1" -Steps 1000'
    if low in {"list files", "show files", "show folder files"}:
        return "Get-ChildItem"
    return None


# ── System prompt ─────────────────────────────────────────────────────────────
def build_system_prompt(root: Path, files: list[Path]) -> str:
    file_context = build_file_context(root, files)
    return f"""You are an AI coding assistant for a local AI project called Jarvis.
The user has a custom GPT model written in PyTorch with a custom BPE tokenizer.
You have full access to their project files shown below.

CAPABILITIES:
- Explain code, find bugs, suggest improvements
- Write new code or rewrite existing files
- When you want to edit a file, wrap its COMPLETE new content like this:
  <file name="filename.py">
  ...full file content...
  </file>
- Only output <file> blocks for files you actually changed
- Always explain what you changed and why

RULES:
- Dont ever be an idot
- alwas remember you can edt files
- Never delete the user's best model file (cpu_gpt_jarvis_big_v1_best.pth)
- Backups are handled automatically — tell the user they can /undo
- Be concise but complete — include the ENTIRE file content in edits, not just diffs
- If there are errors, fix them proactively

PROJECT FILES:
{file_context}
"""


# ── Local model call ──────────────────────────────────────────────────────────
def call_local_model(
    local_model,
    system_prompt: str,
    conversation: list[dict],
) -> str:
    """
    Return a reply from the locally stored PyTorch Jarvis model.

    The local model has a compact context window, so file access
    and execution are handled by deterministic helper commands in this agent.
    """
    if not conversation:
        return local_model.reply("Hello")
    return local_model.reply(conversation[-1]["content"])


# ── Pretty-print response ─────────────────────────────────────────────────────
def print_response(text: str):
    # Hide raw <file> blocks from display (they'll be applied separately)
    display = EDIT_RE.sub(
        lambda m: f"  {green('✎')} {cyan(m.group(1))} — edit ready to apply",
        text
    )
    print(f"\n{bold('Jarvis Agent:')} {display}\n")


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    global SPEAK_REPLIES
    from local_jarvis_model import LocalJarvisModel

    root = Path(".").resolve()
    print(f"\n{bold(cyan('+--------------------------------------+'))}")
    print(f"{bold(cyan('|       JARVIS AGENT  -  Local         |'))}")
    print(f"{bold(cyan('+--------------------------------------+'))}")
    print(f"  Project folder: {root}")
    print(yellow("  Loading local Jarvis model..."))
    local_model = LocalJarvisModel(LOCAL_MODEL_CKPT)
    print(f"  Powered by:     {green('Local PyTorch model')}")
    print(f"  Checkpoint:     {cyan(local_model.metadata['checkpoint'])}")
    print(f"  Training step:  {green(str(local_model.metadata['step']))}")
    print(f"  Best val loss:  {green(str(local_model.metadata['best_val']))}")
    print(f"  Voice replies:  {green('on') if SPEAK_REPLIES else yellow('off')}")

    # Discover files
    files = discover_files(root)
    print(f"  Files loaded:   {green(str(len(files)))}")
    for f in files:
        print(f"    {cyan('-')} {f.name}  ({f.stat().st_size // 1024 + 1} KB)")

    print(f"\n  Type a task, or {yellow('/help')} for commands.\n")
    print("-" * 50)

    conversation: list[dict] = []
    system_prompt = build_system_prompt(root, files)

    while True:
        try:
            user_input = input(f"\n{bold('You:')} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{yellow('Exiting.')}")
            break

        if not user_input:
            continue

        # ── Built-in commands ─────────────────────────────────────────────
        low = user_input.lower()

        if low in {"/quit", "/exit", "exit", "quit"}:
            print(yellow("see ya"))
            break

        if low == "/help":
            print(__doc__)
            continue

        if low == "/listen":
            heard = listen_for_command()
            if not heard:
                continue
            user_input = heard
            low = user_input.lower()

        if low.startswith("/speak"):
            arg = low.replace("/speak", "", 1).strip()
            if arg in {"on", "yes", "true", "1"}:
                SPEAK_REPLIES = True
                print(green("  Spoken replies enabled."))
            elif arg in {"off", "no", "false", "0"}:
                SPEAK_REPLIES = False
                print(yellow("  Spoken replies disabled."))
            else:
                print(f"  Spoken replies are {'on' if SPEAK_REPLIES else 'off'}. Use /speak on or /speak off.")
            continue

        if low == "/files":
            files = discover_files(root)
            for f in files:
                sz = f.stat().st_size
                print(f"  {cyan(f.name):40s} {sz:>8,} bytes")
            continue

        if low.startswith("/read "):
            fname = user_input[6:].strip()
            target = root / fname
            if not target.exists():
                print(red(f"  File not found: {fname}"))
            else:
                print(f"\n{bold(fname)}:\n")
                print(read_file_safe(target))
            continue

        if low.startswith("/run "):
            fname = user_input[5:].strip()
            target = root / fname
            if not target.exists():
                print(red(f"  File not found: {fname}"))
            else:
                print(yellow(f"  Running {fname}..."))
                output = run_python(target)
                print(output)
                # Feed the output into conversation so agent knows about it
                user_input = f"I ran {fname}. Here's the output:\n{output}\nPlease analyse it."
            # fall through to send to the local model

        if low.startswith("/exec "):
            command = user_input[6:].strip()
            try:
                output = run_executor_command(command)
                print(output)
                if SPEAK_REPLIES:
                    speak_text(output)
            except Exception as e:
                print(red(f"  Executor command failed: {e}"))
            continue

        if low.startswith("/queue "):
            command = user_input[7:].strip()
            try:
                queue_executor_command(command)
                print(green(f"  Queued for separate executor: {command}"))
            except Exception as e:
                print(red(f"  Could not queue command: {e}"))
            continue

        if low == "/undo":
            undo_last_backup()
            continue

        if low == "/clear":
            conversation.clear()
            local_model.reset()
            system_prompt = build_system_prompt(root, discover_files(root))
            print(green("  Conversation cleared and files reloaded."))
            continue

        inferred_command = command_from_plain_english(user_input)
        if inferred_command:
            confirm = input(
                f"\n{yellow(f'Run this PowerShell command? {inferred_command} (Y/N): ')}"
            ).strip().upper()
            if confirm == "Y":
                output = run_executor_command(inferred_command)
                print(output)
                if SPEAK_REPLIES:
                    speak_text(output)
            else:
                print(yellow("  Command not run."))
            continue

        # ── Refresh file context if files changed ─────────────────────────
        current_files = discover_files(root)
        if len(current_files) != len(files):
            files = current_files
            system_prompt = build_system_prompt(root, files)

        # ── Send to local model ───────────────────────────────────────────
        conversation.append({"role": "user", "content": user_input})

        print(yellow("\n  Thinking..."))
        try:
            reply = call_local_model(local_model, system_prompt, conversation)

        except Exception as e:
            print(red(f"\n  ✗ Local model error: {e}"))
            continue

        # ── Display response ──────────────────────────────────────────────
        print_response(reply)
        if SPEAK_REPLIES:
            speak_text(reply)

        # ── Apply any file edits ──────────────────────────────────────────
        edits = parse_edits(reply)
        if edits:
            apply_edits(edits, root)
            # Reload file context after edits
            files = discover_files(root)
            system_prompt = build_system_prompt(root, files)

        # ── Add reply to conversation history ─────────────────────────────
        conversation.append({"role": "assistant", "content": reply})

        # ── Trim history to avoid token overflow (keep last 20 turns) ─────
        if len(conversation) > 40:
            conversation = conversation[-40:]


if __name__ == "__main__":
    main()

