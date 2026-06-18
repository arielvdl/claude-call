"""Sons curtos de feedback (macOS afplay, não-bloqueante).

- heard: captei sua fala, vou executar
- on:    mic ATIVADO (voltou a ouvir)
- off:   mic MUTADO
Desliga com CALL_SOUNDS=0.
"""
import os
import subprocess
import sys
from pathlib import Path

_DIR = Path("/System/Library/Sounds")
_MAP = {"heard": "Pop.aiff", "wake": "Ping.aiff", "on": "Hero.aiff", "off": "Bottle.aiff", "done": "Tink.aiff"}
_ON = os.getenv("CALL_SOUNDS", "1") not in ("0", "false", "no")


def play(kind: str):
    if not _ON or sys.platform != "darwin":
        return
    f = _DIR / _MAP.get(kind, "Pop.aiff")
    try:
        subprocess.Popen(["afplay", str(f)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:  # noqa: BLE001
        pass
