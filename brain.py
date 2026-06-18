"""ClaudeBrain — o cerebro da call é o PROPRIO Claude Code, como daemon persistente.

Em vez de spawnar um `claude -p` novo a cada fala (cold, recarrega o contexto toda vez),
sobe UM processo `claude` em modo stream-json que fica vivo a call inteira:

  - o mic "conecta" nele escrevendo cada fala no stdin (mensagens JSON)
  - ele streama a resposta no stdout (falamos frase a frase)
  - contexto + cache ficam QUENTES entre os turnos (rapido depois do 1o)
  - com --resume <sua sessao>, o daemon É a sua conversa do Claude Code

Honestidade: nem um daemon "pensa" entre as falas — cada fala dispara uma inferencia
(o modelo roda nos servidores da Anthropic). O daemon mantem a sessao quente e viva;
nao e uma consciencia continua. Mas e um processo unico que o mic alimenta.
"""
import asyncio
import json
import math
import os
import re
import signal
import time
import uuid
from collections import deque

from loguru import logger

from pipecat.frames.frames import Frame, TranscriptionFrame, TTSSpeakFrame, UserStartedSpeakingFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

_SENT_BOUNDARY = re.compile(r"[.!?…](?=\s)|\n")


def _norm_words(s: str) -> list[str]:
    return [w for w in re.split(r"\W+", (s or "").lower()) if w]

# Palavra curta = só INTERROMPE a fala (nao vira prompt). Depois você continua falando.
_INTERRUPT_RE = re.compile(
    r"^\s*(ei+|ai+|opa|para|pára|parou|pera+|peraí|espera|calma|chega|stop|wait|hey)[\s!.,…]*$",
    re.IGNORECASE,
)

# MUTAÇÃO real (vai MEXER no código/sistema) -> só ISSO paga o modo code (Opus xHigh +
# respawn com --resume). Tirei 'debug/bug/fix/erro/analisa' daqui: são ambíguos e quase
# sempre o pedido é só ENTENDER. Trocar pra code nesses casos = resume pesado = trava.
_MUTATE_RE = re.compile(
    r"\b(implementa|refator|conserta|corrig|edita|altera(r)?|substitu|cria(r)?\s+"
    r"(o\s+|um\s+|uma\s+)?(arquivo|fun[çc]|componente|script|classe|m[ée]todo|teste)|"
    r"escrev[ae]\s+o|deploy|comita|commit|builda|instala|apaga|remov|delet|renomeia|"
    r"roda\s+o|executa\s+o)\b",
    re.IGNORECASE,
)

# "só pra entender, NÃO executa nada" -> NUNCA entra em code (read-only puro).
_NO_EXEC_RE = re.compile(
    r"\b(s[óo]\s+(pra|para)\s+(ver|entender|analis|olhar)|s[óo]\s+analis|apenas\s+analis|"
    r"n[ãa]o\s+(executa|mexe|mex|altera|muda|edita|faz)|sem\s+(executar|mexer|alterar|"
    r"mudar|editar|mexe))\b",
    re.IGNORECASE,
)

# Investigar/entender (read-only) -> fica no modo ATUAL (rápido), não respawna pro Opus.
_ANALYZE_RE = re.compile(
    r"\b(analis|entend|explica|explique|diagnostic|investig|revis|avalia|"
    r"por\s*qu[eê]|o\s+que\s+|d[áa]\s+uma\s+olhada|olha\s+(o|no|a)|v[êe]\s+(o|se|por))\b",
    re.IGNORECASE,
)

# Comando de voz pra abrir a janela "Claude Code ao vivo".
_SHOW_RE = re.compile(
    r"\b(mostra(r)?|abre|abrir|abra)\b.*\b(tela|terminal|janela|claude|fazendo|acontec|trabalh)",
    re.IGNORECASE,
)

# Campos de input de tool que viram "alvo" legivel no painel.
_TOOL_TARGET_FIELDS = ("file_path", "path", "notebook_path", "command",
                       "pattern", "query", "url", "prompt")


def _fuzzy_wake(word: str, wake: str) -> bool:
    """True se 'word' é provável mishear da wake word (sem deps externas).

    Cobre:
      - substring direto:  "audinha"  ⊆ "claudinha"  ✓
      - sufixo fonético:   "odinha"   → ends with "dinha" = wake[-5:]  ✓
    Rejeita palavras comuns: "galinha", "vizinha", "rainha" → False
    """
    w = word.strip(",.!?:;-—").lower()
    if not w or len(w) < 4:
        return False
    if w in wake and len(w) >= len(wake) * 0.60:
        return True
    min_suf = max(4, math.ceil(len(w) * 0.75))
    return len(wake) >= min_suf and w.endswith(wake[-min_suf:])


def _tool_target(inp: dict) -> str | None:
    if not isinstance(inp, dict):
        return None
    for k in _TOOL_TARGET_FIELDS:
        v = inp.get(k)
        if isinstance(v, str) and v.strip():
            v = v.strip().replace("\n", " ")
            return v if len(v) <= 48 else v[:45] + "…"
    return None


def _clean_for_speech(s: str) -> str:
    """Tira markdown antes de falar — TTS nao deve ler *, #, `, bullets, links."""
    s = s.replace("```", " ")
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    s = re.sub(r"[*_~#>|]+", "", s)
    s = re.sub(r"^\s*[-•·]\s*", "", s)
    return re.sub(r"\s+", " ", s).strip()


class ClaudeBrain(FrameProcessor):
    def __init__(
        self,
        *,
        voice_rules: str,
        model: str | None = None,
        effort: str | None = None,
        code_model: str | None = None,
        code_effort: str | None = None,
        permission_flag: str = "--dangerously-skip-permissions",
        wake_words: list[str] | None = None,
        active_window_secs: float = 25.0,
        fillers: list[str] | None = None,
        session_id: str | None = None,
        cwd: str | None = None,
        first_resp_timeout: float = 45.0,
        stall_timeout: float = 120.0,
        recover_phrase: str = "Sorry, I got stuck — please say that again.",
        ui=None,
    ):
        super().__init__()
        self._ui = ui
        self._voice_rules = voice_rules
        self._cwd = cwd or os.getcwd()
        # Dois modos: chat (voz, rapido) e code (programar, forte). Troca on-the-fly.
        self._chat_model, self._chat_effort = model, effort
        self._code_model, self._code_effort = code_model, code_effort
        self._model, self._effort = model, effort
        self._mode = "chat"
        self._permission_flag = permission_flag
        self._wake = [w.lower() for w in (wake_words or []) if w.strip()]
        self._active_window = active_window_secs
        self._fillers = fillers or ["One sec.", "Hold on.", "Let me check.", "Give me a moment."]
        self._filler_i = 0

        self._resume = session_id          # se setado, RETOMA essa conversa
        self._session_id = session_id or str(uuid.uuid4())

        self._proc: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task | None = None
        self._buf = ""
        self._discard = False
        self._narrated_tool = False
        self._spoke = False
        self._active_until = 0.0
        self._awaiting = False   # esperando 1a resposta do turno (pro watchdog)
        # Watchdog do turno: recupera sozinho se o cérebro travar (ex.: --resume de sessão
        # pesada = prefill gigante que nunca volta). Mata + respawna FRESH + avisa.
        self._first_resp_timeout = first_resp_timeout  # s sem NENHUMA resposta -> recomeça
        self._stall_timeout = stall_timeout            # s travado no meio (já respondeu) -> recomeça
        self._recover_phrase = recover_phrase
        self._turn_seq = 0           # id do turno atual (invalida watchdog de turno antigo)
        self._turn_active = False    # há turno em andamento (entre _send e o 'result')
        self._last_activity = 0.0    # monotonic do último evento vindo do claude
        self._ignore_utterance = False   # mute decidido no INÍCIO da fala (deixa a atual terminar)
        self._recent_speech: deque = deque(maxlen=30)  # (t, set(palavras)) — anti-eco
        self._last_text = ""             # último pedido do usuário (pra refazer sozinho no recover)
        self._retried = False            # já refez este turno? (evita loop recover->refaz->recover)

    def _should_answer(self, text: str) -> bool:
        if not self._wake:
            return True
        if time.monotonic() < self._active_until:
            return True
        low = text.lower()
        words = re.split(r"\W+", low)
        return any(
            (w in low) or any(_fuzzy_wake(word, w) for word in words)
            for w in self._wake
        )

    def _has_wake(self, text: str) -> bool:
        low = text.lower()
        words = re.split(r"\W+", low)
        return any(
            (w in low) or any(_fuzzy_wake(word, w) for word in words)
            for w in self._wake
        )

    def _strip_wake(self, text: str) -> str:
        """Tira wake word e misheards ('audinha', 'odinha'…) do texto — são só gatilho."""
        if not self._wake:
            return text
        out = text
        for wake in self._wake:
            # exact substring removal
            low = out.lower()
            idx = low.find(wake)
            while idx != -1:
                out = out[:idx] + " " + out[idx + len(wake):]
                low = out.lower()
                idx = low.find(wake)
            # fuzzy word removal (misheards)
            parts = re.split(r"(\W+)", out)
            out = "".join(
                " " if (i % 2 == 0 and _fuzzy_wake(parts[i], wake)) else parts[i]
                for i, _ in enumerate(parts)
            )
        return re.sub(r"\s+", " ", out).strip(" ,.!?:;-—").strip()

    def _build_cmd(self) -> list[str]:
        cmd = [
            "claude", "-p",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--include-partial-messages", "--verbose",
        ]
        cmd += ["--resume", self._session_id] if self._resume else ["--session-id", self._session_id]
        if self._permission_flag:
            cmd += [self._permission_flag]
        if self._model:
            cmd += ["--model", self._model]
        if self._effort:
            cmd += ["--effort", self._effort]
        cmd += ["--append-system-prompt", self._voice_rules]
        return cmd

    async def _ensure_proc(self):
        if self._proc and self._proc.returncode is None:
            return
        cmd = self._build_cmd()
        logger.info(f"[brain] up ({'resume '+self._session_id if self._resume else 'new session'})")
        self._proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=self._cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True, limit=2**20, env={**os.environ},
        )
        self._reader = self.create_task(self._read_loop())
        self.create_task(self._drain_stderr())

    async def _drain_stderr(self):
        try:
            async for raw in self._proc.stderr:
                line = raw.decode(errors="ignore").rstrip()
                if line:
                    logger.debug(f"[claude] {line}")
        except Exception:  # noqa: BLE001
            pass

    async def _send(self, text: str, *, is_retry: bool = False):
        if not is_retry:
            self._last_text = text   # guarda pro recover refazer; retry não sobrescreve nem rearma
            self._retried = False
        await self._ensure_proc()
        from sounds import play
        play("heard")           # bip: captei sua fala, vou executar
        if self._ui:
            self._ui.heard(text)
        self._buf = ""
        self._discard = False
        self._narrated_tool = False
        self._spoke = False
        self._awaiting = True
        self._turn_active = True
        self._turn_seq += 1
        self._last_activity = time.monotonic()
        self.create_task(self._slow_watch(self._turn_seq))
        msg = {"type": "user", "message": {"role": "user", "content": text}}
        payload = (json.dumps(msg) + "\n").encode()
        try:
            self._proc.stdin.write(payload)
            await self._proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            logger.error("[brain] daemon dropped; restarting")
            self._proc = None
            await self._ensure_proc()
            self._proc.stdin.write(payload)
            await self._proc.stdin.drain()

    async def _iter_lines(self):
        """Le o stdout em chunks e quebra por '\\n' na mao. O `async for ... in stdout`
        usa readline (limite de 1MB/linha) e estoura LimitOverrunError quando o claude
        emite uma linha JSON gigante (tool/conteudo grande). read() nao tem esse limite."""
        buf = b""
        while True:
            chunk = await self._proc.stdout.read(65536)
            if not chunk:
                if buf.strip():
                    yield buf.decode(errors="ignore").strip()
                break
            buf += chunk
            while True:
                nl = buf.find(b"\n")
                if nl < 0:
                    break
                raw, buf = buf[:nl], buf[nl + 1:]
                line = raw.decode(errors="ignore").strip()
                if line:
                    yield line

    async def _read_loop(self):
        try:
            async for line in self._iter_lines():
                self._last_activity = time.monotonic()   # sinal de vida p/ o watchdog
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = ev.get("type")

                if etype == "system" and ev.get("subtype") == "init":
                    sid = ev.get("session_id")
                    if sid:
                        self._session_id = sid
                        self._resume = sid
                        if self._ui:
                            self._ui.set_session(sid)
                    continue

                # deltas: latencia baixa pra VOZ (fala frase a frase) + filler ao usar tool
                if etype == "stream_event":
                    self._awaiting = False   # chegou resposta: cancela o watchdog
                    inner = ev.get("event", {})
                    itype = inner.get("type")
                    if itype == "content_block_start":
                        if inner.get("content_block", {}).get("type") == "tool_use" \
                           and not self._narrated_tool and not self._discard:
                            self._narrated_tool = True
                            await self._say(self._next_filler())
                    elif itype == "content_block_delta":
                        delta = inner.get("delta", {})
                        if delta.get("type") == "text_delta" and not self._discard:
                            self._buf += delta.get("text", "")
                            self._buf, sents = self._drain(self._buf)
                            for s in sents:
                                if await self._say(s):
                                    self._spoke = True
                                    if self._ui:
                                        self._ui.reply_append(s)

                # mensagem COMPLETA do assistente: tool_use (input cheio) + thinking + texto
                # -> alimenta o painel (compacto) e o work.feed (Claude Code programando)
                elif etype == "assistant":
                    for b in (ev.get("message", {}).get("content") or []):
                        bt = b.get("type")
                        if bt == "tool_use":
                            name, inp = b.get("name") or "tool", b.get("input") or {}
                            if self._ui:
                                self._ui.tool(name, _tool_target(inp))
                                self._ui.work_tool(name, inp)
                        elif bt == "thinking" and self._ui:
                            self._ui.work_think(b.get("thinking", ""))
                        elif bt == "text" and self._ui:
                            self._ui.work_text(b.get("text", ""))

                # resultado de tool (stdout/erro) -> work.feed
                elif etype == "user":
                    for b in (ev.get("message", {}).get("content") or []):
                        if b.get("type") == "tool_result" and self._ui:
                            self._ui.work_result(b.get("content"), b.get("is_error"))

                elif etype == "result":
                    self._turn_active = False     # turno fechou: desarma o watchdog
                    self._awaiting = False
                    tail = self._buf.strip()
                    if not self._discard and await self._say(self._buf):
                        self._spoke = True
                        if self._ui and tail:
                            self._ui.reply_append(tail)
                    self._buf = ""
                    self._active_until = time.monotonic() + self._active_window
                    if self._ui:
                        self._ui.done()
            # stdout fechou: daemon saiu. Se foi NO MEIO de um turno, não deixa travado —
            # zera o relógio do watchdog pra ele recuperar (respawn fresh) já no próximo tick.
            if self._turn_active:
                logger.warning("[brain] daemon saiu sem 'result' no meio do turno")
                self._last_activity = 0.0
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.exception("[brain] read loop died")
            self._last_activity = 0.0    # deixa o watchdog recuperar
            if self._ui:
                self._ui.error(f"cérebro caiu: {e}", e)

    def _next_filler(self) -> str:
        f = self._fillers[self._filler_i % len(self._fillers)]
        self._filler_i += 1
        return f

    def _remember_speech(self, text: str):
        """Guarda o que a assistente FALOU (pro filtro anti-eco)."""
        words = set(_norm_words(text))
        if words:
            self._recent_speech.append((time.monotonic(), words))

    def _is_echo(self, text: str) -> bool:
        """True se o 'ouvido' é a propria voz dela voltando pelo alto-falante.
        Compara com o que ela falou nos ultimos ~8s (sobreposicao de palavras)."""
        words = _norm_words(text)
        if len(words) < 2:
            return False
        now = time.monotonic()
        spoken = set()
        for t, ws in self._recent_speech:
            if now - t <= 8.0:
                spoken |= ws
        if not spoken:
            return False
        hit = sum(1 for w in set(words) if w in spoken)
        return hit / len(set(words)) >= 0.7

    async def _say(self, text: str) -> bool:
        clean = _clean_for_speech(text)
        if not clean:
            return False
        self._remember_speech(clean)
        await self.push_frame(TTSSpeakFrame(clean), FrameDirection.DOWNSTREAM)
        return True

    def _drain(self, buf: str):
        sents = []
        while True:
            m = _SENT_BOUNDARY.search(buf)
            if not m:
                break
            end = m.end()
            s = buf[:end].strip()
            buf = buf[end:]
            if s:
                sents.append(s)
        return buf, sents

    def _watch_decision(self, awaiting: bool, idle: float):
        """Puro/testável: dado (esperando 1a resposta?, segundos ocioso) decide o que o
        watchdog faz -> ('none'|'hint'|'recover', mensagem)."""
        if awaiting:                                   # nenhuma resposta ainda neste turno
            if idle >= self._first_resp_timeout:       # resume pesado / daemon morto -> recupera
                return ("recover", "sem resposta (resume pesado ou daemon morto)")
            if idle >= 12:
                return ("hint", "demorando… (recomeço sozinho se travar)")
            return ("none", "")
        # já respondeu: pode estar num tool longo. Só recupera em stall LONGO (acima do
        # timeout do próprio Bash do claude), pra não matar ferramenta legítima.
        if idle >= self._stall_timeout:
            return ("recover", "travou no meio do turno")
        return ("none", "")

    async def _slow_watch(self, seq: int):
        """Watchdog do turno: avisa se demora e RECUPERA (mata + respawn FRESH) se travar,
        pra nunca ficar preso em 'pensando' pra sempre."""
        hinted = False
        try:
            while True:
                await asyncio.sleep(3)
                if seq != self._turn_seq or not self._turn_active:
                    return                              # turno acabou ou foi substituído
                idle = time.monotonic() - self._last_activity
                action, msg = self._watch_decision(self._awaiting, idle)
                if action == "hint" and not hinted and self._ui:
                    self._ui.hint(msg)
                    hinted = True
                elif action == "recover":
                    await self._recover(msg, seq)
                    return
        except asyncio.CancelledError:
            pass

    async def _recover(self, reason: str, seq: int):
        """Sai do travamento: mata o daemon e recomeça FRESH (sem re-resume da sessão pesada,
        que é a causa do trava). Se o turno travado era read-only (modo chat), REFAZ sozinho
        o último pedido — sem fazer o usuário repetir. Em modo code pode ter mutação no meio,
        então não refaz cego: pede pra repetir."""
        if seq != self._turn_seq:
            return
        logger.warning(f"[brain] watchdog recuperando ({reason}) — respawn FRESH")
        # read-only e ainda não refez? -> dá pra refazer sozinho com segurança.
        safe_redo = self._mode == "chat" and bool(self._last_text) and not self._retried
        self._kill()
        self._proc = None
        self._resume = None                    # FRESH: NÃO re-resume a sessão pesada
        self._session_id = str(uuid.uuid4())
        self._turn_active = False
        self._awaiting = False
        self._discard = True                   # descarta qualquer cauda do turno morto
        self._buf = ""
        if self._ui:
            self._ui.set_session(self._session_id)
        if safe_redo:
            self._retried = True
            if self._ui:
                self._ui.status("refazendo…")
            await self._say("Perai, deixa eu refazer isso.")
            await self._send(self._last_text, is_retry=True)
            return
        if self._ui:
            self._ui.error("travei e recomecei do zero — pode repetir?")
            self._ui.status("ouvindo")
        await self._say(self._recover_phrase)  # feedback por voz (usuário pode não estar olhando)

    def _desired_mode(self, text: str) -> str:
        """Escolhe chat<->code MINIMIZANDO flips: cada troca MATA + faz --resume da sessão
        (caro numa call longa = causa raiz do 'travei e recomecei'). Regras:
          1. Mutação explícita ('conserta', 'edita', 'deploy'...) -> 'code'. Único gatilho
             que paga o respawn pro Opus xHigh.
          2. 'só analisa / não executa' ou pergunta de investigação ('analisa', 'por que',
             'o que') -> read-only: FICA no modo atual (sem flip, sem resume pesado). Antes
             'bug/debug/erro' jogava pra code e travava resumindo sessão pesada.
          3. Fala neutra ('ok', 'agora testa') -> stickiness: fica no modo atual na janela."""
        now = time.monotonic()
        sticky = now < self._active_until
        if _MUTATE_RE.search(text):
            return "code"
        if _NO_EXEC_RE.search(text) or _ANALYZE_RE.search(text):
            return self._mode if sticky else "chat"   # read-only: sem flip
        if self._mode == "code" and sticky:
            return "code"
        return "chat"

    async def _switch_mode(self, mode: str):
        """Troca chat<->code (modelo/effort). Se ja tem daemon, respawna resumindo a conversa."""
        if mode == self._mode:
            return
        self._mode = mode
        if mode == "code":
            self._model, self._effort = self._code_model, self._code_effort
        else:
            self._model, self._effort = self._chat_model, self._chat_effort
        if self._ui:
            self._ui.hint("modo código (Opus xHigh)…" if mode == "code" else "modo voz (Sonnet)")
        logger.info(f"[brain] modo={mode} model={self._model} effort={self._effort}")
        if self._proc is not None and self._proc.returncode is None:
            self._resume = self._session_id     # mantem a conversa no novo modelo
            self._kill()
            self._proc = None
            await self._ensure_proc()

    async def inject_text(self, text: str):
        """Injeta um turno de TEXTO (colado/digitado) como se você tivesse falado."""
        text = (text or "").strip()
        if not text:
            return
        await self._switch_mode(self._desired_mode(text))
        await self._send(text)

    async def _open_tab_bg(self):
        """Abre a aba do lado fora do event loop (osascript bloqueia ~ segundos)."""
        try:
            await asyncio.to_thread(self._ui.open_window)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[brain] open tab: {e}")
            if self._ui:
                self._ui.error(f"abrir aba: {e}", e)

    def _kill(self):
        if self._proc and self._proc.returncode is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # grava a fala que passa pelo pipeline (ex.: greeting), pro filtro anti-eco
        if isinstance(frame, TTSSpeakFrame) and (frame.text or "").strip():
            self._remember_speech(frame.text)

        if isinstance(frame, UserStartedSpeakingFrame):
            # decide AQUI (no início da fala) se ela vale. Mutar DEPOIS não descarta a fala
            # que já começou — ela termina de transcrever e executar.
            self._ignore_utterance = bool(self._ui and self._ui.muted)
            if not self._ignore_utterance:  # barge-in só quando não está mutado
                self._discard = True
                self._buf = ""
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TranscriptionFrame) and (frame.text or "").strip():
            text = frame.text.strip()
            if self._ignore_utterance or (self._ui and self._ui.muted):
                # fala que COMEÇOU mutada, ou mute apertado no meio da fala -> ignora
                # (o EchoGate já fecha o mic; isto pega o parcial que já estava no buffer).
                logger.debug(f"[brain] mutado, ignorando: {text!r}")
                return
            if _INTERRUPT_RE.match(text):  # "ei"/"para" = só interrompe a fala
                self._discard = True
                self._buf = ""
                if self._ui:
                    self._ui.status("ouvindo")
                logger.debug(f"[brain] interrompido por: {text!r}")
                return
            if self._is_echo(text):       # a propria voz dela voltando pelo alto-falante
                logger.debug(f"[brain] eco ignorado: {text!r}")
                return
            if not self._should_answer(text):
                logger.debug(f"[brain] gated (sem '{'/'.join(self._wake)}'): {text!r}")
                return
            # disse a wake word -> tira ela do pedido (é só gatilho). Se só disse "claudinha"
            # (sem comando), abre a janela ativa pro próximo turno não precisar repetir.
            if self._wake and self._has_wake(text):
                text = self._strip_wake(text)
                if self._ui:
                    self._ui.status("captei")   # feedback imediato: "ouvi você"
                from sounds import play as _play; _play("wake")
                if not text:
                    self._active_until = time.monotonic() + self._active_window
                    logger.debug("[brain] wake word só, abrindo janela ativa")
                    # volta pra "ouvindo" após 1.5s (só confirmou que ouviu)
                    async def _reset_captei():
                        await asyncio.sleep(1.5)
                        if self._ui and self._ui._status == "captei":
                            self._ui.status("ouvindo")
                    self.create_task(_reset_captei())
                    return
            # comando de voz: abrir a aba "Claude Code ao vivo" (sem travar a call)
            if self._ui and _SHOW_RE.search(text):
                self._ui.status("abrindo a aba…")
                self.create_task(self._open_tab_bg())
                await self._say("Abrindo a aba aqui do lado.")
                return
            # programar? troca pro modelo forte (Opus xHigh); senão fica no chat (Sonnet)
            await self._switch_mode(self._desired_mode(text))
            await self._send(text)
            return

        await self.push_frame(frame, direction)

    async def cleanup(self):
        await super().cleanup()
        if self._reader:
            self._reader.cancel()
        self._kill()
