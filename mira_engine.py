"""
Fast rule-based consultation engine for MIRA.
Multi-step dialog with quick-reply suggestions for the interactive VA.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

REQUIRED_COLUMNS = [
    "condition",
    "symptoms",
    "duration",
    "severity",
    "risk_factors",
    "suggested_action",
]

_DURATION_RANGE = re.compile(r"(\d+)\s*-\s*(\d+)", re.I)
_DURATION_SINGLE = re.compile(r"(\d+)")
_CHRONIC = frozenset({"chronic", "lifetime", "months-years", "weeks-months", "months"})
_ACUTE = frozenset({"acute", "hours-days", "hours", "6-48 hours"})

GREETING_WORDS = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "yes",
        "yeah",
        "yep",
        "ok",
        "okay",
        "ready",
        "start",
        "sure",
        "go",
        "begin",
    }
)
SKIP_WORDS = frozenset({"no", "none", "n/a", "skip", "just this", "only this", "that's all", "thats all"})
SEVERE_CONDITIONS = frozenset(
    {
        "heart attack",
        "stroke",
        "pneumonia",
        "appendicitis",
        "asthma attack",
    }
)

COMMON_SYMPTOM_CHIPS = [
    "Fever",
    "Cough",
    "Headache",
    "Sore throat",
    "Chest pain",
    "Nausea",
    "Fatigue",
    "Shortness of breath",
]
DURATION_CHIPS = ["1-2 days", "3-5 days", "About a week", "2+ weeks", "Over a month"]
TREND_CHIPS = ["Getting worse", "About the same", "Improving"]
SEVERITY_CHIPS = ["Mild", "Moderate", "Severe"]
IMPACT_CHIPS = ["Yes, a lot", "Somewhat", "Not really"]
RISK_CHIPS = ["None of these", "Skip"]
YES_NO_CHIPS = ["Yes", "No", "Not sure"]


def _default_csv_path() -> Path:
    here = Path(__file__).resolve().parent
    for name in ("med.csv", "eval_texts.csv"):
        p = here / name
        if p.exists():
            return p
    return here / "med.csv"


def _tokenize_symptoms(text: str) -> list[str]:
    text = str(text).lower()
    parts = re.split(r"[,;/|]+", text)
    tokens: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        tokens.append(part)
        for word in part.split():
            w = word.strip()
            if len(w) > 2:
                tokens.append(w)
    return tokens


def _parse_duration_bounds(raw: str) -> tuple[Optional[int], Optional[int], str]:
    s = str(raw).strip().lower()
    if not s or s in ("varies", "unknown"):
        return None, None, "varies"
    if s in _CHRONIC:
        return None, None, "chronic"
    if s in _ACUTE or "hour" in s:
        return 0, 3, "acute"
    if "week" in s:
        parts = re.findall(r"\d+", s)
        if len(parts) >= 2:
            return int(parts[0]) * 7, int(parts[1]) * 7, "range"
        if len(parts) == 1:
            w = int(parts[0])
            return w * 7, w * 7 + 6, "range"
        if s in ("weeks", "week"):
            return 7, 56, "range"
    m = _DURATION_RANGE.search(s)
    if m:
        return int(m.group(1)), int(m.group(2)), "range"
    m = _DURATION_SINGLE.search(s)
    if m:
        n = int(m.group(1))
        return n, n, "exact"
    return None, None, "other"


def duration_matches(user_days: int, raw: str) -> bool:
    low, high, category = _parse_duration_bounds(raw)
    if category == "varies":
        return True
    if category == "chronic":
        return user_days >= 30
    if category == "acute":
        return user_days <= 3
    if low is not None and high is not None:
        return low <= user_days <= high
    return False


def parse_duration_input(text: str) -> Optional[int]:
    t = text.strip().lower()
    if "month" in t or "over a month" in t:
        return 45
    if "2+" in t and "week" in t:
        return 14
    if "week" in t:
        m = _DURATION_SINGLE.search(t)
        if m:
            return int(m.group(1)) * 7
        return 7
    m = _DURATION_SINGLE.search(t)
    if m:
        n = int(m.group(1))
        if "1-2" in t:
            return 2
        if "3-5" in t:
            return 4
        return n
    return None


@dataclass
class ConsultationResult:
    message: str
    messages: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    input_hint: str = ""
    done: bool = False
    condition: Optional[str] = None
    suggested_action: Optional[str] = None
    matches: int = 0
    emergency: bool = False

    def all_messages(self) -> list[str]:
        if self.messages:
            return self.messages
        return [self.message] if self.message else []


@dataclass
class ConsultationSession:
    engine: "MiraEngine"
    candidate_ids: list[int] = field(default_factory=list)
    step: str = "welcome"
    last_symptom: str = ""
    user_days: Optional[int] = None
    trend: str = ""
    severity: str = ""
    impact: str = ""

    def reset(self) -> None:
        self.candidate_ids = []
        self.step = "welcome"
        self.last_symptom = ""
        self.user_days = None
        self.trend = ""
        self.severity = ""
        self.impact = ""

    @property
    def candidates(self) -> pd.DataFrame:
        if not self.candidate_ids:
            return self.engine.df.iloc[0:0]
        return self.engine.df.loc[self.candidate_ids]


class MiraEngine:
    def __init__(self, csv_path: Optional[str | Path] = None):
        path = Path(csv_path) if csv_path else _default_csv_path()
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        self.csv_path = path
        self.df = self._load(path)
        self._symptom_index: dict[str, set[int]] = {}
        self._build_index()

    @staticmethod
    def _load(path: Path) -> pd.DataFrame:
        for sep in ("\t", ",", "|"):
            try:
                df = pd.read_csv(path, sep=sep, dtype=str)
                if df.shape[1] >= len(REQUIRED_COLUMNS):
                    break
            except Exception:
                continue
        else:
            df = pd.read_csv(path, dtype=str)

        df.columns = df.columns.str.strip().str.lower()
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns {missing}; found {list(df.columns)}")

        df = df[REQUIRED_COLUMNS].fillna("")
        for col in REQUIRED_COLUMNS:
            df[col] = df[col].astype(str).str.strip()
        df["symptoms_lower"] = df["symptoms"].str.lower()
        df["severity_lower"] = df["severity"].str.lower()
        df["risk_lower"] = df["risk_factors"].str.lower()
        return df.reset_index(drop=True)

    def _build_index(self) -> None:
        index: dict[str, set[int]] = {}
        for i, row in self.df.iterrows():
            for token in _tokenize_symptoms(row["symptoms"]):
                index.setdefault(token, set()).add(i)
        self._symptom_index = index

    def match_symptom(self, symptom: str) -> list[int]:
        q = symptom.strip().lower()
        if not q:
            return []

        hits: set[int] = set()
        for token, ids in self._symptom_index.items():
            if q in token or token in q:
                hits |= ids

        if hits:
            return sorted(hits)

        mask = self.df["symptoms_lower"].str.contains(
            re.escape(q), na=False, regex=True
        )
        return self.df.index[mask].tolist()

    def new_session(self) -> ConsultationSession:
        return ConsultationSession(engine=self)

    def get_welcome(self) -> ConsultationResult:
        return ConsultationResult(
            message="",
            messages=[
                "Hello! I'm Dr. MIRA, your virtual health guide.",
                "I'll ask a few short questions to understand how you're feeling and suggest sensible next steps.",
                "This is not emergency care — if you're in danger, call your local emergency number now.",
                "When you're ready, tell me your main symptom or tap one below.",
            ],
            suggestions=COMMON_SYMPTOM_CHIPS + ["I'm ready"],
            input_hint="e.g. fever, cough, chest pain…",
        )

    def process(self, session: ConsultationSession, user_input: str) -> ConsultationResult:
        text = user_input.strip()
        if not text:
            return ConsultationResult(
                "I'm here when you're ready — type a reply or tap a suggestion below.",
                suggestions=self._chips_for_step(session.step),
            )

        low = text.lower()
        if low in {"exit", "quit", "bye", "stop"}:
            session.reset()
            return ConsultationResult(
                "Take care of yourself. If symptoms worsen, seek professional care. Goodbye!",
                done=True,
            )

        if session.step == "welcome":
            return self._step_welcome(session, text)
        if session.step == "symptom":
            return self._step_symptom(session, text)
        if session.step == "symptom_extra":
            return self._step_symptom_extra(session, text)
        if session.step == "duration":
            return self._step_duration(session, text)
        if session.step == "trend":
            return self._step_trend(session, text)
        if session.step == "severity":
            return self._step_severity(session, text)
        if session.step == "impact":
            return self._step_impact(session, text)
        if session.step == "risk":
            return self._step_risk(session, text)
        if session.step == "aftercare":
            return self._step_aftercare(session, text)

        session.reset()
        return self.get_welcome()

    def _chips_for_step(self, step: str) -> list[str]:
        return {
            "welcome": COMMON_SYMPTOM_CHIPS,
            "symptom": COMMON_SYMPTOM_CHIPS,
            "symptom_extra": ["No other symptoms"] + COMMON_SYMPTOM_CHIPS[:4],
            "duration": DURATION_CHIPS,
            "trend": TREND_CHIPS,
            "severity": SEVERITY_CHIPS,
            "impact": IMPACT_CHIPS,
            "risk": RISK_CHIPS,
            "aftercare": ["Start a new consultation", "Thank you, goodbye"],
        }.get(step, [])

    def _preview_conditions(self, ids: list[int], limit: int = 3) -> str:
        if not ids:
            return ""
        names = self.df.loc[ids[:limit], "condition"].tolist()
        if len(ids) > limit:
            return ", ".join(names) + f", and {len(ids) - limit} more"
        return ", ".join(names)

    def _step_welcome(self, session: ConsultationSession, text: str) -> ConsultationResult:
        low = text.lower()
        if low in GREETING_WORDS or low == "i'm ready":
            session.step = "symptom"
            return ConsultationResult(
                message="",
                messages=[
                    "Great — let's take this one step at a time.",
                    "What is the main symptom bothering you most right now?",
                ],
                suggestions=COMMON_SYMPTOM_CHIPS,
                input_hint="Describe your main symptom…",
            )
        session.step = "symptom"
        return self._step_symptom(session, text)

    def _step_symptom(self, session: ConsultationSession, text: str) -> ConsultationResult:
        ids = self.match_symptom(text)
        if not ids:
            return ConsultationResult(
                message="",
                messages=[
                    f"I didn't find a clear match for '{text}' in my records.",
                    "Could you try a simpler word — like fever, cough, or headache?",
                ],
                suggestions=COMMON_SYMPTOM_CHIPS,
                input_hint="Try another symptom…",
            )

        session.candidate_ids = ids
        session.last_symptom = text
        session.step = "symptom_extra"
        preview = self._preview_conditions(ids)
        n = len(ids)

        return ConsultationResult(
            message="",
            messages=[
                f"Thank you for sharing. I hear you're dealing with {text}.",
                f"I see {n} possible condition{'s' if n != 1 else ''} in my guide"
                + (f", including {preview}." if preview else "."),
                "Do you have any other symptoms right now, or is that the main one?",
            ],
            suggestions=["No other symptoms"] + COMMON_SYMPTOM_CHIPS[:5],
            input_hint="Another symptom, or say 'no'…",
        )

    def _step_symptom_extra(self, session: ConsultationSession, text: str) -> ConsultationResult:
        low = text.lower()
        if low in SKIP_WORDS or "no other" in low:
            session.step = "duration"
            return ConsultationResult(
                message="",
                messages=[
                    "Understood — we'll focus on that.",
                    "Roughly how long have you had this symptom?",
                ],
                suggestions=DURATION_CHIPS,
                input_hint="Number of days or pick a range…",
            )

        extra_ids = self.match_symptom(text)
        if extra_ids:
            combined = sorted(set(session.candidate_ids) & set(extra_ids))
            if combined:
                session.candidate_ids = combined
            session.last_symptom = f"{session.last_symptom}, {text}"

        session.step = "duration"
        n = len(session.candidate_ids)
        return ConsultationResult(
            message="",
            messages=[
                f"Got it — I've noted {text} as well.",
                f"That narrows us to {n} possible condition{'s' if n != 1 else ''}.",
                "How many days have you had these symptoms overall?",
            ],
            suggestions=DURATION_CHIPS,
            input_hint="e.g. 3 days, about a week…",
        )

    def _step_duration(self, session: ConsultationSession, text: str) -> ConsultationResult:
        days = parse_duration_input(text)
        if days is None:
            return ConsultationResult(
                message="",
                messages=[
                    "I need a rough timeline to help narrow things down.",
                    "About how many days has this been going on?",
                ],
                suggestions=DURATION_CHIPS,
                input_hint="e.g. 5 or 'about a week'",
            )

        session.user_days = days
        subset = session.candidates
        mask = subset["duration"].apply(lambda d: duration_matches(days, d))
        ids = subset.index[mask].tolist()

        if not ids:
            session.step = "symptom"
            session.candidate_ids = []
            return ConsultationResult(
                message="",
                messages=[
                    "That timeline doesn't match what I have on file for those symptoms.",
                    "If you're worried, please speak with a clinician.",
                    "Want to try describing the symptom again?",
                ],
                suggestions=COMMON_SYMPTOM_CHIPS,
                input_hint="Start again with your symptom…",
            )

        session.candidate_ids = ids
        session.step = "trend"
        return ConsultationResult(
            message="",
            messages=[
                f"Thanks — about {days} day{'s' if days != 1 else ''} helps a lot.",
                "Since it started, would you say things are getting worse, about the same, or improving?",
            ],
            suggestions=TREND_CHIPS,
            input_hint="How symptoms are changing…",
        )

    def _step_trend(self, session: ConsultationSession, text: str) -> ConsultationResult:
        low = text.lower()
        if "worse" in low:
            session.trend = "worse"
            ack = "I'm sorry it's getting worse — we'll factor that in."
        elif "improv" in low or "better" in low:
            session.trend = "improving"
            ack = "Good to hear there's some improvement."
        else:
            session.trend = "same"
            ack = "Stable symptoms are useful to know."

        session.step = "severity"
        return ConsultationResult(
            message="",
            messages=[
                ack,
                "How would you describe the intensity right now — mild, moderate, or severe?",
            ],
            suggestions=SEVERITY_CHIPS,
            input_hint="Mild, moderate, or severe…",
        )

    def _step_severity(self, session: ConsultationSession, text: str) -> ConsultationResult:
        low = text.lower()
        sev = None
        if "mild" in low:
            sev = "mild"
        elif "moderate" in low:
            sev = "moderate"
        elif "severe" in low:
            sev = "severe"

        if not sev:
            return ConsultationResult(
                message="",
                messages=["Please choose mild, moderate, or severe so I can guide you better."],
                suggestions=SEVERITY_CHIPS,
            )

        session.severity = sev
        subset = session.candidates
        mask = subset["severity_lower"].str.contains(sev, na=False, regex=False)
        ids = subset.index[mask].tolist()

        if not ids:
            session.step = "trend"
            return ConsultationResult(
                message="",
                messages=[
                    "That severity doesn't match the patterns I have for these symptoms.",
                    "Try mild, moderate, or severe again.",
                ],
                suggestions=SEVERITY_CHIPS,
            )

        session.candidate_ids = ids
        session.step = "impact"

        if sev == "severe":
            names = subset.loc[ids, "condition"].str.lower().tolist()
            if any(any(sc in n for sc in SEVERE_CONDITIONS) for n in names):
                return ConsultationResult(
                    message="",
                    messages=[
                        "Because you've marked symptoms as severe, I want to check in carefully.",
                        "Is this affecting your ability to breathe, speak, or stay awake normally?",
                    ],
                    suggestions=YES_NO_CHIPS,
                    input_hint="Yes or no…",
                    emergency=True,
                )

        return ConsultationResult(
            message="",
            messages=[
                f"Noted — {sev} intensity.",
                "Is this affecting your sleep, work, or daily activities in a major way?",
            ],
            suggestions=IMPACT_CHIPS,
            input_hint="How much it affects daily life…",
        )

    def _step_impact(self, session: ConsultationSession, text: str) -> ConsultationResult:
        low = text.lower()
        if "yes" in low or "lot" in low:
            session.impact = "high"
        elif "somewhat" in low or "some" in low:
            session.impact = "medium"
        else:
            session.impact = "low"

        session.step = "risk"
        subset = session.candidates
        risks = list(subset["risk_factors"].dropna().unique())[:5]
        risk_list = "\n".join(f"• {r}" for r in risks) if risks else "• (none listed)"

        return ConsultationResult(
            message="",
            messages=[
                "Almost done — a few lifestyle factors can help narrow guidance.",
                "Do any of these risk factors apply to you?",
                risk_list,
                "Reply with one that fits, or say 'none'.",
            ],
            suggestions=[*risks[:4], "None of these"] if risks else ["None"],
            input_hint="Risk factor or 'none'…",
        )

    def _step_risk(self, session: ConsultationSession, text: str) -> ConsultationResult:
        return self._finalize(session, text)

    def _finalize(self, session: ConsultationSession, risk_text: str) -> ConsultationResult:
        subset = session.candidates
        low = risk_text.lower()
        if low not in SKIP_WORDS and "none of" not in low:
            mask = subset["risk_lower"].str.contains(low, na=False, regex=False)
            filtered = subset.index[mask].tolist()
            if filtered:
                subset = subset.loc[filtered]

        if subset.empty:
            session.reset()
            return ConsultationResult(
                message="",
                messages=[
                    "I couldn't find a strong match with those details.",
                    "Please consider speaking with a healthcare professional for a proper assessment.",
                ],
                suggestions=["Start a new consultation"],
                done=True,
            )

        row = subset.iloc[0]
        condition = row["condition"]
        action = row["suggested_action"]
        others = len(subset) - 1

        trend_note = ""
        if session.trend == "worse":
            trend_note = "Since symptoms are worsening, don't wait too long to get checked if they continue."
        elif session.trend == "improving":
            trend_note = "Even if you're improving, keep monitoring and rest as needed."

        messages = [
            f"Based on what you've told me, **{condition}** is the closest match in my guide.",
            f"**Suggested next steps:** {action}",
        ]
        if others > 0:
            messages.append(
                f"I also had {others} other possible match{'es' if others != 1 else ''} — "
                "a clinician can confirm with an exam."
            )
        if trend_note:
            messages.append(trend_note)
        messages.append(
            "Remember: I'm an AI guide, not a doctor. Would you like to start a new consultation or end here?"
        )

        session.step = "aftercare"
        emergency = condition.lower() in SEVERE_CONDITIONS or session.severity == "severe"

        return ConsultationResult(
            message=messages[0],
            messages=messages,
            suggestions=["Start a new consultation", "Thank you, goodbye"],
            done=False,
            condition=condition,
            suggested_action=action,
            matches=len(subset),
            emergency=emergency,
        )

    def _step_aftercare(self, session: ConsultationSession, text: str) -> ConsultationResult:
        low = text.lower()
        if "new" in low or "start" in low or "again" in low:
            session.reset()
            return self.get_welcome()
        session.reset()
        return ConsultationResult(
            "I'm glad I could help. Take care, and seek urgent care if anything feels emergency-level.",
            done=True,
        )


class SpeechOutput:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._engine = None
        self._lock = threading.Lock()
        if enabled:
            try:
                import pyttsx3

                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", 185)
            except Exception:
                self._engine = None

    def say(self, text: str, *, print_text: bool = True) -> None:
        if print_text:
            print(f"\nMIRA: {text}\n")
        if not self.enabled or not self._engine:
            return

        def _run() -> None:
            with self._lock:
                self._engine.say(text)
                self._engine.runAndWait()

        threading.Thread(target=_run, daemon=True).start()


def run_text_consultation(
    engine: Optional[MiraEngine] = None,
    *,
    use_voice: bool = False,
    on_reply: Optional[Callable[[ConsultationResult], None]] = None,
) -> None:
    eng = engine or MiraEngine()
    session = eng.new_session()
    tts = SpeechOutput(enabled=use_voice)

    welcome = eng.get_welcome()
    for msg in welcome.all_messages():
        if on_reply:
            on_reply(welcome)
        else:
            tts.say(msg)

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        result = eng.process(session, user)
        for msg in result.all_messages():
            if on_reply:
                on_reply(result)
            else:
                tts.say(msg)
        if result.done:
            break


def run_voice_consultation(engine: Optional[MiraEngine] = None) -> None:
    import speech_recognition as sr

    eng = engine or MiraEngine()
    session = eng.new_session()
    tts = SpeechOutput(enabled=True)
    recognizer = sr.Recognizer()

    try:
        names = sr.Microphone.list_microphone_names()
    except Exception:
        names = []
    if not names:
        raise RuntimeError("No microphone found. Use run_text_consultation() instead.")

    mic = sr.Microphone()

    def listen() -> str:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            print("Listening…")
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=12)
        try:
            return recognizer.recognize_google(audio).strip()
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            raise RuntimeError(f"Speech recognition unavailable: {e}") from e

    for msg in eng.get_welcome().all_messages():
        tts.say(msg)

    while True:
        try:
            user = listen()
        except Exception as e:
            tts.say(str(e))
            continue

        if not user:
            tts.say("I didn't catch that. Please repeat.")
            continue

        print(f"You: {user}")
        result = eng.process(session, user)
        for msg in result.all_messages():
            tts.say(msg)
        if result.done:
            break
