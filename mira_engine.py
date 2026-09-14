"""
Fast rule-based consultation engine for MIRA.
Multi-step dialog with quick-reply suggestions for the interactive VA.
"""

from __future__ import annotations

import math
import re
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher, get_close_matches
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
OPTIONAL_COLUMNS = ["category", "source", "keywords"]

SYMPTOM_ALIASES: dict[str, str] = {
    "temperature": "fever",
    "temp": "fever",
    "high temp": "fever",
    "flu": "influenza",
    "the flu": "influenza",
    "runny nose": "congestion",
    "blocked nose": "nasal congestion",
    "stuffy nose": "nasal congestion",
    "stomach ache": "abdominal pain",
    "stomach pain": "abdominal pain",
    "tummy pain": "abdominal pain",
    "belly pain": "abdominal pain",
    "stomach bug": "gastroenteritis",
    "food bug": "food poisoning",
    "throwing up": "vomiting",
    "puking": "vomiting",
    "being sick": "vomiting",
    "sick": "nausea",
    "feel sick": "nausea",
    "breathless": "shortness of breath",
    "can't breathe": "difficulty breathing",
    "cant breathe": "difficulty breathing",
    "hard to breathe": "shortness of breath",
    "wheezing": "wheezing",
    "hives": "urticaria",
    "uti": "burning urination",
    "pee pain": "painful urination",
    "pain when peeing": "burning urination",
    "back ache": "back pain",
    "migraine headache": "migraine",
    "panic": "panic attack",
    "panic attack": "panic attack",
    "anxiety": "worry",
    "tired": "fatigue",
    "exhausted": "fatigue",
    "worn out": "fatigue",
    "cold": "runny nose",
    "common cold": "runny nose",
    "sore throat": "sore throat",
    "throat pain": "sore throat",
    "toothache": "tooth pain",
    "tooth pain": "toothache",
    "heartburn": "heartburn",
    "acid reflux": "heartburn",
    "can't sleep": "insomnia",
    "cant sleep": "insomnia",
    "trouble sleeping": "insomnia",
    "dizzy": "dizziness",
    "spinning": "vertigo",
    "room spinning": "vertigo",
    "hay fever": "seasonal allergies",
    "allergies": "seasonal allergies",
    "hangover": "hangover",
    "hungover": "hangover",
    "snoring": "sleep apnea",
    "zits": "acne",
    "pimples": "acne",
    "ringing ears": "tinnitus",
    "pink eye": "red eye",
    "blood sugar low": "hypoglycemia",
    "blood sugar high": "hyperglycemia",
    "not feeling well": "fatigue",
    "don't feel well": "fatigue",
    "dont feel well": "fatigue",
    "feel unwell": "fatigue",
    "feeling unwell": "fatigue",
    "under the weather": "fatigue",
    "my head hurts": "headache",
    "head hurts": "headache",
    "stomach hurts": "abdominal pain",
    "tummy hurts": "abdominal pain",
    "chest hurts": "chest pain",
    "body hurts": "body aches",
    "body ache": "body aches",
    "muscle ache": "body aches",
    "coronavirus": "covid",
    "corona": "covid",
    "covid": "covid",
    "covid 19": "covid",
    "loss of smell": "loss of taste or smell",
    "loss of taste": "loss of taste or smell",
    "cannot breathe": "difficulty breathing",
    "hard time breathing": "shortness of breath",
    "pee hurts": "burning urination",
    "burning when peeing": "burning urination",
    "itchy skin": "itchy rash",
    "skin problem": "rash",
    "throat hurts": "sore throat",
    "ear hurts": "ear pain",
    "eye hurts": "eye pain",
}

# Exact UI chip / topic labels → canonical symptom phrase
CHIP_SYMPTOM_MAP: dict[str, str] = {
    "fever": "fever",
    "cough": "cough",
    "headache": "headache",
    "sore throat": "sore throat",
    "chest pain": "chest pain",
    "nausea": "nausea",
    "fatigue": "fatigue",
    "shortness of breath": "shortness of breath",
    "rash": "rash",
    "joint pain": "joint pain",
    "dizziness": "dizziness",
    "diarrhea": "diarrhea",
    "breathing problem": "shortness of breath",
    "stomach issue": "abdominal pain",
    "skin rash": "rash",
    "urinary symptoms": "burning urination",
}

_SKIP_PHRASE_SPLIT = re.compile(r"[,;/|]+")
_WORD_RE = re.compile(r"[a-z0-9']{2,}")
_ALIAS_ORDER = sorted(SYMPTOM_ALIASES.items(), key=lambda item: -len(item[0]))

TOPIC_CHIPS = [
    "Breathing problem",
    "Stomach issue",
    "Headache",
    "Skin rash",
    "Joint pain",
    "Urinary symptoms",
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
QUERY_STOP_WORDS = frozenset(
    {
        "symptom",
        "symptoms",
        "problem",
        "issues",
        "issue",
        "feeling",
        "feel",
        "having",
        "getting",
        "something",
        "wrong",
        "help",
        "random",
        "really",
        "very",
        "just",
        "like",
        "been",
        "have",
        "has",
        "had",
    }
)
SEVERE_CONDITIONS = frozenset(
    {
        "heart attack",
        "stroke",
        "pneumonia",
        "appendicitis",
        "asthma attack",
        "meningitis",
        "pulmonary embolism",
        "deep vein thrombosis",
        "seizure",
        "kidney stones",
        "kidney infection",
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
    "Rash",
    "Joint pain",
    "Dizziness",
    "Diarrhea",
]
AGE_CHIPS = ["Under 18", "18 to 64", "65 or older", "Prefer not to say"]
FEVER_CHIPS = ["Low fever", "High fever (over 102F)", "No thermometer", "No fever now"]
MEDICATION_CHIPS = ["None", "Blood pressure medicine", "Blood thinners", "Diabetes medicine", "Other regular medicine"]
RED_FLAG_CHIPS = ["Yes — one of these", "No", "Not sure"]
DURATION_CHIPS = ["1-2 days", "3-5 days", "About a week", "2+ weeks", "Over a month"]
TREND_CHIPS = ["Getting worse", "About the same", "Improving"]
SEVERITY_CHIPS = ["Mild", "Moderate", "Severe"]
IMPACT_CHIPS = ["Yes, a lot", "Somewhat", "Not really"]
RISK_CHIPS = ["None of these", "Skip"]
YES_NO_CHIPS = ["Yes", "No", "Not sure"]

GENERIC_SYMPTOM_TOKENS = frozenset(
    {
        "fever",
        "pain",
        "ache",
        "fatigue",
        "tired",
        "cough",
        "headache",
        "rash",
        "nausea",
        "dizziness",
        "diarrhea",
    }
)

# Typical first-line matches for common presenting symptoms
PREFERRED_CONDITIONS: dict[str, list[str]] = {
    "fever": ["Influenza", "COVID-19", "Common Cold", "Strep Throat", "Mononucleosis"],
    "cough": ["Common Cold", "Bronchitis", "Influenza", "RSV Infection", "COVID-19"],
    "headache": ["Tension Headache", "Migraine", "Sinus Headache", "Cluster Headache"],
    "diarrhea": ["Gastroenteritis", "Norovirus", "Food Poisoning", "Rotavirus"],
    "nausea": ["Gastroenteritis", "Food Poisoning", "Norovirus", "Migraine"],
    "fatigue": ["Influenza", "Common Cold", "Anemia", "Post-Viral Fatigue"],
    "rash": ["Urticaria (Hives)", "Eczema", "Contact Dermatitis", "Chickenpox"],
    "chest pain": ["Angina", "GERD", "Anxiety Disorder", "Pneumonia"],
    "sore throat": ["Pharyngitis", "Strep Throat", "Common Cold", "Laryngitis"],
    "abdominal pain": ["Gastroenteritis", "Irritable Bowel Syndrome", "GERD", "Peptic Ulcer"],
}


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
    source: Optional[str] = None
    category: Optional[str] = None

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
    age_group: str = ""
    fever_level: str = ""
    medication: str = ""
    needs_red_flag: bool = False
    match_scores: dict[int, float] = field(default_factory=dict)
    symptom_queries: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.candidate_ids = []
        self.step = "welcome"
        self.last_symptom = ""
        self.user_days = None
        self.trend = ""
        self.severity = ""
        self.impact = ""
        self.age_group = ""
        self.fever_level = ""
        self.medication = ""
        self.needs_red_flag = False
        self.match_scores = {}
        self.symptom_queries = []

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
        self._duration_bounds: list[tuple[Optional[int], Optional[int], str]] = []
        self._condition_names_lower: list[str] = []
        self._conditions: list[str] = []
        self._search_blobs: list[str] = []
        self._symptom_phrases: list[tuple[str, ...]] = []
        self._keyword_phrases: list[tuple[str, ...]] = []
        self._condition_words: list[tuple[str, ...]] = []
        self._severity_lower_list: list[str] = []
        self._risk_lower_list: list[str] = []
        self._risk_factors_list: list[str] = []
        self._actions: list[str] = []
        self._sources: list[str] = []
        self._categories: list[str] = []
        self._name_to_idx: dict[str, int] = {}
        self._token_df: dict[str, int] = {}
        self._primary_symptom_token: list[str] = []
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

        for col in OPTIONAL_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        keep = [c for c in REQUIRED_COLUMNS + OPTIONAL_COLUMNS if c in df.columns]
        df = df[keep].fillna("")
        for col in keep:
            df[col] = df[col].astype(str).str.strip()
        df["symptoms_lower"] = df["symptoms"].str.lower()
        df["severity_lower"] = df["severity"].str.lower()
        df["risk_lower"] = df["risk_factors"].str.lower()
        df["condition_lower"] = df["condition"].str.lower()
        df["keywords_lower"] = df["keywords"].str.lower()
        df["search_blob"] = (
            df["condition_lower"]
            + " "
            + df["symptoms_lower"]
            + " "
            + df["keywords_lower"]
        )
        df = df[df["condition"].str.len() > 0].reset_index(drop=True)
        return df

    def _build_index(self) -> None:
        index: dict[str, set[int]] = {}
        df = self.df
        n = len(df)

        self._duration_bounds = [_parse_duration_bounds(d) for d in df["duration"]]
        self._condition_names_lower = df["condition_lower"].tolist()
        self._conditions = df["condition"].tolist()
        self._search_blobs = df["search_blob"].tolist()
        self._severity_lower_list = df["severity_lower"].tolist()
        self._risk_lower_list = df["risk_lower"].tolist()
        self._risk_factors_list = df["risk_factors"].tolist()
        self._actions = df["suggested_action"].tolist()
        self._sources = df["source"].tolist() if "source" in df.columns else [""] * n
        self._categories = df["category"].tolist() if "category" in df.columns else [""] * n
        self._name_to_idx = {name: i for i, name in enumerate(self._condition_names_lower)}

        symptom_phrases: list[tuple[str, ...]] = []
        keyword_phrases: list[tuple[str, ...]] = []
        condition_words: list[tuple[str, ...]] = []

        for i in range(n):
            row = df.iloc[i]
            sym_phrases = tuple(
                p.strip()
                for p in _SKIP_PHRASE_SPLIT.split(row["symptoms_lower"])
                if len(p.strip()) >= 4
            )
            kw_phrases = tuple(
                p.strip()
                for p in _SKIP_PHRASE_SPLIT.split(row["keywords_lower"])
                if p.strip()
            )
            symptom_phrases.append(sym_phrases)
            keyword_phrases.append(kw_phrases)
            condition_words.append(
                tuple(_WORD_RE.findall(row["condition_lower"]))
            )

            tokens = _tokenize_symptoms(row["symptoms"])
            tokens.extend(_tokenize_symptoms(row["keywords"]))
            tokens.extend(_tokenize_symptoms(row["condition"]))
            for part in re.split(r"[\s\-/()]+", row["condition_lower"]):
                part = part.strip()
                if len(part) > 2:
                    tokens.append(part)
            for token in tokens:
                if len(token) > 2:
                    index.setdefault(token, set()).add(i)

        self._symptom_phrases = symptom_phrases
        self._keyword_phrases = keyword_phrases
        self._condition_words = condition_words
        self._symptom_index = index
        self._token_df = {token: len(idset) for token, idset in index.items()}
        self._primary_symptom_token = []
        for i in range(n):
            first = df.iloc[i]["symptoms_lower"].split(",")[0].strip()
            tokens = _WORD_RE.findall(first)
            self._primary_symptom_token.append(tokens[0] if tokens else "")

    def _normalize_symptom_query(self, symptom: str) -> str:
        q = symptom.strip().lower()
        q = q.replace("-", " ").replace("’", "'")
        q = re.sub(r"\s+", " ", q).strip()

        if q in CHIP_SYMPTOM_MAP:
            return CHIP_SYMPTOM_MAP[q]

        for alias, canonical in _ALIAS_ORDER:
            if alias in q:
                q = q.replace(alias, canonical)
        q = re.sub(r"\s+", " ", q).strip()

        topic_map = {
            "breathing problem": "shortness of breath",
            "stomach issue": "abdominal pain",
            "headache": "headache",
            "skin rash": "rash",
            "joint pain": "joint pain",
            "urinary symptoms": "burning urination",
        }
        for topic, mapped in topic_map.items():
            if topic in q:
                return mapped
        return q

    def _query_tokens(self, text: str) -> set[str]:
        tokens = {
            w
            for w in _WORD_RE.findall(text.lower())
            if w not in SKIP_WORDS and w not in QUERY_STOP_WORDS
        }
        return {t for t in tokens if len(t) > 2 or t in ("ut",)}

    def _build_match_pool(self, q: str, q_tokens: set[str]) -> set[int]:
        pool: set[int] = set()

        for token in q_tokens:
            pool |= self._symptom_index.get(token, set())

        if q in self._symptom_index:
            pool |= self._symptom_index[q]

        q_compact = q.replace(" ", "")
        for idx, name in enumerate(self._condition_names_lower):
            if q == name or q in name or name in q:
                pool.add(idx)
            name_compact = name.replace("-", "").replace(" ", "")
            if q_compact and (q_compact in name_compact or name_compact in q_compact):
                pool.add(idx)

        for name in get_close_matches(q, self._condition_names_lower, n=8, cutoff=0.5):
            pool.add(self._name_to_idx[name])

        if not pool:
            for idx, cond in enumerate(self._condition_names_lower):
                if SequenceMatcher(None, q, cond).ratio() >= 0.62:
                    pool.add(idx)

        if not pool and q_tokens:
            for token in q_tokens:
                pool |= self._symptom_index.get(token, set())

        return pool

    def _score_candidate(
        self,
        idx: int,
        q: str,
        q_tokens: set[str],
        scores: dict[int, float],
    ) -> None:
        cond = self._condition_names_lower[idx]
        blob = self._search_blobs[idx]
        token_hits = sum(1 for token in q_tokens if token in blob)

        if cond == q:
            scores[idx] += 25
        elif cond in q or q in cond:
            scores[idx] += 14

        cond_compact = cond.replace("-", "").replace(" ", "")
        q_compact = q.replace(" ", "")
        if q_compact and (q_compact in cond_compact or cond_compact in q_compact):
            scores[idx] += 16

        for word in self._condition_words[idx]:
            if word in q_tokens:
                scores[idx] += 4

        if q in blob:
            scores[idx] += 10

        if q_tokens:
            for token in q_tokens:
                if token in blob:
                    df = self._token_df.get(token, 1)
                    scores[idx] += 6.0 / (1 + math.log1p(df))
            if len(q_tokens) >= 2 and token_hits >= 2:
                scores[idx] += 8

        primary = self._primary_symptom_token[idx]
        if primary and primary in q_tokens:
            scores[idx] += 12
        if primary and q == primary:
            scores[idx] += 8

        for token in q_tokens:
            if idx in self._symptom_index.get(token, set()):
                df = self._token_df.get(token, 1)
                scores[idx] += 2.5 / (1 + math.log1p(df))

        for phrase in self._keyword_phrases[idx]:
            if phrase in q or q in phrase:
                scores[idx] += 9
            elif SequenceMatcher(None, q, phrase).ratio() >= 0.72:
                scores[idx] += 7

        for phrase in self._symptom_phrases[idx]:
            if phrase in q or q in phrase:
                scores[idx] += 8
            elif len(q) >= 5 and SequenceMatcher(None, q, phrase).ratio() >= 0.78:
                scores[idx] += 5

        if "breath" in q or "breathing" in q:
            if any(t in blob for t in ("breath", "breathing", "wheez", "wheezing")):
                scores[idx] += 8

    def _duration_matches_idx(self, user_days: int, idx: int) -> bool:
        low, high, category = self._duration_bounds[idx]
        if category == "varies":
            return True
        if category == "chronic":
            return user_days >= 30
        if category == "acute":
            return user_days <= 3
        if category == "other":
            return True
        if low is not None and high is not None:
            margin = max(2, int((high - low) * 0.25))
            return (low - margin) <= user_days <= (high + margin)
        return True

    def _filter_by_duration(self, ids: list[int], user_days: int) -> tuple[list[int], bool]:
        matched = [i for i in ids if self._duration_matches_idx(user_days, i)]
        return (matched, False) if matched else (ids, True)

    def _filter_by_severity(self, ids: list[int], severity: str) -> tuple[list[int], bool]:
        sev = severity.lower()
        matched = [i for i in ids if sev in self._severity_lower_list[i]]
        if matched:
            return matched, False
        return ids, True

    def _closest_condition_names(self, text: str, limit: int = 4) -> list[str]:
        q = text.strip().lower()
        if not q:
            return []
        hits = get_close_matches(q, self._condition_names_lower, n=limit, cutoff=0.45)
        if hits:
            return [self._conditions[self._name_to_idx[h]] for h in hits]

        q_tokens = self._query_tokens(q)
        scored: list[tuple[float, str]] = []
        for name, low in zip(self._conditions, self._condition_names_lower):
            ratio = SequenceMatcher(None, q, low).ratio()
            if any(tok in low for tok in q_tokens):
                ratio += 0.25
            if ratio >= 0.35:
                scored.append((ratio, name))
        scored.sort(reverse=True)
        return [name for _, name in scored[:limit]]

    def _compute_match_scores(self, symptom: str) -> dict[int, float]:
        raw = symptom.strip()
        if not raw:
            return {}

        q = self._normalize_symptom_query(raw)
        if not q:
            return {}

        scores: dict[int, float] = defaultdict(float)
        q_tokens = self._query_tokens(q)
        pool = self._build_match_pool(q, q_tokens)

        for idx in pool:
            self._score_candidate(idx, q, q_tokens, scores)

        if not scores:
            parts = [p.strip() for p in re.split(r"\band\b|\bwith\b|\balso\b|,|;", q) if p.strip()]
            if len(parts) > 1:
                for part in parts:
                    if len(part) < 3:
                        continue
                    for idx, sc in self._compute_match_scores(part).items():
                        scores[idx] += sc * 0.85

        return dict(scores)

    def _ids_from_scores(self, scores: dict[int, float], *, min_score: float = 1.5) -> list[int]:
        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        best_score = ranked[0][1]
        cutoff = max(min_score, best_score * 0.62)
        return [idx for idx, score in ranked if score >= cutoff][:8]

    def _rank_candidates(self, session: ConsultationSession, ids: list[int]) -> list[int]:
        if not ids:
            return []

        query_tokens: set[str] = set()
        for query in session.symptom_queries or [session.last_symptom]:
            if query:
                query_tokens |= self._query_tokens(self._normalize_symptom_query(query))

        ranked: dict[int, float] = {}
        for idx in ids:
            score = session.match_scores.get(idx, 0.0)

            if session.user_days is not None:
                score += 10 if self._duration_matches_idx(session.user_days, idx) else -5

            if session.severity:
                sev_blob = self._severity_lower_list[idx]
                if session.severity in sev_blob:
                    score += 8
                elif session.severity == "severe" and "severe" not in sev_blob:
                    score -= 6

            if session.trend == "worse" and "severe" in self._severity_lower_list[idx]:
                score += 4
            elif session.trend == "improving" and "mild" in self._severity_lower_list[idx]:
                score += 3

            if session.fever_level == "high" and "fever" in self._search_blobs[idx]:
                score += 5
            elif session.fever_level == "low" and "mild" in self._severity_lower_list[idx]:
                score += 2

            primary = self._primary_symptom_token[idx]
            if primary and primary in query_tokens:
                score += 10

            for token in query_tokens:
                if token in self._search_blobs[idx]:
                    df = self._token_df.get(token, 1)
                    score += 3.5 / (1 + math.log1p(df))

            score += self._category_fit_score(session, idx, query_tokens)

            ranked[idx] = score

        return sorted(ids, key=lambda i: (-ranked.get(i, 0.0), i))

    def _category_fit_score(
        self, session: ConsultationSession, idx: int, query_tokens: set[str]
    ) -> float:
        if not query_tokens or len(session.symptom_queries) != 1:
            return 0.0

        if not query_tokens.issubset(GENERIC_SYMPTOM_TOKENS):
            return 0.0

        cat = self._categories[idx]
        blob = self._search_blobs[idx]
        bonus = 0.0

        if "fever" in query_tokens:
            if cat in ("respiratory", "infectious"):
                bonus += 8
            if cat == "urological" and "urin" not in session.last_symptom.lower():
                bonus -= 12
            if cat in ("dermatological", "mental_health", "musculoskeletal"):
                bonus -= 4

        if "headache" in query_tokens and cat in ("neurological", "general", "respiratory"):
            bonus += 6

        if "cough" in query_tokens and cat == "respiratory":
            bonus += 8

        if "chest" in session.last_symptom.lower() and cat == "cardiovascular":
            bonus += 10

        if "stomach" in session.last_symptom.lower() or "abdominal" in blob:
            if cat == "gastrointestinal":
                bonus += 8

        return bonus

    def _set_candidates(self, session: ConsultationSession, ids: list[int]) -> None:
        session.candidate_ids = self._rank_candidates(session, ids)

    def match_symptom(self, symptom: str, *, min_score: float = 1.5) -> list[int]:
        scores = self._compute_match_scores(symptom)
        ids = self._ids_from_scores(scores, min_score=min_score)
        if ids:
            return ids

        q = self._normalize_symptom_query(symptom)
        q_tokens = self._query_tokens(q)
        pool = self._build_match_pool(q, q_tokens)
        if pool:
            return sorted(pool)[:6]

        return []

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
            suggestions=TOPIC_CHIPS + COMMON_SYMPTOM_CHIPS[:6] + ["I'm ready"],
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
        if session.step == "age_group":
            return self._step_age_group(session, text)
        if session.step == "duration":
            return self._step_duration(session, text)
        if session.step == "fever_check":
            return self._step_fever_check(session, text)
        if session.step == "trend":
            return self._step_trend(session, text)
        if session.step == "severity":
            return self._step_severity(session, text)
        if session.step == "red_flags":
            return self._step_red_flags(session, text)
        if session.step == "impact":
            return self._step_impact(session, text)
        if session.step == "medication":
            return self._step_medication(session, text)
        if session.step == "risk":
            return self._step_risk(session, text)
        if session.step == "aftercare":
            return self._step_aftercare(session, text)

        session.reset()
        return self.get_welcome()

    def _chips_for_step(self, step: str) -> list[str]:
        return {
            "welcome": TOPIC_CHIPS + COMMON_SYMPTOM_CHIPS[:6],
            "symptom": TOPIC_CHIPS + COMMON_SYMPTOM_CHIPS,
            "symptom_extra": ["No other symptoms"] + COMMON_SYMPTOM_CHIPS[:5],
            "age_group": AGE_CHIPS,
            "duration": DURATION_CHIPS,
            "fever_check": FEVER_CHIPS,
            "trend": TREND_CHIPS,
            "severity": SEVERITY_CHIPS,
            "red_flags": RED_FLAG_CHIPS,
            "impact": IMPACT_CHIPS,
            "medication": MEDICATION_CHIPS,
            "risk": RISK_CHIPS,
            "aftercare": ["Start a new consultation", "Thank you, goodbye"],
        }.get(step, [])

    def _has_fever_symptom(self, session: ConsultationSession) -> bool:
        return "fever" in session.last_symptom.lower()

    def _needs_red_flag_screen(self, session: ConsultationSession) -> bool:
        s = session.last_symptom.lower()
        triggers = ("chest pain", "headache", "abdominal", "breath", "confusion", "numbness")
        return any(t in s for t in triggers) or session.severity == "severe"

    def _after_symptom_extra(self, session: ConsultationSession, messages: list[str]) -> ConsultationResult:
        session.step = "age_group"
        return ConsultationResult(
            message="",
            messages=messages
            + ["Which age group are you in? This helps me tailor general guidance."],
            suggestions=AGE_CHIPS,
            input_hint="Select your age group…",
        )

    def _preview_conditions(self, ids: list[int], limit: int = 3) -> str:
        if not ids:
            return ""
        names = [self._conditions[i] for i in ids[:limit]]
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
                suggestions=TOPIC_CHIPS + COMMON_SYMPTOM_CHIPS,
                input_hint="Describe your main symptom…",
            )
        session.step = "symptom"
        return self._step_symptom(session, text)

    def _step_symptom(self, session: ConsultationSession, text: str) -> ConsultationResult:
        ids = self.match_symptom(text)
        if not ids:
            # Last resort: match any meaningful word from the user's message
            for word in _WORD_RE.findall(text.lower()):
                if len(word) > 3 and word not in SKIP_WORDS and word not in QUERY_STOP_WORDS:
                    ids = self.match_symptom(word)
                    if ids:
                        break

        if not ids:
            closest = self._closest_condition_names(text)
            messages = [
                f"I didn't find an exact match for '{text}', but I can still help.",
            ]
            if closest:
                messages.append("Did you mean: " + ", ".join(closest) + "?")
            messages.append("Try a common symptom word, or pick one of the suggestions below.")
            return ConsultationResult(
                message="",
                messages=messages,
                suggestions=(closest[:3] if closest else []) + TOPIC_CHIPS + COMMON_SYMPTOM_CHIPS,
                input_hint="Try another symptom…",
            )

        scores = self._compute_match_scores(text)
        session.match_scores = {idx: scores[idx] for idx in ids if idx in scores}
        session.symptom_queries = [text]
        session.last_symptom = text
        session.step = "symptom_extra"
        self._set_candidates(session, ids)
        ids = session.candidate_ids
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
            return self._after_symptom_extra(
                session,
                ["Understood — we'll focus on that."],
            )

        extra_ids = self.match_symptom(text)
        if extra_ids:
            combined = sorted(set(session.candidate_ids) & set(extra_ids))
            if combined:
                extra_scores = self._compute_match_scores(text)
                for idx in combined:
                    session.match_scores[idx] = session.match_scores.get(idx, 0) + extra_scores.get(idx, 0) * 0.6
                session.symptom_queries.append(text)
                self._set_candidates(session, combined)
            session.last_symptom = f"{session.last_symptom}, {text}"

        n = len(session.candidate_ids)
        return self._after_symptom_extra(
            session,
            [
                f"Got it — I've noted {text} as well.",
                f"That narrows us to {n} possible condition{'s' if n != 1 else ''}.",
            ],
        )

    def _step_age_group(self, session: ConsultationSession, text: str) -> ConsultationResult:
        low = text.lower()
        if "under" in low or "18" in low and "64" not in low:
            session.age_group = "child"
        elif "65" in low or "older" in low:
            session.age_group = "senior"
        elif "prefer" in low:
            session.age_group = "unspecified"
        else:
            session.age_group = "adult"

        age_note = {
            "child": "Thanks — I'll keep pediatric guidance in mind.",
            "senior": "Thanks — I'll consider guidance relevant for older adults.",
            "adult": "Thanks — that helps.",
            "unspecified": "No problem — we'll continue.",
        }[session.age_group]

        session.step = "duration"
        return ConsultationResult(
            message="",
            messages=[age_note, "Roughly how long have you had this symptom?"],
            suggestions=DURATION_CHIPS,
            input_hint="Number of days or pick a range…",
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
        ids, soft = self._filter_by_duration(session.candidate_ids, days)

        self._set_candidates(session, ids)
        duration_msgs = [
            f"Thanks — about {days} day{'s' if days != 1 else ''} helps a lot.",
        ]
        if soft:
            duration_msgs.append(
                "Your timeline is a bit outside typical patterns — I'll continue with the closest matches."
            )

        if self._has_fever_symptom(session):
            session.step = "fever_check"
            return ConsultationResult(
                message="",
                messages=duration_msgs
                + [
                    "Have you measured a fever recently? A rough level helps me guide you.",
                ],
                suggestions=FEVER_CHIPS,
                input_hint="Fever level if known…",
            )

        session.step = "trend"
        return ConsultationResult(
            message="",
            messages=duration_msgs
            + [
                "Since it started, would you say things are getting worse, about the same, or improving?",
            ],
            suggestions=TREND_CHIPS,
            input_hint="How symptoms are changing…",
        )

    def _step_fever_check(self, session: ConsultationSession, text: str) -> ConsultationResult:
        low = text.lower()
        if "high" in low or "102" in low or "39" in low:
            session.fever_level = "high"
            note = "A higher fever can be more concerning — I'll factor that in."
        elif "low" in low:
            session.fever_level = "low"
            note = "A mild fever is noted."
        elif "no fever" in low:
            session.fever_level = "none"
            note = "Good to know fever has settled."
        else:
            session.fever_level = "unknown"
            note = "That's okay — we'll work with what we have."

        session.step = "trend"
        return ConsultationResult(
            message="",
            messages=[
                note,
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
        ids, soft = self._filter_by_severity(session.candidate_ids, sev)
        self._set_candidates(session, ids)
        session.step = "impact"

        severity_msgs = [f"Noted — {sev} intensity."]
        if soft:
            severity_msgs.append(
                "Intensity varies a lot between conditions — I'll use the nearest matches in my guide."
            )

        if self._needs_red_flag_screen(session):
            session.needs_red_flag = True
            session.step = "red_flags"
            return ConsultationResult(
                message="",
                messages=severity_msgs
                + [
                    "Quick safety check: any sudden confusion, one-sided weakness, "
                    "or trouble breathing right now?",
                ],
                suggestions=RED_FLAG_CHIPS,
                input_hint="Yes or no…",
                emergency=True,
            )

        if sev == "severe":
            names = [self._condition_names_lower[i] for i in session.candidate_ids]
            if any(any(sc in n for sc in SEVERE_CONDITIONS) for n in names):
                return ConsultationResult(
                    message="",
                    messages=severity_msgs
                    + [
                        "Because you've marked symptoms as severe, I want to check in carefully.",
                        "Is this affecting your ability to breathe, speak, or stay awake normally?",
                    ],
                    suggestions=YES_NO_CHIPS,
                    input_hint="Yes or no…",
                    emergency=True,
                )

        return ConsultationResult(
            message="",
            messages=severity_msgs
            + [
                "Is this affecting your sleep, work, or daily activities in a major way?",
            ],
            suggestions=IMPACT_CHIPS,
            input_hint="How much it affects daily life…",
        )

    def _step_red_flags(self, session: ConsultationSession, text: str) -> ConsultationResult:
        low = text.lower()
        if "yes" in low:
            session.step = "aftercare"
            return ConsultationResult(
                message="",
                messages=[
                    "Those can be emergency warning signs.",
                    "**Please call your local emergency number or go to the nearest emergency department now.**",
                    "I can't safely continue a routine consultation when red-flag symptoms are present.",
                ],
                suggestions=["Thank you, goodbye"],
                done=True,
                emergency=True,
            )

        session.step = "impact"
        return ConsultationResult(
            message="",
            messages=[
                "Thank you for clarifying — that's reassuring for now.",
                "Is this affecting your sleep, work, or daily activities in a major way?",
            ],
            suggestions=IMPACT_CHIPS,
        )

    def _step_impact(self, session: ConsultationSession, text: str) -> ConsultationResult:
        low = text.lower()
        if "yes" in low or "lot" in low:
            session.impact = "high"
        elif "somewhat" in low or "some" in low:
            session.impact = "medium"
        else:
            session.impact = "low"

        session.step = "medication"
        return ConsultationResult(
            message="",
            messages=[
                "Are you taking any regular medicines I should keep in mind?",
                "This won't change your diagnosis here, but it helps me mention safer next steps.",
            ],
            suggestions=MEDICATION_CHIPS,
            input_hint="Medication type or 'none'…",
        )

    def _step_medication(self, session: ConsultationSession, text: str) -> ConsultationResult:
        low = text.lower()
        session.medication = text
        med_note = ""
        if "thin" in low:
            med_note = "Since you mentioned blood thinners, seek prompt care for any unusual bleeding or severe headache."
        elif "diabetes" in low:
            med_note = "With diabetes, monitor blood sugar and hydration closely while you're unwell."

        session.step = "risk"
        seen: set[str] = set()
        risks: list[str] = []
        for idx in session.candidate_ids:
            risk = self._risk_factors_list[idx]
            if risk and risk not in seen:
                seen.add(risk)
                risks.append(risk)
            if len(risks) >= 5:
                break
        risk_list = "\n".join(f"• {r}" for r in risks) if risks else "• (none listed)"

        msgs = ["Almost done — a few lifestyle factors can help narrow guidance."]
        if med_note:
            msgs.append(med_note)
        msgs.extend(
            [
                "Do any of these risk factors apply to you?",
                risk_list,
                "Reply with one that fits, or say 'none'.",
            ]
        )

        return ConsultationResult(
            message="",
            messages=msgs,
            suggestions=[*risks[:4], "None of these"] if risks else ["None"],
            input_hint="Risk factor or 'none'…",
        )

    def _step_risk(self, session: ConsultationSession, text: str) -> ConsultationResult:
        return self._finalize(session, text)

    def _finalize(self, session: ConsultationSession, risk_text: str) -> ConsultationResult:
        ids = list(session.candidate_ids)
        low = risk_text.lower()
        if low not in SKIP_WORDS and "none of" not in low:
            filtered = [i for i in ids if low in self._risk_lower_list[i]]
            if filtered:
                ids = filtered

        if not ids and session.candidate_ids:
            ids = list(session.candidate_ids)

        if not ids:
            ids = self.match_symptom(session.last_symptom, min_score=1.5)[:5]

        if not ids:
            session.reset()
            return ConsultationResult(
                message="",
                messages=[
                    "I couldn't narrow this to one strong match with those details.",
                    "Please consider speaking with a healthcare professional for a proper assessment.",
                ],
                suggestions=["Start a new consultation"],
                done=True,
            )

        ids = self._rank_candidates(session, ids)
        best = ids[0]
        condition = self._conditions[best]
        action = self._actions[best]
        source = self._sources[best]
        category = self._categories[best]
        others = len(ids) - 1
        alt_names = [self._conditions[i] for i in ids[1:3]]

        trend_note = ""
        if session.trend == "worse":
            trend_note = "Since symptoms are worsening, don't wait too long to get checked if they continue."
        elif session.trend == "improving":
            trend_note = "Even if you're improving, keep monitoring and rest as needed."

        messages = [
            f"Based on what you've told me, **{condition}** is the closest match in my guide.",
            f"**Suggested next steps:** {action}",
        ]
        if category:
            messages.append(f"**Category:** {category.replace('_', ' ').title()}")
        if source:
            messages.append(
                f"**Reference:** General public-health guidance aligned with {source} materials — not a clinical diagnosis."
            )
        if others > 0:
            if alt_names:
                messages.append(
                    f"Other close possibilities: {', '.join(alt_names)} — "
                    "a clinician can confirm with an exam."
                )
            else:
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
            matches=len(ids),
            emergency=emergency,
            source=source or None,
            category=category or None,
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
