"""Team name canonicalization helpers (NO pandas dependency).

This module is safe to import in Streamlit Cloud even if pandas isn't available at import time.
It provides:
- canonical_team(name): map a raw team string to a canonical label using substring rules.
- add_canonical_column(df, source_col, target_col="TEAM_CANON"): works with pandas DataFrame *if* pandas is installed,
  but pandas is NOT imported by this module. It will operate on any object that supports
  df.copy(), df[col].apply(func), and df[new_col] assignment (i.e., a pandas DataFrame).

Rules included (as provided):
- "Perugia"    -> PERUGIA
- "Modena"     -> MODENA
- "Trentino"   -> TRENTO
- "Milano"     -> MILANO
- "Piacenza"   -> PIACENZA
- "Civitanova" -> CIVITANOVA
- "Verona"     -> VERONA
- "Cuneo"      -> CUNEO
- "Cisterna"   -> CISTERNA
- "Padova"     -> PADOVA
- "Monza"      -> MONZA
- "Grotta"     -> GROTTA
"""

import re
import unicodedata
from typing import List, Tuple, Optional, Any

# Canonical team name -> list of substrings that, if found, map to that canonical name.
TEAM_RULES = [
    ("PERUGIA",    ["perugia", "sir", "susa scai"]),
    ("MODENA",     ["modena", "valsa group", "valsa"]),
    ("TRENTO",     ["trentino", "trento", "itas"]),
    ("MILANO",     ["milano", "allianz"]),
    ("PIACENZA",   ["piacenza", "gas sales", "bluenergy", "gas sales bluenergy p", "bluenergy p"]),
    ("CIVITANOVA", ["civitanova", "lube", "cucine lube"]),
    ("VERONA",     ["verona", "rana"]),
    ("CUNEO",      ["cuneo", "s bernardo", "s. bernardo", "acqua s bernardo", "ma acqua"]),
    ("CISTERNA",   ["cisterna", "cisterna volley"]),
    ("PADOVA",     ["padova", "sonepar"]),
    ("MONZA",      ["monza", "vero volley"]),
    ("GROTTA",     ["grotta", "grottazzolina", "yuasa", "yuasa battery"]),

def _normalize_text(value: Optional[str]) -> str:
    """Normalize a team string: lower, remove accents, strip punctuation, collapse spaces."""
    if value is None:
        return ""
    s = str(value).strip().lower()

    # Remove accents/diacritics
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # Replace non alphanumeric with spaces
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def canonical_team(team_name: Optional[str]) -> str:
    """Return canonical team label based on substring rules."""
    normalized = _normalize_text(team_name)
    for canon, needles in TEAM_RULES:
        if any(needle in normalized for needle in needles):
            return canon
    if team_name is None:
        return "UNKNOWN"
    return str(team_name).strip().upper()

def add_canonical_column(df: Any, source_col: str, target_col: str = "TEAM_CANON"):
    """Return a copy of df with an extra canonical team column.

    Works with pandas DataFrames without importing pandas here.
    """
    out = df.copy()
    out[target_col] = out[source_col].apply(canonical_team)
    return out

def list_unmapped_values(df: Any, source_col: str):
    """Return unique source values that are NOT mapped to known canonical teams.

    Requires pandas-like operations; will work on a pandas DataFrame.
    """
    tmp = df[source_col].astype(str)
    canon = tmp.apply(canonical_team)
    known = set([c for c, _ in TEAM_RULES])
    mask = ~canon.isin(known)
    return tmp[mask].dropna().drop_duplicates().sort_values(ignore_index=True)
