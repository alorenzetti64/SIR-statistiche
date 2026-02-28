
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Volley App", layout="wide")

DB_PATH = Path(__file__).resolve().parent / "volley.db"
DB_URL = f"sqlite:///{DB_PATH.as_posix()}"
engine = create_engine(DB_URL, future=True)

# =========================
# HELPERS
# =========================
def norm(s: str | None) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = " ".join(s.split())
    return s


def build_match_key(team_a: str, team_b: str, competition: str | None, phase: str, round_number: int) -> str:
    return f"{norm(team_a)}|{norm(team_b)}|{norm(competition)}|{phase}{round_number:02d}"


def extract_round_code(filename: str) -> tuple[str, int]:
    for start in (12, 13):
        if len(filename) >= start + 3:
            code = filename[start : start + 3]
            if re.match(r"^[AR]\d{2}$", code):
                return code[0], int(code[1:3])

    m = re.search(r"([AR]\d{2})", filename)
    if m:
        code = m.group(1)
        return code[0], int(code[1:3])

    raise ValueError("Codice giornata non trovato nel filename (atteso A01 / R06).")


def parse_dvw_minimal(dvw_text: str) -> dict:
    season = None
    competition = None
    teams: list[str] = []

    lines = dvw_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == "[3MATCH]":
            if i + 1 < len(lines):
                parts = lines[i + 1].split(";")
                if len(parts) >= 4:
                    season = parts[2].strip() or None
                    competition = parts[3].strip() or None
            i += 1

        if line == "[3TEAMS]":
            j = i + 1
            while j < len(lines):
                row = lines[j].strip()
                if not row or row.startswith("["):
                    break
                parts = row.split(";")
                if len(parts) >= 2:
                    teams.append(parts[1].strip())
                j += 1
            i = j
            continue

        i += 1

    team_a = teams[0] if len(teams) >= 1 else ""
    team_b = teams[1] if len(teams) >= 2 else ""

    return {"season": season, "competition": competition, "team_a": team_a, "team_b": team_b}


def extract_scout_lines(dvw_text: str) -> list[str]:
    lines = dvw_text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == "[3SCOUT]":
            start = idx + 1
            break
    if start is None:
        return []

    out = []
    for line in lines[start:]:
        if line.strip().startswith("["):
            break
        s = line.strip()
        if not s:
            continue

        k = 0
        while k < len(s) and ord(s[k]) < 32:
            k += 1
        s2 = s[k:].lstrip()
        if not s2:
            continue

        if s2[0] in ("*", "a"):
            out.append(s2)

    return out


def code6(line: str) -> str:
    if not line:
        return ""
    i = 0
    while i < len(line) and ord(line[i]) < 32:
        i += 1
    s = line[i:].lstrip()
    return s[:6]


def is_home_rece(c6: str) -> bool:
    return len(c6) >= 5 and c6[0] == "*" and c6[3:5] in ("RQ", "RM")


def is_away_rece(c6: str) -> bool:
    return len(c6) >= 5 and c6[0] == "a" and c6[3:5] in ("RQ", "RM")


def is_home_spin(c6: str) -> bool:
    return len(c6) >= 5 and c6[0] == "*" and c6[3:5] == "RQ"


def is_away_spin(c6: str) -> bool:
    return len(c6) >= 5 and c6[0] == "a" and c6[3:5] == "RQ"


def is_home_float(c6: str) -> bool:
    return len(c6) >= 5 and c6[0] == "*" and c6[3:5] == "RM"


def is_away_float(c6: str) -> bool:
    return len(c6) >= 5 and c6[0] == "a" and c6[3:5] == "RM"


def is_serve(c6: str) -> bool:
    # include SQ/SM (battuta)
    return len(c6) >= 5 and c6[0] in ("*", "a") and c6[3:5] in ("SQ", "SM")


def is_home_point(c6: str) -> bool:
    return c6.startswith("*p")


def is_away_point(c6: str) -> bool:
    return c6.startswith("ap")


def is_attack(c6: str, prefix: str) -> bool:
    return len(c6) >= 6 and c6[0] == prefix and c6[3] == "A"


def first_attack_after_reception_is_winner(rally: list[str], prefix: str) -> bool:
    rece_idx = None
    for i, c in enumerate(rally):
        if len(c) >= 5 and c[0] == prefix and c[3:5] in ("RQ", "RM"):
            rece_idx = i
            break
    if rece_idx is None:
        return False

    for c in rally[rece_idx + 1 :]:
        if is_attack(c, prefix):
            return len(c) >= 6 and c[5] == "#"
    return False


# =========================
# SIDE OUT filters
# =========================
PLAYABLE_RECV = {"#", "+", "!", "-"}  # giocabili: scarta '=' e '/'
GOOD_RECV = {"#", "+"}               # buone
EXC_RECV = {"!"}                     # esclamativa
NEG_RECV = {"-"}                     # negativa

def is_home_rece_playable(c6: str) -> bool:
    return len(c6) >= 6 and c6[0] == "*" and c6[3:5] in ("RQ", "RM") and c6[5] in PLAYABLE_RECV

def is_away_rece_playable(c6: str) -> bool:
    return len(c6) >= 6 and c6[0] == "a" and c6[3:5] in ("RQ", "RM") and c6[5] in PLAYABLE_RECV

def is_home_rece_good(c6: str) -> bool:
    return len(c6) >= 6 and c6[0] == "*" and c6[3:5] in ("RQ", "RM") and c6[5] in GOOD_RECV

def is_away_rece_good(c6: str) -> bool:
    return len(c6) >= 6 and c6[0] == "a" and c6[3:5] in ("RQ", "RM") and c6[5] in GOOD_RECV

def is_home_rece_exc(c6: str) -> bool:
    return len(c6) >= 6 and c6[0] == "*" and c6[3:5] in ("RQ", "RM") and c6[5] in EXC_RECV

def is_away_rece_exc(c6: str) -> bool:
    return len(c6) >= 6 and c6[0] == "a" and c6[3:5] in ("RQ", "RM") and c6[5] in EXC_RECV

def is_home_rece_neg(c6: str) -> bool:
    return len(c6) >= 6 and c6[0] == "*" and c6[3:5] in ("RQ", "RM") and c6[5] in NEG_RECV

def is_away_rece_neg(c6: str) -> bool:
    return len(c6) >= 6 and c6[0] == "a" and c6[3:5] in ("RQ", "RM") and c6[5] in NEG_RECV


def pct(wins: int, attempts: int) -> float:
    return (wins / attempts * 100.0) if attempts else 0.0

# =========================
# ROSTER / RUOLI HELPERS
# =========================
def fix_team_name(name: str) -> str:
    """
    Normalizza nome squadra per match con roster.
    (stesse regole usate nelle pagine Break/Confronto)
    """
    n = " ".join((name or "").split())
    nl = n.lower()
    if nl.startswith("gas sales bluenergy p"):
        return "Gas Sales Bluenergy Piacenza"
    if "grottazzolina" in nl:
        return "Yuasa Battery Grottazzolina"
    return n


def team_norm(name: str) -> str:
    """Chiave stabile (lower + pulizia) per join DB."""
    n = fix_team_name(name).strip().lower()
    n = re.sub(r"[^a-z0-9\s]", " ", n)
    return " ".join(n.split())


def serve_player_number(c6: str) -> int | None:
    """
    Estrae numero maglia dal code6 della battuta:
    es: *06SQ- -> 6 ; a08SM+ -> 8
    """
    if not c6 or len(c6) < 3:
        return None
    if c6[0] not in ("*", "a"):
        return None
    digits = c6[1:3]
    if not digits.isdigit():
        return None
    return int(digits)


def serve_sign(c6: str) -> str:
    """Valutazione battuta: 6° carattere ( -, +, !, /, #, = )"""
    return c6[5] if c6 and len(c6) >= 6 else ""



def compute_counts_from_scout(scout_lines: list[str]) -> dict:
    # --- helper BT (break tendency) dal testo SERVIZIO ---
    def detect_bt(raw_line: str) -> str | None:
        if not raw_line:
            return None
        s = raw_line.strip()

        # Priorità: simboli tra parentesi quadre
        if "[-]" in s:
            return "NEG"
        if "[+]" in s:
            return "POS"
        if "[!]" in s:
            return "EXC"
        if "[½]" in s:
            return "HALF"

        # Varianti non-bracketed
        if "½" in s or "1/2" in s or "0.5" in s:
            return "HALF"

        tail = s[-6:]  # spesso il segno è vicino alla fine
        if "!" in tail:
            return "EXC"
        if "+" in tail:
            return "POS"
        if "-" in tail:
            return "NEG"

        return None

    # --- costruzione rallies: teniamo (c6, raw) per non perdere i segni BT ---
    rallies: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []

    for raw in scout_lines:
        c = code6(raw)
        if not c:
            continue

        if is_serve(c):
            if current:
                rallies.append(current)
            current = [(c, raw)]
            continue

        if not current:
            continue

        current.append((c, raw))

        if is_home_point(c) or is_away_point(c):
            rallies.append(current)
            current = []

    # =========================
    # SIDE OUT counters
    # =========================
    so_home_attempts = so_home_wins = 0
    so_away_attempts = so_away_wins = 0

    bp_home_attempts = bp_home_wins = 0
    bp_away_attempts = bp_away_wins = 0

    so_spin_home_attempts = so_spin_home_wins = 0
    so_spin_away_attempts = so_spin_away_wins = 0

    so_float_home_attempts = so_float_home_wins = 0
    so_float_away_attempts = so_float_away_wins = 0

    so_dir_home_wins = 0
    so_dir_away_wins = 0

    so_play_home_attempts = so_play_home_wins = 0
    so_play_away_attempts = so_play_away_wins = 0

    so_good_home_attempts = so_good_home_wins = 0
    so_good_away_attempts = so_good_away_wins = 0

    so_exc_home_attempts = so_exc_home_wins = 0
    so_exc_away_attempts = so_exc_away_wins = 0

    so_neg_home_attempts = so_neg_home_wins = 0
    so_neg_away_attempts = so_neg_away_wins = 0

    # =========================
    # BREAK "GIOCATO" + BT counters
    # =========================
    bp_play_home_attempts = bp_play_home_wins = 0
    bp_play_away_attempts = bp_play_away_wins = 0

    bt_neg_home = bt_pos_home = bt_exc_home = bt_half_home = 0
    bt_neg_away = bt_pos_away = bt_exc_away = bt_half_away = 0

    for r in rallies:
        first_c6, first_raw = r[0]
        home_served = first_c6.startswith("*")
        away_served = first_c6.startswith("a")

        home_point = any(is_home_point(c6) for c6, _ in r)
        away_point = any(is_away_point(c6) for c6, _ in r)

        home_rece = any(is_home_rece(c6) for c6, _ in r)
        away_rece = any(is_away_rece(c6) for c6, _ in r)

        home_spin = any(is_home_spin(c6) for c6, _ in r)
        away_spin = any(is_away_spin(c6) for c6, _ in r)

        home_float = any(is_home_float(c6) for c6, _ in r)
        away_float = any(is_away_float(c6) for c6, _ in r)

        # SideOut totale
        if home_rece:
            so_home_attempts += 1
            if home_point:
                so_home_wins += 1

        if away_rece:
            so_away_attempts += 1
            if away_point:
                so_away_wins += 1

        # SPIN
        if home_spin:
            so_spin_home_attempts += 1
            if home_point:
                so_spin_home_wins += 1

        if away_spin:
            so_spin_away_attempts += 1
            if away_point:
                so_spin_away_wins += 1

        # FLOAT
        if home_float:
            so_float_home_attempts += 1
            if home_point:
                so_float_home_wins += 1

        if away_float:
            so_float_away_attempts += 1
            if away_point:
                so_float_away_wins += 1

        # DIRETTO
        rally_c6 = [c6 for c6, _ in r]
        if home_rece and home_point and first_attack_after_reception_is_winner(rally_c6, "*"):
            so_dir_home_wins += 1

        if away_rece and away_point and first_attack_after_reception_is_winner(rally_c6, "a"):
            so_dir_away_wins += 1

        # GIOCATO (# + ! -)
        home_play = any(is_home_rece_playable(c6) for c6, _ in r)
        away_play = any(is_away_rece_playable(c6) for c6, _ in r)

        if home_play:
            so_play_home_attempts += 1
            if home_point:
                so_play_home_wins += 1

        if away_play:
            so_play_away_attempts += 1
            if away_point:
                so_play_away_wins += 1

        # BUONA (#,+)
        home_good = any(is_home_rece_good(c6) for c6, _ in r)
        away_good = any(is_away_rece_good(c6) for c6, _ in r)

        if home_good:
            so_good_home_attempts += 1
            if home_point:
                so_good_home_wins += 1

        if away_good:
            so_good_away_attempts += 1
            if away_point:
                so_good_away_wins += 1

        # ESCLAMATIVA (!)
        home_exc = any(is_home_rece_exc(c6) for c6, _ in r)
        away_exc = any(is_away_rece_exc(c6) for c6, _ in r)

        if home_exc:
            so_exc_home_attempts += 1
            if home_point:
                so_exc_home_wins += 1

        if away_exc:
            so_exc_away_attempts += 1
            if away_point:
                so_exc_away_wins += 1

        # NEGATIVA (-)
        home_neg = any(is_home_rece_neg(c6) for c6, _ in r)
        away_neg = any(is_away_rece_neg(c6) for c6, _ in r)

        if home_neg:
            so_neg_home_attempts += 1
            if home_point:
                so_neg_home_wins += 1

        if away_neg:
            so_neg_away_attempts += 1
            if away_point:
                so_neg_away_wins += 1

        # Break totale
        if home_served:
            bp_home_attempts += 1
            if home_point:
                bp_home_wins += 1

        if away_served:
            bp_away_attempts += 1
            if away_point:
                bp_away_wins += 1

        # Break giocato + BT
        bt = detect_bt(first_raw)
        if bt is not None:
            if home_served:
                bp_play_home_attempts += 1
                if home_point:
                    bp_play_home_wins += 1

                if bt == "NEG":
                    bt_neg_home += 1
                elif bt == "POS":
                    bt_pos_home += 1
                elif bt == "EXC":
                    bt_exc_home += 1
                elif bt == "HALF":
                    bt_half_home += 1

            if away_served:
                bp_play_away_attempts += 1
                if away_point:
                    bp_play_away_wins += 1

                if bt == "NEG":
                    bt_neg_away += 1
                elif bt == "POS":
                    bt_pos_away += 1
                elif bt == "EXC":
                    bt_exc_away += 1
                elif bt == "HALF":
                    bt_half_away += 1

    return {
        "so_home_attempts": so_home_attempts,
        "so_home_wins": so_home_wins,
        "so_away_attempts": so_away_attempts,
        "so_away_wins": so_away_wins,
        "sideout_home_pct": pct(so_home_wins, so_home_attempts),
        "sideout_away_pct": pct(so_away_wins, so_away_attempts),

        "bp_home_attempts": bp_home_attempts,
        "bp_home_wins": bp_home_wins,
        "bp_away_attempts": bp_away_attempts,
        "bp_away_wins": bp_away_wins,
        "break_home_pct": pct(bp_home_wins, bp_home_attempts),
        "break_away_pct": pct(bp_away_wins, bp_away_attempts),

        "so_spin_home_attempts": so_spin_home_attempts,
        "so_spin_home_wins": so_spin_home_wins,
        "so_spin_away_attempts": so_spin_away_attempts,
        "so_spin_away_wins": so_spin_away_wins,

        "so_float_home_attempts": so_float_home_attempts,
        "so_float_home_wins": so_float_home_wins,
        "so_float_away_attempts": so_float_away_attempts,
        "so_float_away_wins": so_float_away_wins,

        "so_dir_home_wins": so_dir_home_wins,
        "so_dir_away_wins": so_dir_away_wins,

        "so_play_home_attempts": so_play_home_attempts,
        "so_play_home_wins": so_play_home_wins,
        "so_play_away_attempts": so_play_away_attempts,
        "so_play_away_wins": so_play_away_wins,

        "so_good_home_attempts": so_good_home_attempts,
        "so_good_home_wins": so_good_home_wins,
        "so_good_away_attempts": so_good_away_attempts,
        "so_good_away_wins": so_good_away_wins,

        "so_exc_home_attempts": so_exc_home_attempts,
        "so_exc_home_wins": so_exc_home_wins,
        "so_exc_away_attempts": so_exc_away_attempts,
        "so_exc_away_wins": so_exc_away_wins,

        "so_neg_home_attempts": so_neg_home_attempts,
        "so_neg_home_wins": so_neg_home_wins,
        "so_neg_away_attempts": so_neg_away_attempts,
        "so_neg_away_wins": so_neg_away_wins,

        # NEW: Break giocato + BT
        "bp_play_home_attempts": bp_play_home_attempts,
        "bp_play_home_wins": bp_play_home_wins,
        "bp_play_away_attempts": bp_play_away_attempts,
        "bp_play_away_wins": bp_play_away_wins,

        "bt_neg_home": bt_neg_home,
        "bt_pos_home": bt_pos_home,
        "bt_exc_home": bt_exc_home,
        "bt_half_home": bt_half_home,

        "bt_neg_away": bt_neg_away,
        "bt_pos_away": bt_pos_away,
        "bt_exc_away": bt_exc_away,
        "bt_half_away": bt_half_away,
    }


# =========================
# DB INIT + MIGRATION
# =========================
def init_db():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                phase TEXT,
                round_number INTEGER,
                season TEXT,
                competition TEXT,
                team_a TEXT,
                team_b TEXT,
                n_azioni INTEGER,
                preview TEXT,
                scout_text TEXT,
                match_key TEXT UNIQUE,
                created_at TEXT,

                so_home_attempts INTEGER,
                so_home_wins INTEGER,
                so_away_attempts INTEGER,
                so_away_wins INTEGER,
                sideout_home_pct REAL,
                sideout_away_pct REAL,

                bp_home_attempts INTEGER,
                bp_home_wins INTEGER,
                bp_away_attempts INTEGER,
                bp_away_wins INTEGER,
                break_home_pct REAL,
                break_away_pct REAL,

                so_spin_home_attempts INTEGER,
                so_spin_home_wins INTEGER,
                so_spin_away_attempts INTEGER,
                so_spin_away_wins INTEGER,

                so_float_home_attempts INTEGER,
                so_float_home_wins INTEGER,
                so_float_away_attempts INTEGER,
                so_float_away_wins INTEGER,

                so_dir_home_wins INTEGER,
                so_dir_away_wins INTEGER,

                so_play_home_attempts INTEGER,
                so_play_home_wins INTEGER,
                so_play_away_attempts INTEGER,
                so_play_away_wins INTEGER,

                so_good_home_attempts INTEGER,
                so_good_home_wins INTEGER,
                so_good_away_attempts INTEGER,
                so_good_away_wins INTEGER,

                so_exc_home_attempts INTEGER,
                so_exc_home_wins INTEGER,
                so_exc_away_attempts INTEGER,
                so_exc_away_wins INTEGER,

                so_neg_home_attempts INTEGER,
                so_neg_home_wins INTEGER,
                so_neg_away_attempts INTEGER,
                so_neg_away_wins INTEGER
            )
        """))

        cols_to_add = [
            ("scout_text", "TEXT"),
            ("preview", "TEXT"),
            ("n_azioni", "INTEGER"),

            ("so_home_attempts", "INTEGER"),
            ("so_home_wins", "INTEGER"),
            ("so_away_attempts", "INTEGER"),
            ("so_away_wins", "INTEGER"),
            ("sideout_home_pct", "REAL"),
            ("sideout_away_pct", "REAL"),

            ("bp_home_attempts", "INTEGER"),
            ("bp_home_wins", "INTEGER"),
            ("bp_away_attempts", "INTEGER"),
            ("bp_away_wins", "INTEGER"),
            ("break_home_pct", "REAL"),
            ("break_away_pct", "REAL"),

            ("so_spin_home_attempts", "INTEGER"),
            ("so_spin_home_wins", "INTEGER"),
            ("so_spin_away_attempts", "INTEGER"),
            ("so_spin_away_wins", "INTEGER"),

            ("so_float_home_attempts", "INTEGER"),
            ("so_float_home_wins", "INTEGER"),
            ("so_float_away_attempts", "INTEGER"),
            ("so_float_away_wins", "INTEGER"),

            ("so_dir_home_wins", "INTEGER"),
            ("so_dir_away_wins", "INTEGER"),

            ("so_play_home_attempts", "INTEGER"),
            ("so_play_home_wins", "INTEGER"),
            ("so_play_away_attempts", "INTEGER"),
            ("so_play_away_wins", "INTEGER"),

            ("so_good_home_attempts", "INTEGER"),
            ("so_good_home_wins", "INTEGER"),
            ("so_good_away_attempts", "INTEGER"),
            ("so_good_away_wins", "INTEGER"),

            ("so_exc_home_attempts", "INTEGER"),
            ("so_exc_home_wins", "INTEGER"),
            ("so_exc_away_attempts", "INTEGER"),
            ("so_exc_away_wins", "INTEGER"),

            ("so_neg_home_attempts", "INTEGER"),
            ("so_neg_home_wins", "INTEGER"),
            ("so_neg_away_attempts", "INTEGER"),
            ("so_neg_away_wins", "INTEGER"),
        ]
        for col, coltype in cols_to_add:
            try:
                conn.execute(text(f"ALTER TABLE matches ADD COLUMN {col} {coltype}"))
            except Exception:
                pass

        # --- roster (ruoli giocatori) ---
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS roster (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                season TEXT,
                team_raw TEXT,
                team_norm TEXT,
                jersey_number INTEGER,
                player_name TEXT,
                role TEXT,
                created_at TEXT,
                UNIQUE(season, team_norm, jersey_number)
            )
        """))

def render_import(admin_mode: bool):
    st.header("Import multiplo DVW (settimana)")

    if not admin_mode:
        st.warning("Accesso riservato allo staff (admin).")
        return

    uploaded_files = st.file_uploader(
        "Carica uno o più file .dvw",
        type=["dvw"],
        accept_multiple_files=True
    )

    st.divider()
    st.subheader("Elimina un import (dal database)")
    with engine.begin() as conn:
        del_rows = conn.execute(text("""
            SELECT id, filename, team_a, team_b, phase, round_number, created_at
            FROM matches
            ORDER BY id DESC
            LIMIT 200
        """)).mappings().all()

    if del_rows:
        def label(r):
            rn = int(r.get("round_number") or 0)
            ph = r.get("phase") or ""
            return f"[id {r['id']}] {r['filename']} — {r.get('team_a','')} vs {r.get('team_b','')} ({ph}{rn:02d})"

        selected = st.selectbox("Seleziona il match da eliminare", del_rows, format_func=label)
        confirm = st.checkbox("Confermo: voglio eliminare questo match dal DB", value=False)

        if st.button("Elimina selezionato", disabled=not confirm):
            with engine.begin() as conn:
                conn.execute(text("DELETE FROM matches WHERE id = :id"), {"id": selected["id"]})
            st.success("Eliminato dal DB.")
            st.rerun()
    else:
        st.info("Nessun match da eliminare.")

    st.divider()

    if not uploaded_files:
        st.info("Seleziona uno o più file DVW per importare.")
        return

    st.write(f"File selezionati: {len(uploaded_files)}")

    if st.button("Importa tutti (con dedup/upssert)"):
        saved = 0
        errors = 0
        details = []

        sql_upsert = """
        INSERT INTO matches
        (filename, phase, round_number, season, competition, team_a, team_b,
         n_azioni, preview, scout_text, match_key, created_at,

         so_home_attempts, so_home_wins, so_away_attempts, so_away_wins,
         sideout_home_pct, sideout_away_pct,

         bp_home_attempts, bp_home_wins, bp_away_attempts, bp_away_wins,
         break_home_pct, break_away_pct,

         so_spin_home_attempts, so_spin_home_wins, so_spin_away_attempts, so_spin_away_wins,
         so_float_home_attempts, so_float_home_wins, so_float_away_attempts, so_float_away_wins,
         so_dir_home_wins, so_dir_away_wins,

         so_play_home_attempts, so_play_home_wins, so_play_away_attempts, so_play_away_wins,
         so_good_home_attempts, so_good_home_wins, so_good_away_attempts, so_good_away_wins,
         so_exc_home_attempts, so_exc_home_wins, so_exc_away_attempts, so_exc_away_wins,
         so_neg_home_attempts, so_neg_home_wins, so_neg_away_attempts, so_neg_away_wins,

         bp_play_home_attempts, bp_play_home_wins, bp_play_away_attempts, bp_play_away_wins,
         bt_neg_home, bt_pos_home, bt_exc_home, bt_half_home,
         bt_neg_away, bt_pos_away, bt_exc_away, bt_half_away
        )
        VALUES
        (:filename, :phase, :round_number, :season, :competition, :team_a, :team_b,
         :n_azioni, :preview, :scout_text, :match_key, :created_at,

         :so_home_attempts, :so_home_wins, :so_away_attempts, :so_away_wins,
         :sideout_home_pct, :sideout_away_pct,

         :bp_home_attempts, :bp_home_wins, :bp_away_attempts, :bp_away_wins,
         :break_home_pct, :break_away_pct,

         :so_spin_home_attempts, :so_spin_home_wins, :so_spin_away_attempts, :so_spin_away_wins,
         :so_float_home_attempts, :so_float_home_wins, :so_float_away_attempts, :so_float_away_wins,
         :so_dir_home_wins, :so_dir_away_wins,

         :so_play_home_attempts, :so_play_home_wins, :so_play_away_attempts, :so_play_away_wins,
         :so_good_home_attempts, :so_good_home_wins, :so_good_away_attempts, :so_good_away_wins,
         :so_exc_home_attempts, :so_exc_home_wins, :so_exc_away_attempts, :so_exc_away_wins,
         :so_neg_home_attempts, :so_neg_home_wins, :so_neg_away_attempts, :so_neg_away_wins,

         :bp_play_home_attempts, :bp_play_home_wins, :bp_play_away_attempts, :bp_play_away_wins,
         :bt_neg_home, :bt_pos_home, :bt_exc_home, :bt_half_home,
         :bt_neg_away, :bt_pos_away, :bt_exc_away, :bt_half_away
        )
        ON CONFLICT(match_key) DO UPDATE SET
            filename = excluded.filename,
            phase = excluded.phase,
            round_number = excluded.round_number,
            season = excluded.season,
            competition = excluded.competition,
            team_a = excluded.team_a,
            team_b = excluded.team_b,
            n_azioni = excluded.n_azioni,
            preview = excluded.preview,
            scout_text = excluded.scout_text,
            created_at = excluded.created_at,

            so_home_attempts = excluded.so_home_attempts,
            so_home_wins = excluded.so_home_wins,
            so_away_attempts = excluded.so_away_attempts,
            so_away_wins = excluded.so_away_wins,
            sideout_home_pct = excluded.sideout_home_pct,
            sideout_away_pct = excluded.sideout_away_pct,

            bp_home_attempts = excluded.bp_home_attempts,
            bp_home_wins = excluded.bp_home_wins,
            bp_away_attempts = excluded.bp_away_attempts,
            bp_away_wins = excluded.bp_away_wins,
            break_home_pct = excluded.break_home_pct,
            break_away_pct = excluded.break_away_pct,

            so_spin_home_attempts = excluded.so_spin_home_attempts,
            so_spin_home_wins = excluded.so_spin_home_wins,
            so_spin_away_attempts = excluded.so_spin_away_attempts,
            so_spin_away_wins = excluded.so_spin_away_wins,

            so_float_home_attempts = excluded.so_float_home_attempts,
            so_float_home_wins = excluded.so_float_home_wins,
            so_float_away_attempts = excluded.so_float_away_attempts,
            so_float_away_wins = excluded.so_float_away_wins,

            so_dir_home_wins = excluded.so_dir_home_wins,
            so_dir_away_wins = excluded.so_dir_away_wins,

            so_play_home_attempts = excluded.so_play_home_attempts,
            so_play_home_wins = excluded.so_play_home_wins,
            so_play_away_attempts = excluded.so_play_away_attempts,
            so_play_away_wins = excluded.so_play_away_wins,

            so_good_home_attempts = excluded.so_good_home_attempts,
            so_good_home_wins = excluded.so_good_home_wins,
            so_good_away_attempts = excluded.so_good_away_attempts,
            so_good_away_wins = excluded.so_good_away_wins,

            so_exc_home_attempts = excluded.so_exc_home_attempts,
            so_exc_home_wins = excluded.so_exc_home_wins,
            so_exc_away_attempts = excluded.so_exc_away_attempts,
            so_exc_away_wins = excluded.so_exc_away_wins,

            so_neg_home_attempts = excluded.so_neg_home_attempts,
            so_neg_home_wins = excluded.so_neg_home_wins,
            so_neg_away_attempts = excluded.so_neg_away_attempts,
            so_neg_away_wins = excluded.so_neg_away_wins,

            bp_play_home_attempts = excluded.bp_play_home_attempts,
            bp_play_home_wins = excluded.bp_play_home_wins,
            bp_play_away_attempts = excluded.bp_play_away_attempts,
            bp_play_away_wins = excluded.bp_play_away_wins,

            bt_neg_home = excluded.bt_neg_home,
            bt_pos_home = excluded.bt_pos_home,
            bt_exc_home = excluded.bt_exc_home,
            bt_half_home = excluded.bt_half_home,

            bt_neg_away = excluded.bt_neg_away,
            bt_pos_away = excluded.bt_pos_away,
            bt_exc_away = excluded.bt_exc_away,
            bt_half_away = excluded.bt_half_away
        """

        with engine.begin() as conn:
            for uf in uploaded_files:
                filename = uf.name
                try:
                    dvw_text = uf.getvalue().decode("utf-8", errors="ignore")
                    phase, round_number = extract_round_code(filename)
                    parsed = parse_dvw_minimal(dvw_text)
                    scout_lines = extract_scout_lines(dvw_text)
                    counts = compute_counts_from_scout(scout_lines)

                    match_key = build_match_key(
                        parsed.get("team_a", ""),
                        parsed.get("team_b", ""),
                        parsed.get("competition", ""),
                        phase,
                        round_number,
                    )
                    preview = " | ".join([code6(x) for x in scout_lines[:3]])

                    params = {
                        "filename": filename,
                        "phase": phase,
                        "round_number": int(round_number),
                        "season": parsed.get("season"),
                        "competition": parsed.get("competition"),
                        "team_a": parsed.get("team_a"),
                        "team_b": parsed.get("team_b"),
                        "n_azioni": int(len(scout_lines)),
                        "preview": preview,
                        "scout_text": "\n".join(scout_lines),
                        "match_key": match_key,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        **counts,
                    }

                    conn.execute(text(sql_upsert), params)
                    saved += 1
                    details.append({"file": filename, "esito": "OK"})
                except Exception as e:
                    errors += 1
                    details.append({"file": filename, "esito": f"ERRORE: {e}"})

        st.success(f"Fatto. Importati/aggiornati: {saved} | Errori: {errors}")
        st.dataframe(pd.DataFrame(details), width="stretch", hide_index=True)
        st.rerun()


# =========================
# UI: SIDEOUT TEAM (già completa)
# =========================
def render_sideout_team():
    st.header("Indici Side Out - Squadre")

    voce = st.radio(
        "Seleziona indice",
        [
            "Side Out TOTALE",
            "Side Out SPIN",
            "Side Out FLOAT",
            "Side Out DIRETTO",
            "Side Out GIOCATO",
            "Side Out con RICE BUONA",
            "Side Out con RICE ESCALAMATIVA",
            "Side Out con RICE NEGATIVA",
        ],
        index=0,
    )

    with engine.begin() as conn:
        bounds = conn.execute(text("""
            SELECT MIN(round_number) AS min_r, MAX(round_number) AS max_r
            FROM matches
            WHERE round_number IS NOT NULL
        """)).mappings().first()

    min_r = int((bounds["min_r"] or 1))
    max_r = int((bounds["max_r"] or 1))

    c1, c2 = st.columns(2)
    with c1:
        from_round = st.number_input("Da giornata", min_value=min_r, max_value=max_r, value=min_r, step=1)
    with c2:
        to_round = st.number_input("A giornata", min_value=min_r, max_value=max_r, value=max_r, step=1)

    if from_round > to_round:
        st.error("Range non valido: 'Da giornata' deve essere <= 'A giornata'.")
        st.stop()

    def highlight_perugia(row):
        is_perugia = "perugia" in str(row["squadra"]).lower()
        style = "background-color: #fff3cd; font-weight: 800;" if is_perugia else ""
        return [style] * len(row)

    def show_table(df: pd.DataFrame, fmt: dict):
        if df.empty:
            st.info("Nessun dato nel range selezionato.")
            return
        styled = (
            df.style
              .apply(highlight_perugia, axis=1)
              .format(fmt)
              .set_table_styles([
                  {"selector": "th", "props": [("font-size", "24px"), ("text-align", "left"), ("padding", "10px 12px")]},
                  {"selector": "td", "props": [("font-size", "23px"), ("padding", "10px 12px")]},
              ])
        )
        st.dataframe(styled, width="stretch", hide_index=True)

    # --- TOTALE ---
    if voce == "Side Out TOTALE":
        with engine.begin() as conn:
            rows = conn.execute(
                text("""
                    SELECT
                        squadra,
                        SUM(n_ricezioni) AS n_ricezioni,
                        SUM(n_sideout)   AS n_sideout,
                        COALESCE(ROUND(100.0 * SUM(n_sideout) / NULLIF(SUM(n_ricezioni), 0), 1), 0.0) AS so_pct
                    FROM (
                        SELECT team_a AS squadra,
                               COALESCE(so_home_attempts, 0) AS n_ricezioni,
                               COALESCE(so_home_wins, 0)     AS n_sideout
                        FROM matches
                        WHERE round_number BETWEEN :from_round AND :to_round
                        UNION ALL
                        SELECT team_b AS squadra,
                               COALESCE(so_away_attempts, 0) AS n_ricezioni,
                               COALESCE(so_away_wins, 0)     AS n_sideout
                        FROM matches
                        WHERE round_number BETWEEN :from_round AND :to_round
                    )
                    GROUP BY squadra
                    ORDER BY so_pct DESC, n_ricezioni DESC
                """),
                {"from_round": int(from_round), "to_round": int(to_round)}
            ).mappings().all()

        df = pd.DataFrame(rows).rename(columns={
            "squadra": "squadra",
            "so_pct": "% S.O.",
            "n_ricezioni": "n° ricezioni",
            "n_sideout": "n° Side Out",
        })
        df.insert(0, "Rank", range(1, len(df) + 1))
        df = df[["Rank", "squadra", "% S.O.", "n° ricezioni", "n° Side Out"]].copy()
        show_table(df, {"Rank": "{:.0f}", "% S.O.": "{:.1f}", "n° ricezioni": "{:.0f}", "n° Side Out": "{:.0f}"})

    # --- SPIN ---
    elif voce == "Side Out SPIN":
        with engine.begin() as conn:
            rows = conn.execute(
                text("""
                    SELECT
                        squadra,
                        SUM(spin_att) AS spin_att,
                        SUM(spin_win) AS spin_win,
                        COALESCE(ROUND(100.0 * SUM(spin_win) / NULLIF(SUM(spin_att), 0), 1), 0.0) AS so_spin_pct,
                        COALESCE(ROUND(100.0 * SUM(spin_att) / NULLIF(SUM(tot_att), 0), 1), 0.0) AS spin_share_pct
                    FROM (
                        SELECT team_a AS squadra,
                               COALESCE(so_spin_home_attempts, 0) AS spin_att,
                               COALESCE(so_spin_home_wins, 0)     AS spin_win,
                               COALESCE(so_home_attempts, 0)      AS tot_att
                        FROM matches
                        WHERE round_number BETWEEN :from_round AND :to_round
                        UNION ALL
                        SELECT team_b AS squadra,
                               COALESCE(so_spin_away_attempts, 0) AS spin_att,
                               COALESCE(so_spin_away_wins, 0)     AS spin_win,
                               COALESCE(so_away_attempts, 0)      AS tot_att
                        FROM matches
                        WHERE round_number BETWEEN :from_round AND :to_round
                    )
                    GROUP BY squadra
                    ORDER BY so_spin_pct DESC, spin_att DESC
                """),
                {"from_round": int(from_round), "to_round": int(to_round)}
            ).mappings().all()

        df_spin = pd.DataFrame(rows).rename(columns={
            "squadra": "squadra",
            "so_spin_pct": "% S.O. SPIN",
            "spin_att": "n° ricezioni SPIN",
            "spin_win": "n° Side Out SPIN",
            "spin_share_pct": "% SPIN su TOT",
        })
        df_spin.insert(0, "Rank", range(1, len(df_spin) + 1))
        df_spin = df_spin[["Rank", "squadra", "% S.O. SPIN", "n° ricezioni SPIN", "n° Side Out SPIN", "% SPIN su TOT"]].copy()
        show_table(df_spin, {"Rank": "{:.0f}", "% S.O. SPIN": "{:.1f}", "n° ricezioni SPIN": "{:.0f}",
                             "n° Side Out SPIN": "{:.0f}", "% SPIN su TOT": "{:.1f}"})

    # --- FLOAT ---
    elif voce == "Side Out FLOAT":
        with engine.begin() as conn:
            rows = conn.execute(
                text("""
                    SELECT
                        squadra,
                        SUM(float_att) AS float_att,
                        SUM(float_win) AS float_win,
                        COALESCE(ROUND(100.0 * SUM(float_win) / NULLIF(SUM(float_att), 0), 1), 0.0) AS so_float_pct,
                        COALESCE(ROUND(100.0 * SUM(float_att) / NULLIF(SUM(tot_att), 0), 1), 0.0) AS float_share_pct
                    FROM (
                        SELECT team_a AS squadra,
                               COALESCE(so_float_home_attempts, 0) AS float_att,
                               COALESCE(so_float_home_wins, 0)     AS float_win,
                               COALESCE(so_home_attempts, 0)       AS tot_att
                        FROM matches
                        WHERE round_number BETWEEN :from_round AND :to_round
                        UNION ALL
                        SELECT team_b AS squadra,
                               COALESCE(so_float_away_attempts, 0) AS float_att,
                               COALESCE(so_float_away_wins, 0)     AS float_win,
                               COALESCE(so_away_attempts, 0)       AS tot_att
                        FROM matches
                        WHERE round_number BETWEEN :from_round AND :to_round
                    )
                    GROUP BY squadra
                    ORDER BY so_float_pct DESC, float_att DESC
                """),
                {"from_round": int(from_round), "to_round": int(to_round)}
            ).mappings().all()

        df_float = pd.DataFrame(rows).rename(columns={
            "squadra": "squadra",
            "so_float_pct": "% S.O. FLOAT",
            "float_att": "n° ricezioni FLOAT",
            "float_win": "n° Side Out FLOAT",
            "float_share_pct": "% FLOAT su TOT",
        })
        df_float.insert(0, "Rank", range(1, len(df_float) + 1))
        df_float = df_float[["Rank", "squadra", "% S.O. FLOAT", "n° ricezioni FLOAT", "n° Side Out FLOAT", "% FLOAT su TOT"]].copy()
        show_table(df_float, {"Rank": "{:.0f}", "% S.O. FLOAT": "{:.1f}", "n° ricezioni FLOAT": "{:.0f}",
                              "n° Side Out FLOAT": "{:.0f}", "% FLOAT su TOT": "{:.1f}"})

    # --- DIRETTO ---
    elif voce == "Side Out DIRETTO":
        with engine.begin() as conn:
            rows = conn.execute(
                text("""
                    SELECT
                        squadra,
                        SUM(tot_att) AS n_ricezioni,
                        SUM(dir_win) AS n_sideout_dir,
                        COALESCE(ROUND(100.0 * SUM(dir_win) / NULLIF(SUM(tot_att), 0), 1), 0.0) AS so_dir_pct
                    FROM (
                        SELECT team_a AS squadra,
                               COALESCE(so_home_attempts, 0) AS tot_att,
                               COALESCE(so_dir_home_wins, 0) AS dir_win
                        FROM matches
                        WHERE round_number BETWEEN :from_round AND :to_round
                        UNION ALL
                        SELECT team_b AS squadra,
                               COALESCE(so_away_attempts, 0) AS tot_att,
                               COALESCE(so_dir_away_wins, 0) AS dir_win
                        FROM matches
                        WHERE round_number BETWEEN :from_round AND :to_round
                    )
                    GROUP BY squadra
                    ORDER BY so_dir_pct DESC, n_ricezioni DESC
                """),
                {"from_round": int(from_round), "to_round": int(to_round)}
            ).mappings().all()

        df_dir = pd.DataFrame(rows).rename(columns={
            "squadra": "squadra",
            "so_dir_pct": "% S.O. DIR",
            "n_ricezioni": "n° ricezioni",
            "n_sideout_dir": "n° Side Out DIR",
        })
        df_dir.insert(0, "Rank", range(1, len(df_dir) + 1))
        df_dir = df_dir[["Rank", "squadra", "% S.O. DIR", "n° ricezioni", "n° Side Out DIR"]].copy()
        show_table(df_dir, {"Rank": "{:.0f}", "% S.O. DIR": "{:.1f}", "n° ricezioni": "{:.0f}", "n° Side Out DIR": "{:.0f}"})

    # --- GIOCATO ---
    elif voce == "Side Out GIOCATO":
        with engine.begin() as conn:
            rows = conn.execute(
                text("""
                    SELECT
                        squadra,
                        SUM(att) AS n_ricezioni_giocato,
                        SUM(win) AS n_sideout_giocato,
                        COALESCE(ROUND(100.0 * SUM(win) / NULLIF(SUM(att), 0), 1), 0.0) AS so_giocato_pct
                    FROM (
                        SELECT team_a AS squadra,
                               COALESCE(so_play_home_attempts, 0) AS att,
                               COALESCE(so_play_home_wins, 0)     AS win
                        FROM matches
                        WHERE round_number BETWEEN :from_round AND :to_round
                        UNION ALL
                        SELECT team_b AS squadra,
                               COALESCE(so_play_away_attempts, 0) AS att,
                               COALESCE(so_play_away_wins, 0)     AS win
                        FROM matches
                        WHERE round_number BETWEEN :from_round AND :to_round
                    )
                    GROUP BY squadra
                    ORDER BY so_giocato_pct DESC, n_ricezioni_giocato DESC
                """),
                {"from_round": int(from_round), "to_round": int(to_round)}
            ).mappings().all()

        df_g = pd.DataFrame(rows).rename(columns={
            "squadra": "squadra",
            "so_giocato_pct": "% S.O. GIOCATO",
            "n_ricezioni_giocato": "n° ricezioni (giocabili)",
            "n_sideout_giocato": "n° Side Out",
        })
        df_g.insert(0, "Rank", range(1, len(df_g) + 1))
        df_g = df_g[["Rank", "squadra", "% S.O. GIOCATO", "n° ricezioni (giocabili)", "n° Side Out"]].copy()
        show_table(df_g, {"Rank": "{:.0f}", "% S.O. GIOCATO": "{:.1f}", "n° ricezioni (giocabili)": "{:.0f}", "n° Side Out": "{:.0f}"})

    # --- BUONA ---
    elif voce == "Side Out con RICE BUONA":
        with engine.begin() as conn:
            rows = conn.execute(
                text("""
                    SELECT
                        squadra,
                        SUM(att) AS n_ricezioni_buone,
                        SUM(win) AS n_sideout_buone,
                        COALESCE(ROUND(100.0 * SUM(win) / NULLIF(SUM(att), 0), 1), 0.0) AS so_buona_pct
                    FROM (
                        SELECT team_a AS squadra,
                               COALESCE(so_good_home_attempts, 0) AS att,
                               COALESCE(so_good_home_wins, 0)     AS win
                        FROM matches
                        WHERE round_number BETWEEN :from_round AND :to_round
                        UNION ALL
                        SELECT team_b AS squadra,
                               COALESCE(so_good_away_attempts, 0) AS att,
                               COALESCE(so_good_away_wins, 0)     AS win
                        FROM matches
                        WHERE round_number BETWEEN :from_round AND :to_round
                    )
                    GROUP BY squadra
                    ORDER BY so_buona_pct DESC, n_ricezioni_buone DESC
                """),
                {"from_round": int(from_round), "to_round": int(to_round)}
            ).mappings().all()

        df_b = pd.DataFrame(rows).rename(columns={
            "squadra": "squadra",
            "so_buona_pct": "% S.O. RICE BUONA",
            "n_ricezioni_buone": "n° ricezioni (#,+)",
            "n_sideout_buone": "n° Side Out",
        })
        df_b.insert(0, "Rank", range(1, len(df_b) + 1))
        df_b = df_b[["Rank", "squadra", "% S.O. RICE BUONA", "n° ricezioni (#,+)", "n° Side Out"]].copy()
        show_table(df_b, {"Rank": "{:.0f}", "% S.O. RICE BUONA": "{:.1f}", "n° ricezioni (#,+)": "{:.0f}", "n° Side Out": "{:.0f}"})

    # --- ESCLAMATIVA ---
    elif voce == "Side Out con RICE ESCALAMATIVA":
        with engine.begin() as conn:
            rows = conn.execute(
                text("""
                    SELECT
                        squadra,
                        SUM(att) AS n_ricezioni_exc,
                        SUM(win) AS n_sideout_exc,
                        COALESCE(ROUND(100.0 * SUM(win) / NULLIF(SUM(att), 0), 1), 0.0) AS so_exc_pct
                    FROM (
                        SELECT team_a AS squadra,
                               COALESCE(so_exc_home_attempts, 0) AS att,
                               COALESCE(so_exc_home_wins, 0)     AS win
                        FROM matches
                        WHERE round_number BETWEEN :from_round AND :to_round
                        UNION ALL
                        SELECT team_b AS squadra,
                               COALESCE(so_exc_away_attempts, 0) AS att,
                               COALESCE(so_exc_away_wins, 0)     AS win
                        FROM matches
                        WHERE round_number BETWEEN :from_round AND :to_round
                    )
                    GROUP BY squadra
                    ORDER BY so_exc_pct DESC, n_ricezioni_exc DESC
                """),
                {"from_round": int(from_round), "to_round": int(to_round)}
            ).mappings().all()

        df_e = pd.DataFrame(rows).rename(columns={
            "squadra": "squadra",
            "so_exc_pct": "% S.O. RICE !",
            "n_ricezioni_exc": "n° ricezioni (!)",
            "n_sideout_exc": "n° Side Out",
        })
        df_e.insert(0, "Rank", range(1, len(df_e) + 1))
        df_e = df_e[["Rank", "squadra", "% S.O. RICE !", "n° ricezioni (!)", "n° Side Out"]].copy()
        show_table(df_e, {"Rank": "{:.0f}", "% S.O. RICE !": "{:.1f}", "n° ricezioni (!)": "{:.0f}", "n° Side Out": "{:.0f}"})

    # --- NEGATIVA ---
    elif voce == "Side Out con RICE NEGATIVA":
        with engine.begin() as conn:
            rows = conn.execute(
                text("""
                    SELECT
                        squadra,
                        SUM(att) AS n_ricezioni_neg,
                        SUM(win) AS n_sideout_neg,
                        COALESCE(ROUND(100.0 * SUM(win) / NULLIF(SUM(att), 0), 1), 0.0) AS so_neg_pct
                    FROM (
                        SELECT team_a AS squadra,
                               COALESCE(so_neg_home_attempts, 0) AS att,
                               COALESCE(so_neg_home_wins, 0)     AS win
                        FROM matches
                        WHERE round_number BETWEEN :from_round AND :to_round
                        UNION ALL
                        SELECT team_b AS squadra,
                               COALESCE(so_neg_away_attempts, 0) AS att,
                               COALESCE(so_neg_away_wins, 0)     AS win
                        FROM matches
                        WHERE round_number BETWEEN :from_round AND :to_round
                    )
                    GROUP BY squadra
                    ORDER BY so_neg_pct DESC, n_ricezioni_neg DESC
                """),
                {"from_round": int(from_round), "to_round": int(to_round)}
            ).mappings().all()

        df_n = pd.DataFrame(rows).rename(columns={
            "squadra": "squadra",
            "so_neg_pct": "% S.O. RICE -",
            "n_ricezioni_neg": "n° ricezioni (-)",
            "n_sideout_neg": "n° Side Out",
        })
        df_n.insert(0, "Rank", range(1, len(df_n) + 1))
        df_n = df_n[["Rank", "squadra", "% S.O. RICE -", "n° ricezioni (-)", "n° Side Out"]].copy()
        show_table(df_n, {"Rank": "{:.0f}", "% S.O. RICE -": "{:.1f}", "n° ricezioni (-)": "{:.0f}", "n° Side Out": "{:.0f}"})


# =========================
# UI: BREAK TEAM
# =========================
def render_break_team():
    st.header("Indici Fase Break – Squadre")

    voce = st.radio(
        "Seleziona indice Break",
        [
            "BREAK TOTALE",
            "BREAK GIOCATO",
            "BREAK con BT. NEGATIVA",
            "BREAK con BT. ESCLAMATIVA",
            "BREAK con BT. POSITIVA",
            "BREAK con BT. 1/2 PUNTO",
            "BT punto/errore/ratio",            "Confronto TEAM",
            "GRAFICI",
        ],
        index=0,
    )

    # ===== FILTRO RANGE GIORNATE =====
    with engine.begin() as conn:
        bounds = conn.execute(text("""
            SELECT MIN(round_number) AS min_r, MAX(round_number) AS max_r
            FROM matches
            WHERE round_number IS NOT NULL
        """)).mappings().first()

    min_r = int(bounds["min_r"] or 1)
    max_r = int(bounds["max_r"] or 1)

    c1, c2 = st.columns(2)
    with c1:
        from_round = st.number_input("Da giornata", min_value=min_r, max_value=max_r, value=min_r, step=1, key="bp_from")
    with c2:
        to_round = st.number_input("A giornata", min_value=min_r, max_value=max_r, value=max_r, step=1, key="bp_to")

    if from_round > to_round:
        st.error("Range non valido: 'Da giornata' deve essere <= 'A giornata'.")
        st.stop()

    def fix_team(name: str) -> str:
        n = " ".join((name or "").split())
        low = n.lower()
        # mapping esplicito richiesto
        if low.startswith("gas sales bluenergy p"):
            return "Gas Sales Bluenergy Piacenza"
        if low == "gas sales bluenergy piacenza":
            return "Gas Sales Bluenergy Piacenza"
        if low == "grottazzolina":
            return "Yuasa Battery Grottazzolina"
        # normalizza eventuali varianti di maiuscole
        if "yuasa" in low and "grottazzolina" in low:
            return "Yuasa Battery Grottazzolina"
        return n

    def highlight_perugia(row):
        is_perugia = "perugia" in str(row["squadra"]).lower()
        style = "background-color: #fff3cd; font-weight: 800;" if is_perugia else ""
        return [style] * len(row)

    def show_table(df: pd.DataFrame, fmt: dict):
        if df.empty:
            st.info("Nessun dato nel range selezionato.")
            return
        # applica mapping richiesto
        if "squadra" in df.columns:
            df = df.copy()
            df["squadra"] = df["squadra"].apply(fix_team)

        styled = (
            df.style
              .apply(highlight_perugia, axis=1)
              .format(fmt)
              .set_table_styles([
                  {"selector": "th", "props": [("font-size", "24px"), ("text-align", "left"), ("padding", "10px 12px")]},
                  {"selector": "td", "props": [("font-size", "23px"), ("padding", "10px 12px")]},
              ])
        )
        st.dataframe(styled, width="stretch", hide_index=True)

    # ===== helper: calcola BT giocato/segni direttamente da scout_text (NON usa colonne bp_play_*/bt_* nel DB) =====
    def compute_break_bt_agg() -> pd.DataFrame:
        with engine.begin() as conn:
            rows = conn.execute(
                text("""
                    SELECT
                        team_a, team_b,
                        scout_text,
                        COALESCE(bp_home_attempts, 0) AS bp_home_attempts,
                        COALESCE(bp_away_attempts, 0) AS bp_away_attempts,
                        COALESCE(bp_home_wins, 0)     AS bp_home_wins,
                        COALESCE(bp_away_wins, 0)     AS bp_away_wins
                    FROM matches
                    WHERE round_number BETWEEN :from_round AND :to_round
                """),
                {"from_round": int(from_round), "to_round": int(to_round)}
            ).mappings().all()

        if not rows:
            return pd.DataFrame()

        agg = {}

        def ensure(team: str):
            team = fix_team(team)
            if team not in agg:
                agg[team] = {
                    "squadra": team,
                    "bt_att": 0,      # battute con valutazione (- + !)
                    "bt_bp": 0,       # break point su quelle battute
                    "bt_neg": 0,
                    "bt_pos": 0,
                    "bt_exc": 0,
                    "serves_tot": 0,  # battute totali (da DB)
                    "bp_tot": 0,      # break point totali (da DB)
                }
            return team

        def iter_rallies_from_scout_text(scout_text: str):
            if not scout_text:
                return []
            lines = [ln.strip() for ln in str(scout_text).splitlines() if ln and ln.strip()]
            scout_lines = [ln for ln in lines if ln and ln[0] in ("*", "a")]

            rallies = []
            current = []
            for raw in scout_lines:
                c = code6(raw)
                if not c:
                    continue
                if is_serve(c):
                    if current:
                        rallies.append(current)
                    current = [c]
                    continue
                if not current:
                    continue
                current.append(c)
                if is_home_point(c) or is_away_point(c):
                    rallies.append(current)
                    current = []
            return rallies

        for r in rows:
            home_team = ensure(r.get("team_a") or "")
            away_team = ensure(r.get("team_b") or "")

            # totali dal DB
            agg[home_team]["serves_tot"] += int(r.get("bp_home_attempts") or 0)
            agg[away_team]["serves_tot"] += int(r.get("bp_away_attempts") or 0)
            agg[home_team]["bp_tot"] += int(r.get("bp_home_wins") or 0)
            agg[away_team]["bp_tot"] += int(r.get("bp_away_wins") or 0)

            rallies = iter_rallies_from_scout_text(r.get("scout_text") or "")
            for rally in rallies:
                first = rally[0]
                if len(first) < 6:
                    continue
                sign = first[5]  # '-' '+' '!' ecc.
                if sign not in ("-", "+", "!"):
                    continue  # non "giocato" secondo tua logica BT
                home_served = first.startswith("*")
                away_served = first.startswith("a")

                home_point = any(is_home_point(x) for x in rally)
                away_point = any(is_away_point(x) for x in rally)

                if home_served:
                    agg[home_team]["bt_att"] += 1
                    if home_point:
                        agg[home_team]["bt_bp"] += 1
                    if sign == "-":
                        agg[home_team]["bt_neg"] += 1
                    elif sign == "+":
                        agg[home_team]["bt_pos"] += 1
                    elif sign == "!":
                        agg[home_team]["bt_exc"] += 1

                if away_served:
                    agg[away_team]["bt_att"] += 1
                    if away_point:
                        agg[away_team]["bt_bp"] += 1
                    if sign == "-":
                        agg[away_team]["bt_neg"] += 1
                    elif sign == "+":
                        agg[away_team]["bt_pos"] += 1
                    elif sign == "!":
                        agg[away_team]["bt_exc"] += 1

        return pd.DataFrame(list(agg.values()))

    # ===== BREAK TOTALE (come prima) =====
    if voce == "BREAK TOTALE":
        with engine.begin() as conn:
            rows = conn.execute(text("""
                SELECT
                    squadra,
                    SUM(att) AS n_battute,
                    SUM(win) AS n_bpoint,
                    COALESCE(ROUND(100.0 * SUM(win) / NULLIF(SUM(att), 0), 1), 0.0) AS bp_pct
                FROM (
                    SELECT team_a AS squadra,
                           COALESCE(bp_home_attempts, 0) AS att,
                           COALESCE(bp_home_wins, 0)     AS win
                    FROM matches
                    WHERE round_number BETWEEN :from_round AND :to_round
                    UNION ALL
                    SELECT team_b AS squadra,
                           COALESCE(bp_away_attempts, 0) AS att,
                           COALESCE(bp_away_wins, 0)     AS win
                    FROM matches
                    WHERE round_number BETWEEN :from_round AND :to_round
                )
                GROUP BY squadra
                ORDER BY bp_pct DESC, n_battute DESC
            """), {"from_round": int(from_round), "to_round": int(to_round)}).mappings().all()

        df = pd.DataFrame(rows)
        if df.empty:
            st.info("Nessun dato nel range selezionato.")
            return

        df["squadra"] = df["squadra"].apply(fix_team)
        df = df.groupby("squadra", as_index=False).sum(numeric_only=True)
        df["% B.Point"] = (100.0 * df["n_bpoint"] / df["n_battute"].replace({0: pd.NA})).fillna(0.0)

        df = df.rename(columns={
            "n_battute": "n° Battute",
            "n_bpoint": "n° B.Point",
        })

        df = df.sort_values(by=["% B.Point", "n° Battute"], ascending=[False, False]).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))
        df = df[["Rank", "squadra", "% B.Point", "n° Battute", "n° B.Point"]].copy()

        show_table(df, {"Rank": "{:.0f}", "% B.Point": "{:.1f}", "n° Battute": "{:.0f}", "n° B.Point": "{:.0f}"})

    # ===== BREAK GIOCATO (BT = '-' '+' '!') =====
    elif voce == "BREAK GIOCATO":
        base = compute_break_bt_agg()
        if base.empty:
            st.info("Nessun dato nel range selezionato.")
            return

        df = base.copy()
        df["% B.Point (Giocato)"] = (100.0 * df["bt_bp"] / df["bt_att"].replace({0: pd.NA})).fillna(0.0)

        df = df.rename(columns={
            "bt_att": "n° Battute (BT)",
            "bt_bp": "n° B.Point",
        })

        df = df.sort_values(by=["% B.Point (Giocato)", "n° Battute (BT)"], ascending=[False, False]).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))

        df = df[["Rank", "squadra", "% B.Point (Giocato)", "n° Battute (BT)", "n° B.Point"]].copy()
        show_table(df, {"Rank": "{:.0f}", "% B.Point (Giocato)": "{:.1f}", "n° Battute (BT)": "{:.0f}", "n° B.Point": "{:.0f}"})

    # ===== BT NEGATIVA / POSITIVA / ESCLAMATIVA =====
    elif voce == "BREAK con BT. NEGATIVA":
        # ==========================================================
        # TABELLA RIDEFINITA (richiesta):
        # A Ranking (per % B.Point con Bt-)
        # B Team (12)
        # C % B.Point con Bt-  = BP_su_battuta_negativa / battute_negative
        #    BP: fine azione *p (casa) / ap (ospite) A FAVORE DEL BATTITORE
        # D % Bt-/Bt Tot = battute_negative / battute_totali
        # ==========================================================
        with engine.begin() as conn:
            rows = conn.execute(
                text("""
                    SELECT team_a, team_b, scout_text
                    FROM matches
                    WHERE round_number BETWEEN :from_round AND :to_round
                """),
                {"from_round": int(from_round), "to_round": int(to_round)}
            ).mappings().all()

        if not rows:
            st.info("Nessun dato nel range selezionato.")
            return

        def fix_team(name: str) -> str:
            n = " ".join((name or "").split())
            if n.lower().startswith("gas sales bluenergy p"):
                return "Gas Sales Bluenergy Piacenza"
            if "grottazzolina" in n.lower():
                return "Yuasa Battery Grottazzolina"
            return n

        agg = {}
        def ensure(team: str):
            if team not in agg:
                agg[team] = {"Team": team, "neg_serves": 0, "neg_bp": 0, "tot_serves": 0}

        for r in rows:
            ta = fix_team(r.get("team_a") or "")
            tb = fix_team(r.get("team_b") or "")
            ensure(ta)
            ensure(tb)

            scout_text = r.get("scout_text") or ""
            lines = [ln.strip() for ln in str(scout_text).splitlines() if ln and ln.strip()]
            scout_lines = [ln for ln in lines if ln[0] in ("*", "a")]

            rallies = []
            current = []

            for raw in scout_lines:
                c = code6(raw)
                if not c:
                    continue

                if is_serve(c):
                    if current:
                        rallies.append(current)
                    current = [c]
                    continue

                if not current:
                    continue

                current.append(c)

                if is_home_point(c) or is_away_point(c):
                    rallies.append(current)
                    current = []

            for rally in rallies:
                first = rally[0]
                home_served = first.startswith("*")
                away_served = first.startswith("a")

                home_point = any(is_home_point(x) for x in rally)
                away_point = any(is_away_point(x) for x in rally)

                if home_served:
                    agg[ta]["tot_serves"] += 1
                if away_served:
                    agg[tb]["tot_serves"] += 1

                is_neg = (len(first) >= 6 and first[5] == "-")
                if not is_neg:
                    continue

                if home_served:
                    agg[ta]["neg_serves"] += 1
                    if home_point:
                        agg[ta]["neg_bp"] += 1

                if away_served:
                    agg[tb]["neg_serves"] += 1
                    if away_point:
                        agg[tb]["neg_bp"] += 1

        df = pd.DataFrame(list(agg.values()))
        if df.empty:
            st.info("Nessun dato nel range selezionato.")
            return

        df["% B.Point con Bt-"] = (100.0 * df["neg_bp"] / df["neg_serves"].replace({0: pd.NA})).fillna(0.0)
        df["% Bt-/Bt Tot"] = (100.0 * df["neg_serves"] / df["tot_serves"].replace({0: pd.NA})).fillna(0.0)

        df = df.sort_values(by=["% B.Point con Bt-", "% Bt-/Bt Tot", "Team"], ascending=[False, False, True]).reset_index(drop=True)
        df.insert(0, "Ranking", range(1, len(df) + 1))

        out = df[["Ranking", "Team", "% B.Point con Bt-", "% Bt-/Bt Tot"]].copy()

        def highlight_perugia_row(row):
            is_perugia = "perugia" in str(row["Team"]).lower()
            style = "background-color: #fff3cd; font-weight: 800;" if is_perugia else ""
            return [style] * len(row)

        styled = (
            out.style
              .apply(highlight_perugia_row, axis=1)
              .format({"Ranking": "{:.0f}", "% B.Point con Bt-": "{:.1f}", "% Bt-/Bt Tot": "{:.1f}"})
              .set_table_styles([
                  {"selector": "th", "props": [("font-size", "24px"), ("text-align", "left"), ("padding", "10px 12px")]},
                  {"selector": "td", "props": [("font-size", "23px"), ("padding", "10px 12px")]},
              ])
        )
        st.dataframe(styled, width="stretch", hide_index=True)
    elif voce == "BREAK con BT. POSITIVA":
        # ==========================================================
        # TABELLA POSITIVA (stesso modello della NEGATIVA):
        # A Ranking (per % B.Point con Bt+)
        # B Team (12)
        # C % B.Point con Bt+  = BP_su_battuta_positiva / battute_positive
        #    BP: fine azione *p (casa) / ap (ospite) A FAVORE DEL BATTITORE
        # D % Bt+/Bt Tot = battute_positive / battute_totali
        # ==========================================================
        with engine.begin() as conn:
            rows = conn.execute(
                text("""
                    SELECT team_a, team_b, scout_text
                    FROM matches
                    WHERE round_number BETWEEN :from_round AND :to_round
                """),
                {"from_round": int(from_round), "to_round": int(to_round)}
            ).mappings().all()

        if not rows:
            st.info("Nessun dato nel range selezionato.")
            return

        def fix_team(name: str) -> str:
            n = " ".join((name or "").split())
            if n.lower().startswith("gas sales bluenergy p"):
                return "Gas Sales Bluenergy Piacenza"
            if "grottazzolina" in n.lower():
                return "Yuasa Battery Grottazzolina"
            return n

        agg = {}
        def ensure(team: str):
            if team not in agg:
                agg[team] = {"Team": team, "pos_serves": 0, "pos_bp": 0, "tot_serves": 0}

        for r in rows:
            ta = fix_team(r.get("team_a") or "")
            tb = fix_team(r.get("team_b") or "")
            ensure(ta)
            ensure(tb)

            scout_text = r.get("scout_text") or ""
            lines = [ln.strip() for ln in str(scout_text).splitlines() if ln and ln.strip()]
            scout_lines = [ln for ln in lines if ln[0] in ("*", "a")]

            rallies = []
            current = []

            for raw in scout_lines:
                c = code6(raw)
                if not c:
                    continue

                if is_serve(c):
                    if current:
                        rallies.append(current)
                    current = [c]
                    continue

                if not current:
                    continue

                current.append(c)

                if is_home_point(c) or is_away_point(c):
                    rallies.append(current)
                    current = []

            for rally in rallies:
                first = rally[0]
                home_served = first.startswith("*")
                away_served = first.startswith("a")

                home_point = any(is_home_point(x) for x in rally)
                away_point = any(is_away_point(x) for x in rally)

                # total serves
                if home_served:
                    agg[ta]["tot_serves"] += 1
                if away_served:
                    agg[tb]["tot_serves"] += 1

                # positiva se 6° char del servizio è '+'
                is_pos = (len(first) >= 6 and first[5] == "+")
                if not is_pos:
                    continue

                if home_served:
                    agg[ta]["pos_serves"] += 1
                    # BP su battuta positiva: punto a favore del battitore (casa) -> *p
                    if home_point:
                        agg[ta]["pos_bp"] += 1

                if away_served:
                    agg[tb]["pos_serves"] += 1
                    # BP su battuta positiva: punto a favore del battitore (ospite) -> ap
                    if away_point:
                        agg[tb]["pos_bp"] += 1

        df = pd.DataFrame(list(agg.values()))
        if df.empty:
            st.info("Nessun dato nel range selezionato.")
            return

        df["% B.Point con Bt+"] = (100.0 * df["pos_bp"] / df["pos_serves"].replace({0: pd.NA})).fillna(0.0)
        df["% Bt+/Bt Tot"] = (100.0 * df["pos_serves"] / df["tot_serves"].replace({0: pd.NA})).fillna(0.0)

        df = df.sort_values(by=["% B.Point con Bt+", "% Bt+/Bt Tot", "Team"], ascending=[False, False, True]).reset_index(drop=True)
        df.insert(0, "Ranking", range(1, len(df) + 1))

        out = df[["Ranking", "Team", "% B.Point con Bt+", "% Bt+/Bt Tot"]].copy()

        def highlight_perugia_row(row):
            is_perugia = "perugia" in str(row["Team"]).lower()
            style = "background-color: #fff3cd; font-weight: 800;" if is_perugia else ""
            return [style] * len(row)

        styled = (
            out.style
              .apply(highlight_perugia_row, axis=1)
              .format({"Ranking": "{:.0f}", "% B.Point con Bt+": "{:.1f}", "% Bt+/Bt Tot": "{:.1f}"})
              .set_table_styles([
                  {"selector": "th", "props": [("font-size", "24px"), ("text-align", "left"), ("padding", "10px 12px")]},
                  {"selector": "td", "props": [("font-size", "23px"), ("padding", "10px 12px")]},
              ])
        )
        st.dataframe(styled, width="stretch", hide_index=True)
    elif voce == "BREAK con BT. ESCLAMATIVA":
        # ==========================================================
        # TABELLA ESCLAMATIVA (solo '!'):
        # A Ranking (per % B.Point con Bt!)
        # B Team (12)
        # C % B.Point con Bt! = BP_su_battuta_esclamativa / battute_esclamative
        #    BP: fine azione *p (casa) / ap (ospite) a favore del battitore
        # D % Bt!/Bt Tot = battute_esclamative / battute_totali
        # ==========================================================
        with engine.begin() as conn:
            rows = conn.execute(
                text("""
                    SELECT team_a, team_b, scout_text
                    FROM matches
                    WHERE round_number BETWEEN :from_round AND :to_round
                """),
                {"from_round": int(from_round), "to_round": int(to_round)}
            ).mappings().all()

        if not rows:
            st.info("Nessun dato nel range selezionato.")
            return

        def fix_team(name: str) -> str:
            n = " ".join((name or "").split())
            if n.lower().startswith("gas sales bluenergy p"):
                return "Gas Sales Bluenergy Piacenza"
            if "grottazzolina" in n.lower():
                return "Yuasa Battery Grottazzolina"
            return n

        agg = {}
        def ensure(team: str):
            if team not in agg:
                agg[team] = {"Team": team, "exc_serves": 0, "exc_bp": 0, "tot_serves": 0}

        for r in rows:
            ta = fix_team(r.get("team_a") or "")
            tb = fix_team(r.get("team_b") or "")
            ensure(ta)
            ensure(tb)

            scout_text = r.get("scout_text") or ""
            lines = [ln.strip() for ln in str(scout_text).splitlines() if ln and ln.strip()]
            scout_lines = [ln for ln in lines if ln[0] in ("*", "a")]

            rallies = []
            current = []

            for raw in scout_lines:
                c = code6(raw)
                if not c:
                    continue

                if is_serve(c):
                    if current:
                        rallies.append(current)
                    current = [c]
                    continue

                if not current:
                    continue

                current.append(c)

                if is_home_point(c) or is_away_point(c):
                    rallies.append(current)
                    current = []

            for rally in rallies:
                first = rally[0]
                home_served = first.startswith("*")
                away_served = first.startswith("a")

                home_point = any(is_home_point(x) for x in rally)
                away_point = any(is_away_point(x) for x in rally)

                if home_served:
                    agg[ta]["tot_serves"] += 1
                if away_served:
                    agg[tb]["tot_serves"] += 1

                # esclamativa se 6° char del servizio è '!'
                is_exc = (len(first) >= 6 and first[5] == "!")
                if not is_exc:
                    continue

                if home_served:
                    agg[ta]["exc_serves"] += 1
                    if home_point:
                        agg[ta]["exc_bp"] += 1

                if away_served:
                    agg[tb]["exc_serves"] += 1
                    if away_point:
                        agg[tb]["exc_bp"] += 1

        df = pd.DataFrame(list(agg.values()))
        if df.empty:
            st.info("Nessun dato nel range selezionato.")
            return

        df["% B.Point con Bt!"] = (100.0 * df["exc_bp"] / df["exc_serves"].replace({0: pd.NA})).fillna(0.0)
        df["% Bt!/Bt Tot"] = (100.0 * df["exc_serves"] / df["tot_serves"].replace({0: pd.NA})).fillna(0.0)

        df = df.sort_values(by=["% B.Point con Bt!", "% Bt!/Bt Tot", "Team"], ascending=[False, False, True]).reset_index(drop=True)
        df.insert(0, "Ranking", range(1, len(df) + 1))

        out = df[["Ranking", "Team", "% B.Point con Bt!", "% Bt!/Bt Tot"]].copy()

        def highlight_perugia_row(row):
            is_perugia = "perugia" in str(row["Team"]).lower()
            style = "background-color: #fff3cd; font-weight: 800;" if is_perugia else ""
            return [style] * len(row)

        styled = (
            out.style
              .apply(highlight_perugia_row, axis=1)
              .format({"Ranking": "{:.0f}", "% B.Point con Bt!": "{:.1f}", "% Bt!/Bt Tot": "{:.1f}"})
              .set_table_styles([
                  {"selector": "th", "props": [("font-size", "24px"), ("text-align", "left"), ("padding", "10px 12px")]},
                  {"selector": "td", "props": [("font-size", "23px"), ("padding", "10px 12px")]},
              ])
        )
        st.dataframe(styled, width="stretch", hide_index=True)


    elif voce == "BREAK con BT. 1/2 PUNTO":
        # ==========================================================
        # TABELLA 1/2 PUNTO (stesso modello di NEG / POS / !):
        # A Ranking (per % B.Point con Bt½)
        # B Team (12)
        # C % B.Point con Bt½ = BP_su_battuta_½ / battute_½
        #    BP: fine azione *p (casa) / ap (ospite) a favore del battitore
        # D % Bt½/Bt Tot = battute_½ / battute_totali
        # ==========================================================
        with engine.begin() as conn:
            rows = conn.execute(
                text("""
                    SELECT team_a, team_b, scout_text
                    FROM matches
                    WHERE round_number BETWEEN :from_round AND :to_round
                """),
                {"from_round": int(from_round), "to_round": int(to_round)}
            ).mappings().all()

        if not rows:
            st.info("Nessun dato nel range selezionato.")
            return

        def fix_team(name: str) -> str:
            n = " ".join((name or "").split())
            if n.lower().startswith("gas sales bluenergy p"):
                return "Gas Sales Bluenergy Piacenza"
            if "grottazzolina" in n.lower():
                return "Yuasa Battery Grottazzolina"
            return n

        agg = {}
        def ensure(team: str):
            if team not in agg:
                agg[team] = {"Team": team, "half_serves": 0, "half_bp": 0, "tot_serves": 0}

        for r in rows:
            ta = fix_team(r.get("team_a") or "")
            tb = fix_team(r.get("team_b") or "")
            ensure(ta)
            ensure(tb)

            scout_text = r.get("scout_text") or ""
            lines = [ln.strip() for ln in str(scout_text).splitlines() if ln and ln.strip()]
            scout_lines = [ln for ln in lines if ln[0] in ("*", "a")]

            rallies = []
            current = []

            for raw in scout_lines:
                c = code6(raw)
                if not c:
                    continue

                if is_serve(c):
                    if current:
                        rallies.append(current)
                    current = [c]
                    continue

                if not current:
                    continue

                current.append(c)

                if is_home_point(c) or is_away_point(c):
                    rallies.append(current)
                    current = []

            for rally in rallies:
                first = rally[0]
                home_served = first.startswith("*")
                away_served = first.startswith("a")

                home_point = any(is_home_point(x) for x in rally)
                away_point = any(is_away_point(x) for x in rally)

                if home_served:
                    agg[ta]["tot_serves"] += 1
                if away_served:
                    agg[tb]["tot_serves"] += 1

                # 1/2 punto: nel tuo scout è '/' nel 6° carattere (es: *06SQ/).
                # Nessun fallback: usiamo '/' come da codifica.
                is_half = (len(first) >= 6 and first[5] == "/")
                if not is_half:
                    continue

                if home_served:
                    agg[ta]["half_serves"] += 1
                    if home_point:
                        agg[ta]["half_bp"] += 1

                if away_served:
                    agg[tb]["half_serves"] += 1
                    if away_point:
                        agg[tb]["half_bp"] += 1

        df = pd.DataFrame(list(agg.values()))
        if df.empty:
            st.info("Nessun dato nel range selezionato.")
            return

        df["% B.Point con Bt½"] = (100.0 * df["half_bp"] / df["half_serves"].replace({0: pd.NA})).fillna(0.0)
        df["% Bt½/Bt Tot"] = (100.0 * df["half_serves"] / df["tot_serves"].replace({0: pd.NA})).fillna(0.0)

        df = df.sort_values(by=["% B.Point con Bt½", "% Bt½/Bt Tot", "Team"], ascending=[False, False, True]).reset_index(drop=True)
        df.insert(0, "Ranking", range(1, len(df) + 1))

        out = df[["Ranking", "Team", "% B.Point con Bt½", "% Bt½/Bt Tot"]].copy()

        def highlight_perugia_row(row):
            is_perugia = "perugia" in str(row["Team"]).lower()
            style = "background-color: #fff3cd; font-weight: 800;" if is_perugia else ""
            return [style] * len(row)

        styled = (
            out.style
              .apply(highlight_perugia_row, axis=1)
              .format({"Ranking": "{:.0f}", "% B.Point con Bt½": "{:.1f}", "% Bt½/Bt Tot": "{:.1f}"})
              .set_table_styles([
                  {"selector": "th", "props": [("font-size", "24px"), ("text-align", "left"), ("padding", "10px 12px")]},
                  {"selector": "td", "props": [("font-size", "23px"), ("padding", "10px 12px")]},
              ])
        )
        st.dataframe(styled, width="stretch", hide_index=True)


    elif voce == "BT punto/errore/ratio":
        # ==========================================================
        # BT punto/errore/ratio (richiesta):
        # COL A: Teams
        # COL B: % Bt Punto (#) su Tot Battute
        # COL C: % Bt Errore (=) su Tot Battute
        # COL D: Ratio Errore/Punto = Tot '=' / Tot '#'
        #
        # Definizioni (sul servizio code6):
        # - Battuta = SQ o SM (is_serve)
        # - Punto = 6° carattere '#': *06SQ#
        # - Errore = 6° carattere '=': *06SQ=
        # ==========================================================
        with engine.begin() as conn:
            rows = conn.execute(
                text("""
                    SELECT team_a, team_b, scout_text
                    FROM matches
                    WHERE round_number BETWEEN :from_round AND :to_round
                """),
                {"from_round": int(from_round), "to_round": int(to_round)}
            ).mappings().all()

        if not rows:
            st.info("Nessun dato nel range selezionato.")
            return

        def fix_team(name: str) -> str:
            n = " ".join((name or "").split())
            if n.lower().startswith("gas sales bluenergy p"):
                return "Gas Sales Bluenergy Piacenza"
            if "grottazzolina" in n.lower():
                return "Yuasa Battery Grottazzolina"
            return n

        agg = {}
        def ensure(team: str):
            if team not in agg:
                agg[team] = {"Teams": team, "tot_serves": 0, "bt_punto": 0, "bt_errore": 0}

        for r in rows:
            ta = fix_team(r.get("team_a") or "")
            tb = fix_team(r.get("team_b") or "")
            ensure(ta); ensure(tb)

            scout_text = r.get("scout_text") or ""
            lines = [ln.strip() for ln in str(scout_text).splitlines() if ln and ln.strip()]
            scout_lines = [ln for ln in lines if ln[0] in ("*", "a")]

            rallies = []
            current = []

            for raw in scout_lines:
                c = code6(raw)
                if not c:
                    continue

                if is_serve(c):
                    if current:
                        rallies.append(current)
                    current = [c]
                    continue

                if not current:
                    continue

                current.append(c)

                if is_home_point(c) or is_away_point(c):
                    rallies.append(current)
                    current = []

            for rally in rallies:
                first = rally[0]
                home_served = first.startswith("*")
                away_served = first.startswith("a")

                if home_served:
                    agg[ta]["tot_serves"] += 1
                    if len(first) >= 6 and first[5] == "#":
                        agg[ta]["bt_punto"] += 1
                    elif len(first) >= 6 and first[5] == "=":
                        agg[ta]["bt_errore"] += 1

                if away_served:
                    agg[tb]["tot_serves"] += 1
                    if len(first) >= 6 and first[5] == "#":
                        agg[tb]["bt_punto"] += 1
                    elif len(first) >= 6 and first[5] == "=":
                        agg[tb]["bt_errore"] += 1

        df = pd.DataFrame(list(agg.values()))
        if df.empty:
            st.info("Nessun dato nel range selezionato.")
            return

        df["% Bt Punto/Tot Bt"] = (100.0 * df["bt_punto"] / df["tot_serves"].replace({0: pd.NA})).fillna(0.0)
        df["% Bt Errore/Tot Bt"] = (100.0 * df["bt_errore"] / df["tot_serves"].replace({0: pd.NA})).fillna(0.0)
        df["Ratio Errore/Punto"] = (df["bt_errore"] / df["bt_punto"].replace({0: pd.NA})).fillna(0.0)
        show_details = st.checkbox("Mostra colonne di controllo (Tot Bt, Bt#, Bt=)", value=False)

        df["Tot Bt"] = df["tot_serves"]
        df["Bt#"] = df["bt_punto"]
        df["Bt="] = df["bt_errore"]

        base_cols = ["Teams", "% Bt Punto/Tot Bt", "% Bt Errore/Tot Bt", "Ratio Errore/Punto"]
        detail_cols = ["Tot Bt", "Bt#", "Bt="] if show_details else []

        out = df[base_cols + detail_cols].copy()
        out = out.sort_values(by=["% Bt Punto/Tot Bt", "Teams"], ascending=[False, True]).reset_index(drop=True)


        def highlight_perugia_row(row):
            is_perugia = "perugia" in str(row["Teams"]).lower()
            style = "background-color: #fff3cd; font-weight: 800;" if is_perugia else ""
            return [style] * len(row)

        styled = (
            out.style
              .apply(highlight_perugia_row, axis=1)
              .format({
                  "% Bt Punto/Tot Bt": "{:.1f}",
                  "% Bt Errore/Tot Bt": "{:.1f}",
                  "Ratio Errore/Punto": "{:.2f}",
              })
              .set_table_styles([
                  {"selector": "th", "props": [("font-size", "24px"), ("text-align", "left"), ("padding", "10px 12px")]},
                  {"selector": "td", "props": [("font-size", "23px"), ("padding", "10px 12px")]},
              ])
        )
        st.dataframe(styled, width="stretch", hide_index=True)


    elif voce == "Confronto TEAM":
        # ==========================================================
        # Confronto TEAM (max 4 squadre) nel range giornate selezionato
        # Colonne:
        # TEAM | % BREAK TOTALE | BREAK GIOCATO | Bt- | Bt! | Bt+ | 1/2 | BT punto/errore/ratio
        # Definizioni (sul servizio code6):
        # - Battuta = is_serve(code6) -> SQ o SM
        # - Break totale % = punti a favore del battitore / battute totali
        # - Break giocato % = punti a favore del battitore / battute con segno tra (-,+,!,/)
        # - Bt-: % BP con Bt- = punti a favore del battitore / battute con '-' (6° char)
        # - Bt!: % BP con Bt! = punti a favore del battitore / battute con '!' (6° char)
        # - Bt+: % BP con Bt+ = punti a favore del battitore / battute con '+' (6° char)
        # - 1/2: % BP con Bt/ = punti a favore del battitore / battute con '/' (6° char)
        # - BT punto/errore/ratio: "P% / E% / R" su totale battute (P=#, E==, R=E/P)
        # ==========================================================

        with engine.begin() as conn:
            teams_raw = conn.execute(text("""
                SELECT DISTINCT squadra FROM (
                    SELECT team_a AS squadra FROM matches
                    WHERE round_number BETWEEN :from_round AND :to_round
                    UNION
                    SELECT team_b AS squadra FROM matches
                    WHERE round_number BETWEEN :from_round AND :to_round
                )
                WHERE squadra IS NOT NULL AND TRIM(squadra) <> ''
                ORDER BY squadra
            """), {"from_round": int(from_round), "to_round": int(to_round)}).mappings().all()

        def fix_team(name: str) -> str:
            n = " ".join((name or "").split())
            if n.lower().startswith("gas sales bluenergy p"):
                return "Gas Sales Bluenergy Piacenza"
            if "grottazzolina" in n.lower():
                return "Yuasa Battery Grottazzolina"
            return n

        teams = sorted({fix_team(r["squadra"]) for r in teams_raw if r.get("squadra")})

        selected = st.multiselect(
            "Seleziona fino a 4 squadre",
            options=teams,
            default=[],
            max_selections=4,
        )

        if not selected:
            st.info("Seleziona 1–4 squadre per vedere il confronto.")
            return

        # carica match nel range
        with engine.begin() as conn:
            matches = conn.execute(
                text("""
                    SELECT team_a, team_b, scout_text
                    FROM matches
                    WHERE round_number BETWEEN :from_round AND :to_round
                """),
                {"from_round": int(from_round), "to_round": int(to_round)}
            ).mappings().all()

        if not matches:
            st.info("Nessun match nel range selezionato.")
            return

        # aggregatori per team
        agg = {}
        def ensure(team: str):
            if team not in agg:
                agg[team] = {
                    "TEAM": team,
                    "tot_serves": 0, "tot_bp": 0,
                    "g_att": 0, "g_bp": 0,     # giocato (- + ! /)
                    "neg_att": 0, "neg_bp": 0,
                    "exc_att": 0, "exc_bp": 0,
                    "pos_att": 0, "pos_bp": 0,
                    "half_att": 0, "half_bp": 0,
                    "pt_cnt": 0, "err_cnt": 0,  # # and =
                }

        # helper per parse rally da scout_text
        def rallies_from_scout_text(scout_text: str):
            if not scout_text:
                return []
            lines = [ln.strip() for ln in str(scout_text).splitlines() if ln and ln.strip()]
            scout_lines = [ln for ln in lines if ln[0] in ("*", "a")]

            rallies = []
            current = []
            for raw in scout_lines:
                c = code6(raw)
                if not c:
                    continue
                if is_serve(c):
                    if current:
                        rallies.append(current)
                    current = [c]
                    continue
                if not current:
                    continue
                current.append(c)
                if is_home_point(c) or is_away_point(c):
                    rallies.append(current)
                    current = []
            return rallies

        def serve_sign(c6: str) -> str:
            return c6[5] if len(c6) >= 6 else ""

        for m in matches:
            ta = fix_team(m.get("team_a") or "")
            tb = fix_team(m.get("team_b") or "")
            # assicurati in agg solo se selezionate (ottimizziamo)
            if ta not in selected and tb not in selected:
                continue
            ensure(ta); ensure(tb)

            for rally in rallies_from_scout_text(m.get("scout_text") or ""):
                first = rally[0]
                if not is_serve(first):
                    continue

                home_served = first.startswith("*")
                away_served = first.startswith("a")

                home_point = any(is_home_point(x) for x in rally)
                away_point = any(is_away_point(x) for x in rally)

                # team che batte e "bp" (punto a favore del battitore)
                if home_served:
                    team = ta
                    bp = 1 if home_point else 0
                elif away_served:
                    team = tb
                    bp = 1 if away_point else 0
                else:
                    continue

                if team not in selected:
                    continue

                sgn = serve_sign(first)

                agg[team]["tot_serves"] += 1
                agg[team]["tot_bp"] += bp

                # Break giocato: solo - + ! /
                if sgn in ("-", "+", "!", "/"):
                    agg[team]["g_att"] += 1
                    agg[team]["g_bp"] += bp

                # per segno
                if sgn == "-":
                    agg[team]["neg_att"] += 1
                    agg[team]["neg_bp"] += bp
                elif sgn == "!":
                    agg[team]["exc_att"] += 1
                    agg[team]["exc_bp"] += bp
                elif sgn == "+":
                    agg[team]["pos_att"] += 1
                    agg[team]["pos_bp"] += bp
                elif sgn == "/":
                    agg[team]["half_att"] += 1
                    agg[team]["half_bp"] += bp

                # BT punto/errore/ratio
                if sgn == "#":
                    agg[team]["pt_cnt"] += 1
                elif sgn == "=":
                    agg[team]["err_cnt"] += 1

        df = pd.DataFrame([agg[t] for t in selected])
        if df.empty:
            st.info("Nessun dato per le squadre selezionate nel range.")
            return

        def safe_pct(num, den):
            return float(num) / float(den) * 100.0 if den else 0.0

        # calcoli finali
        df["% BREAK TOTALE"] = df.apply(lambda r: safe_pct(r["tot_bp"], r["tot_serves"]), axis=1)
        df["BREAK GIOCATO"] = df.apply(lambda r: safe_pct(r["g_bp"], r["g_att"]), axis=1)

        df["BREAK con BT. NEGATIVA"] = df.apply(lambda r: safe_pct(r["neg_bp"], r["neg_att"]), axis=1)
        df["BREAK con BT. ESCLAMATIVA"] = df.apply(lambda r: safe_pct(r["exc_bp"], r["exc_att"]), axis=1)
        df["BREAK con BT. POSITIVA"] = df.apply(lambda r: safe_pct(r["pos_bp"], r["pos_att"]), axis=1)
        df["1/2 PUNTO"] = df.apply(lambda r: safe_pct(r["half_bp"], r["half_att"]), axis=1)

        df["BT Punto %"] = df.apply(lambda r: safe_pct(r["pt_cnt"], r["tot_serves"]), axis=1)
        df["BT Errore %"] = df.apply(lambda r: safe_pct(r["err_cnt"], r["tot_serves"]), axis=1)
        df["BT Ratio (=/#)"] = df.apply(lambda r: (float(r["err_cnt"]) / float(r["pt_cnt"]) if r["pt_cnt"] else 0.0), axis=1)


        out = df[[
            "TEAM",
            "% BREAK TOTALE",
            "BREAK GIOCATO",
            "BREAK con BT. NEGATIVA",
            "BREAK con BT. ESCLAMATIVA",
            "BREAK con BT. POSITIVA",
            "1/2 PUNTO",
            "BT Punto %",
            "BT Errore %",
            "BT Ratio (=/#)",
        ]].copy()

        # formatting + highlight Perugia
        def highlight_perugia_row(row):
            is_perugia = "perugia" in str(row["TEAM"]).lower()
            style = "background-color: #fff3cd; font-weight: 800;" if is_perugia else ""
            return [style] * len(row)

        styled = (
            out.style
              .apply(highlight_perugia_row, axis=1)
              .format({
                  "% BREAK TOTALE": "{:.1f}",
                  "BREAK GIOCATO": "{:.1f}",
                  "BREAK con BT. NEGATIVA": "{:.1f}",
                  "BREAK con BT. ESCLAMATIVA": "{:.1f}",
                  "BREAK con BT. POSITIVA": "{:.1f}",
                  "1/2 PUNTO": "{:.1f}",
                                "BT Punto %": "{:.1f}",
                  "BT Errore %": "{:.1f}",
                  "BT Ratio (=/#)": "{:.2f}",
})
              .set_table_styles([
                  {"selector": "th", "props": [("font-size", "24px"), ("text-align", "left"), ("padding", "10px 12px")]},
                  {"selector": "td", "props": [("font-size", "23px"), ("padding", "10px 12px")]},
              ])
        )
        st.dataframe(styled, width="stretch", hide_index=True)

    elif voce == "GRAFICI":
        # ==========================================================
        # GRAFICI (max 4 squadre) nel range giornate selezionato
        # Opzioni:
        # 1) Distribuzione BP per tipo di Battuta
        # 2) Distribuzione tipo di battuta (su Tot battute)
        #
        # Codifica (6° carattere del servizio):
        #   '-' negativa, '!' esclamativa, '+' positiva, '/' mezzo punto,
        #   '#' punto, '=' errore
        # ==========================================================
        import matplotlib.pyplot as plt

        with engine.begin() as conn:
            teams_raw = conn.execute(text("""
                SELECT DISTINCT squadra FROM (
                    SELECT team_a AS squadra FROM matches
                    WHERE round_number BETWEEN :from_round AND :to_round
                    UNION
                    SELECT team_b AS squadra FROM matches
                    WHERE round_number BETWEEN :from_round AND :to_round
                )
                WHERE squadra IS NOT NULL AND TRIM(squadra) <> ''
                ORDER BY squadra
            """), {"from_round": int(from_round), "to_round": int(to_round)}).mappings().all()

        def fix_team(name: str) -> str:
            n = " ".join((name or "").split())
            if n.lower().startswith("gas sales bluenergy p"):
                return "Gas Sales Bluenergy Piacenza"
            if "grottazzolina" in n.lower():
                return "Yuasa Battery Grottazzolina"
            return n

        teams = sorted({fix_team(r["squadra"]) for r in teams_raw if r.get("squadra")})

        selected = st.multiselect(
            "Seleziona fino a 4 squadre",
            options=teams,
            default=[],
            max_selections=4,
            key="grafici_teams",
        )

        option = st.radio(
            "Seleziona grafico",
            [
                "Vedi distribuzione BP per tipo di Battuta",
                "Vedi distribuzione tipo di battuta",
            ],
            index=0,
            key="grafici_option",
        )

        # --- grafici compatti (2 colonne) ---
        FIGSIZE = (3.4, 3.4)   # ancora più compatto
        DPI = 120
        LABEL_FONTSIZE = 8
        TITLE_FONTSIZE = 10

        if not selected:
            st.info("Seleziona 1–4 squadre per vedere i grafici.")
            return

        with engine.begin() as conn:
            matches = conn.execute(
                text("""
                    SELECT team_a, team_b, scout_text
                    FROM matches
                    WHERE round_number BETWEEN :from_round AND :to_round
                """),
                {"from_round": int(from_round), "to_round": int(to_round)}
            ).mappings().all()

        if not matches:
            st.info("Nessun match nel range selezionato.")
            return

        def rallies_from_scout_text(scout_text: str):
            if not scout_text:
                return []
            lines = [ln.strip() for ln in str(scout_text).splitlines() if ln and ln.strip()]
            scout_lines = [ln for ln in lines if ln[0] in ("*", "a")]

            rallies = []
            current = []
            for raw in scout_lines:
                c = code6(raw)
                if not c:
                    continue
                if is_serve(c):
                    if current:
                        rallies.append(current)
                    current = [c]
                    continue
                if not current:
                    continue
                current.append(c)
                if is_home_point(c) or is_away_point(c):
                    rallies.append(current)
                    current = []
            return rallies

        def serve_sign(c6: str) -> str:
            return c6[5] if len(c6) >= 6 else ""

        stats = {}
        for t in selected:
            stats[t] = {
                "tot_serves": 0,
                "serve_counts": {"-": 0, "!": 0, "+": 0, "/": 0, "#": 0, "=": 0},
                "bp_counts": {"-": 0, "!": 0, "+": 0, "/": 0, "#": 0},
            }

        for m in matches:
            ta = fix_team(m.get("team_a") or "")
            tb = fix_team(m.get("team_b") or "")
            if ta not in selected and tb not in selected:
                continue

            for rally in rallies_from_scout_text(m.get("scout_text") or ""):
                first = rally[0]
                if not is_serve(first):
                    continue

                home_served = first.startswith("*")
                away_served = first.startswith("a")

                home_point = any(is_home_point(x) for x in rally)
                away_point = any(is_away_point(x) for x in rally)

                if home_served:
                    team = ta
                    bp = 1 if home_point else 0
                elif away_served:
                    team = tb
                    bp = 1 if away_point else 0
                else:
                    continue

                if team not in selected:
                    continue

                sgn = serve_sign(first)
                stats[team]["tot_serves"] += 1
                if sgn in stats[team]["serve_counts"]:
                    stats[team]["serve_counts"][sgn] += 1
                if sgn in stats[team]["bp_counts"]:
                    stats[team]["bp_counts"][sgn] += bp

        label_map_bp = {"-": "Neg", "!": "!", "+": "+", "/": "½", "#": "#"}
        label_map_all = {"-": "Neg", "!": "!", "+": "+", "/": "½", "#": "#", "=": "="}

        def counts_df(keys, labels, values):
            return pd.DataFrame({"Tipo": labels, "Conteggio": values})

        cols = st.columns(2)
        for i, t in enumerate(selected):
            col = cols[i % 2]
            with col:
                st.markdown(f"### {t}")

                if option == "Vedi distribuzione BP per tipo di Battuta":
                    keys = ["-", "!", "+", "/", "#"]
                    labels = [label_map_bp[k] for k in keys]
                    values = [stats[t]["bp_counts"].get(k, 0) for k in keys]
                    tot_bp = sum(values)

                    if tot_bp == 0:
                        st.info("Nessun Break Point nel range selezionato.")
                    else:
                        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
                        ax.pie(values, labels=labels, autopct="%1.1f%%", textprops={"fontsize": LABEL_FONTSIZE})
                        ax.set_title("BP per battuta", fontsize=TITLE_FONTSIZE)
                        st.pyplot(fig, clear_figure=True)

                        # Chicca: totale + dettaglio conteggi in expander
                        st.caption(f"Tot BP: **{tot_bp}** | Tot battute: **{stats[t]['tot_serves']}**")
                        with st.expander("Dettaglio conteggi BP", expanded=False):
                            st.dataframe(counts_df(keys, labels, values), width="stretch", hide_index=True)

                else:
                    keys = ["-", "!", "+", "/", "#", "="]
                    labels = [label_map_all[k] for k in keys]
                    values = [stats[t]["serve_counts"].get(k, 0) for k in keys]
                    tot_serves = sum(values)

                    if tot_serves == 0:
                        st.info("Nessuna battuta nel range selezionato.")
                    else:
                        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
                        ax.pie(values, labels=labels, autopct="%1.1f%%", textprops={"fontsize": LABEL_FONTSIZE})
                        ax.set_title("Distribuzione battute", fontsize=TITLE_FONTSIZE)
                        st.pyplot(fig, clear_figure=True)

                        # Chicca: totale + dettaglio conteggi in expander
                        st.caption(f"Tot battute: **{tot_serves}**")
                        with st.expander("Dettaglio conteggi battute", expanded=False):
                            st.dataframe(counts_df(keys, labels, values), width="stretch", hide_index=True)

            if i % 2 == 1 and i < len(selected) - 1:
                st.divider()
                cols = st.columns(2)


# =========================
# UI: GRAFICI 4 QUADRANTI
# =========================
def render_grafici_4_quadranti():
    st.header("GRAFICI 4 Quadranti")

    with engine.begin() as conn:
        bounds = conn.execute(text("""
            SELECT MIN(round_number) AS min_r, MAX(round_number) AS max_r
            FROM matches
            WHERE round_number IS NOT NULL
        """)).mappings().first()

    min_r = int((bounds["min_r"] or 1))
    max_r = int((bounds["max_r"] or 1))

    c1, c2 = st.columns(2)
    with c1:
        from_round = st.number_input("Da giornata", min_value=min_r, max_value=max_r, value=min_r, step=1, key="q_from")
    with c2:
        to_round = st.number_input("A giornata", min_value=min_r, max_value=max_r, value=max_r, step=1, key="q_to")

    if from_round > to_round:
        st.error("Range non valido: 'Da giornata' deve essere <= 'A giornata'.")
        st.stop()

    x_options = [
        "Side Out TOTALE",
        "Side Out SPIN",
        "Side Out FLOAT",
        "Side Out DIRETTO",
        "Side Out GIOCATO",
        "Side Out con RICE BUONA",
        "Side Out con RICE ESCALAMATIVA",
        "Side Out con RICE NEGATIVA",
        "Ricezione Errore e ½ punto",
    ]
    y_options = [
        "BREAK TOTALE",
        "BREAK GIOCATO",
        "BREAK con BT. NEGATIVA",
        "BREAK con BT. ESCLAMATIVA",
        "BREAK con BT. POSITIVA",
        "BT Punto e Bt ½ punto",
        "BT errore",
    ]

    cx, cy = st.columns(2)
    with cx:
        x_metric = st.selectbox("Ascisse (X)", x_options, index=0)
    with cy:
        y_metric = st.selectbox("Ordinate (Y)", y_options, index=0)

    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT
                    team_a, team_b, scout_text,
                    COALESCE(so_home_attempts,0) AS so_home_attempts,
                    COALESCE(so_home_wins,0)     AS so_home_wins,
                    COALESCE(so_away_attempts,0) AS so_away_attempts,
                    COALESCE(so_away_wins,0)     AS so_away_wins,

                    COALESCE(so_spin_home_attempts,0) AS so_spin_home_attempts,
                    COALESCE(so_spin_home_wins,0)     AS so_spin_home_wins,
                    COALESCE(so_spin_away_attempts,0) AS so_spin_away_attempts,
                    COALESCE(so_spin_away_wins,0)     AS so_spin_away_wins,

                    COALESCE(so_float_home_attempts,0) AS so_float_home_attempts,
                    COALESCE(so_float_home_wins,0)     AS so_float_home_wins,
                    COALESCE(so_float_away_attempts,0) AS so_float_away_attempts,
                    COALESCE(so_float_away_wins,0)     AS so_float_away_wins,

                    COALESCE(so_dir_home_wins,0) AS so_dir_home_wins,
                    COALESCE(so_dir_away_wins,0) AS so_dir_away_wins,

                    COALESCE(so_play_home_attempts,0) AS so_play_home_attempts,
                    COALESCE(so_play_home_wins,0)     AS so_play_home_wins,
                    COALESCE(so_play_away_attempts,0) AS so_play_away_attempts,
                    COALESCE(so_play_away_wins,0)     AS so_play_away_wins,

                    COALESCE(so_good_home_attempts,0) AS so_good_home_attempts,
                    COALESCE(so_good_home_wins,0)     AS so_good_home_wins,
                    COALESCE(so_good_away_attempts,0) AS so_good_away_attempts,
                    COALESCE(so_good_away_wins,0)     AS so_good_away_wins,

                    COALESCE(so_exc_home_attempts,0) AS so_exc_home_attempts,
                    COALESCE(so_exc_home_wins,0)     AS so_exc_home_wins,
                    COALESCE(so_exc_away_attempts,0) AS so_exc_away_attempts,
                    COALESCE(so_exc_away_wins,0)     AS so_exc_away_wins,

                    COALESCE(so_neg_home_attempts,0) AS so_neg_home_attempts,
                    COALESCE(so_neg_home_wins,0)     AS so_neg_home_wins,
                    COALESCE(so_neg_away_attempts,0) AS so_neg_away_attempts,
                    COALESCE(so_neg_away_wins,0)     AS so_neg_away_wins,

                    COALESCE(bp_home_attempts,0) AS bp_home_attempts,
                    COALESCE(bp_home_wins,0)     AS bp_home_wins,
                    COALESCE(bp_away_attempts,0) AS bp_away_attempts,
                    COALESCE(bp_away_wins,0)     AS bp_away_wins
                FROM matches
                WHERE round_number BETWEEN :from_round AND :to_round
            """),
            {"from_round": int(from_round), "to_round": int(to_round)}
        ).mappings().all()

    if not rows:
        st.info("Nessun match nel range selezionato.")
        return

    def fix_team(name: str) -> str:
        n = " ".join((name or "").split())
        if n.lower().startswith("gas sales bluenergy p"):
            return "Gas Sales Bluenergy Piacenza"
        if "grottazzolina" in n.lower():
            return "Yuasa Battery Grottazzolina"
        return n

    def safe_pct(num: float, den: float) -> float:
        return (float(num) / float(den) * 100.0) if den else 0.0

    agg = {}

    def ensure(team: str):
        if team not in agg:
            agg[team] = {
                "TEAM": team,
                "so_att": 0, "so_win": 0,
                "spin_att": 0, "spin_win": 0,
                "float_att": 0, "float_win": 0,
                "dir_win": 0,
                "so_play_att": 0, "so_play_win": 0,
                "so_good_att": 0, "so_good_win": 0,
                "so_exc_att": 0, "so_exc_win": 0,
                "so_neg_att": 0, "so_neg_win": 0,
                "rec_tot": 0, "rec_errhalf": 0,
                "bp_att": 0, "bp_win": 0,
                "g_att": 0, "g_bp": 0,
                "neg_att": 0, "neg_bp": 0,
                "exc_att": 0, "exc_bp": 0,
                "pos_att": 0, "pos_bp": 0,
                "half_att": 0, "half_bp": 0,
                "bt_hash": 0,
                "bt_err": 0,
                "tot_serves": 0,
            }

    def parse_scout(scout_text: str):
        if not scout_text:
            return [], []
        lines = [ln.strip() for ln in str(scout_text).splitlines() if ln and ln.strip()]
        scout_lines = [ln for ln in lines if ln[0] in ("*", "a")]
        rallies = []
        current = []
        for raw in scout_lines:
            c = code6(raw)
            if not c:
                continue
            if is_serve(c):
                if current:
                    rallies.append(current)
                current = [c]
                continue
            if not current:
                continue
            current.append(c)
            if is_home_point(c) or is_away_point(c):
                rallies.append(current)
                current = []
        return rallies, scout_lines

    for r in rows:
        ta = fix_team(r.get("team_a") or "")
        tb = fix_team(r.get("team_b") or "")
        ensure(ta); ensure(tb)

        # SideOut from DB
        agg[ta]["so_att"] += int(r["so_home_attempts"]); agg[ta]["so_win"] += int(r["so_home_wins"])
        agg[tb]["so_att"] += int(r["so_away_attempts"]); agg[tb]["so_win"] += int(r["so_away_wins"])

        agg[ta]["spin_att"] += int(r["so_spin_home_attempts"]); agg[ta]["spin_win"] += int(r["so_spin_home_wins"])
        agg[tb]["spin_att"] += int(r["so_spin_away_attempts"]); agg[tb]["spin_win"] += int(r["so_spin_away_wins"])

        agg[ta]["float_att"] += int(r["so_float_home_attempts"]); agg[ta]["float_win"] += int(r["so_float_home_wins"])
        agg[tb]["float_att"] += int(r["so_float_away_attempts"]); agg[tb]["float_win"] += int(r["so_float_away_wins"])

        agg[ta]["dir_win"] += int(r["so_dir_home_wins"]); agg[tb]["dir_win"] += int(r["so_dir_away_wins"])

        agg[ta]["so_play_att"] += int(r["so_play_home_attempts"]); agg[ta]["so_play_win"] += int(r["so_play_home_wins"])
        agg[tb]["so_play_att"] += int(r["so_play_away_attempts"]); agg[tb]["so_play_win"] += int(r["so_play_away_wins"])

        agg[ta]["so_good_att"] += int(r["so_good_home_attempts"]); agg[ta]["so_good_win"] += int(r["so_good_home_wins"])
        agg[tb]["so_good_att"] += int(r["so_good_away_attempts"]); agg[tb]["so_good_win"] += int(r["so_good_away_wins"])

        agg[ta]["so_exc_att"] += int(r["so_exc_home_attempts"]); agg[ta]["so_exc_win"] += int(r["so_exc_home_wins"])
        agg[tb]["so_exc_att"] += int(r["so_exc_away_attempts"]); agg[tb]["so_exc_win"] += int(r["so_exc_away_wins"])

        agg[ta]["so_neg_att"] += int(r["so_neg_home_attempts"]); agg[ta]["so_neg_win"] += int(r["so_neg_home_wins"])
        agg[tb]["so_neg_att"] += int(r["so_neg_away_attempts"]); agg[tb]["so_neg_win"] += int(r["so_neg_away_wins"])

        # Break total from DB
        agg[ta]["bp_att"] += int(r["bp_home_attempts"]); agg[ta]["bp_win"] += int(r["bp_home_wins"])
        agg[tb]["bp_att"] += int(r["bp_away_attempts"]); agg[tb]["bp_win"] += int(r["bp_away_wins"])

        rallies, scout_lines = parse_scout(r.get("scout_text") or "")

        # Reception error + half (/)
        for raw in scout_lines:
            c6 = code6(raw)
            if not c6:
                continue
            if is_home_rece(c6):
                agg[ta]["rec_tot"] += 1
                if len(c6) >= 6 and c6[5] in ("=", "/"):
                    agg[ta]["rec_errhalf"] += 1
            elif is_away_rece(c6):
                agg[tb]["rec_tot"] += 1
                if len(c6) >= 6 and c6[5] in ("=", "/"):
                    agg[tb]["rec_errhalf"] += 1

        # Break giocato + segni + BT #/=
        for rally in rallies:
            first = rally[0]
            if not is_serve(first):
                continue

            home_served = first.startswith("*")
            away_served = first.startswith("a")

            home_point = any(is_home_point(x) for x in rally)
            away_point = any(is_away_point(x) for x in rally)

            if home_served:
                team = ta
                bp = 1 if home_point else 0
            elif away_served:
                team = tb
                bp = 1 if away_point else 0
            else:
                continue

            sgn = first[5] if len(first) >= 6 else ""
            agg[team]["tot_serves"] += 1

            if sgn == "#":
                agg[team]["bt_hash"] += 1
            elif sgn == "=":
                agg[team]["bt_err"] += 1

            if sgn in ("-", "+", "!", "/"):
                agg[team]["g_att"] += 1
                agg[team]["g_bp"] += bp

            if sgn == "-":
                agg[team]["neg_att"] += 1; agg[team]["neg_bp"] += bp
            elif sgn == "!":
                agg[team]["exc_att"] += 1; agg[team]["exc_bp"] += bp
            elif sgn == "+":
                agg[team]["pos_att"] += 1; agg[team]["pos_bp"] += bp
            elif sgn == "/":
                agg[team]["half_att"] += 1; agg[team]["half_bp"] += bp

    df = pd.DataFrame(list(agg.values()))
    if df.empty:
        st.info("Nessun dato disponibile.")
        return

    def compute_x(row):
        if x_metric == "Side Out TOTALE":
            return safe_pct(row["so_win"], row["so_att"])
        if x_metric == "Side Out SPIN":
            return safe_pct(row["spin_win"], row["spin_att"])
        if x_metric == "Side Out FLOAT":
            return safe_pct(row["float_win"], row["float_att"])
        if x_metric == "Side Out DIRETTO":
            return safe_pct(row["dir_win"], row["so_att"])
        if x_metric == "Side Out GIOCATO":
            return safe_pct(row["so_play_win"], row["so_play_att"])
        if x_metric == "Side Out con RICE BUONA":
            return safe_pct(row["so_good_win"], row["so_good_att"])
        if x_metric == "Side Out con RICE ESCALAMATIVA":
            return safe_pct(row["so_exc_win"], row["so_exc_att"])
        if x_metric == "Side Out con RICE NEGATIVA":
            return safe_pct(row["so_neg_win"], row["so_neg_att"])
        if x_metric == "Ricezione Errore e ½ punto":
            return safe_pct(row["rec_errhalf"], row["rec_tot"])
        return 0.0

    def compute_y(row):
        if y_metric == "BREAK TOTALE":
            return safe_pct(row["bp_win"], row["bp_att"])
        if y_metric == "BREAK GIOCATO":
            return safe_pct(row["g_bp"], row["g_att"])
        if y_metric == "BREAK con BT. NEGATIVA":
            return safe_pct(row["neg_bp"], row["neg_att"])
        if y_metric == "BREAK con BT. ESCLAMATIVA":
            return safe_pct(row["exc_bp"], row["exc_att"])
        if y_metric == "BREAK con BT. POSITIVA":
            return safe_pct(row["pos_bp"], row["pos_att"])
        if y_metric == "BT Punto e Bt ½ punto":
            # quota battute (# + /) su tot battute
            return safe_pct((row["bt_hash"] + row["half_att"]), row["tot_serves"])
        if y_metric == "BT errore":
            return safe_pct(row["bt_err"], row["tot_serves"])
        return 0.0

    df["X"] = df.apply(compute_x, axis=1)
    df["Y"] = df.apply(compute_y, axis=1)

    import matplotlib.pyplot as plt

    x_med = float(df["X"].median())
    y_med = float(df["Y"].median())

    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=120)
    ax.scatter(df["X"], df["Y"])

    ax.axvline(x_med, linestyle="--")
    ax.axhline(y_med, linestyle="--")

    for _, row in df.iterrows():
        ax.annotate(str(row["TEAM"]), (row["X"], row["Y"]), fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel(f"{x_metric} (%)")
    ax.set_ylabel(f"{y_metric} (%)")
    ax.set_title("Grafico a 4 quadranti – Squadre (range giornate)")

    st.pyplot(fig, clear_figure=True)
    st.caption(f"Linee dei quadranti: mediana X = {x_med:.1f} | mediana Y = {y_med:.1f}")



# =========================
# UI: IMPORT RUOLI (ROSTER)
# =========================
def render_import_ruoli(admin_mode: bool):
    st.header("Import Ruoli (Roster)")

    if not admin_mode:
        st.warning("Accesso riservato allo staff (admin).")
        return

    # --- Import XLSX ---
    up = st.file_uploader("Carica file Ruoli (.xlsx)", type=["xlsx"])
    st.info("Attese colonne: Team | Nome | Ruolo | N° (ID facoltativo).")

    season = st.text_input("Stagione (obbligatoria, es. 2025-26)", value="2025-26")

    if up is not None:
        df = pd.read_excel(up)
        df.columns = [str(c).strip() for c in df.columns]

        required = ["Team", "Nome", "Ruolo", "N°"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Mancano colonne: {missing}")
            st.stop()

        df = df.copy()
        df["Team"] = df["Team"].astype(str).apply(fix_team_name)
        df["team_norm"] = df["Team"].astype(str).apply(team_norm)
        df["Nome"] = df["Nome"].astype(str).str.strip()
        df["Ruolo"] = df["Ruolo"].astype(str).str.strip()
        df["N°"] = pd.to_numeric(df["N°"], errors="coerce").astype("Int64")

        df = df.dropna(subset=["N°", "team_norm"])
        df["N°"] = df["N°"].astype(int)

        st.subheader("Preview import")
        st.dataframe(df[["Team", "Nome", "Ruolo", "N°"]].head(80), width="stretch", hide_index=True)

        if st.button("Importa/aggiorna roster nel DB", key="btn_roster_import"):
            sql = """
            INSERT INTO roster (season, team_raw, team_norm, jersey_number, player_name, role, created_at)
            VALUES (:season, :team_raw, :team_norm, :jersey_number, :player_name, :role, :created_at)
            ON CONFLICT(season, team_norm, jersey_number) DO UPDATE SET
                team_raw = excluded.team_raw,
                player_name = excluded.player_name,
                role = excluded.role,
                created_at = excluded.created_at
            """
            now = datetime.now(timezone.utc).isoformat()
            with engine.begin() as conn:
                for _, r in df.iterrows():
                    conn.execute(
                        text(sql),
                        {
                            "season": season,
                            "team_raw": str(r["Team"]),
                            "team_norm": str(r["team_norm"]),
                            "jersey_number": int(r["N°"]),
                            "player_name": str(r["Nome"]),
                            "role": str(r["Ruolo"]),
                            "created_at": now,
                        }
                    )
            st.success("Roster importato/aggiornato.")
            st.rerun()

    st.divider()
    st.subheader("Correggi / Elimina record")

    # --- Carica roster dal DB per la stagione ---
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT id, season, team_raw, team_norm, jersey_number, player_name, role
                FROM roster
                WHERE season = :season
                ORDER BY team_raw, jersey_number
            """),
            {"season": season}
        ).mappings().all()

    if not rows:
        st.info("Nessun record in roster per questa stagione. Importa un file .xlsx sopra.")
        return

    df_db = pd.DataFrame(rows)

    # ===== Selezione: Team -> Giocatore (con filtro nome) =====
    teams = sorted(df_db["team_raw"].dropna().unique().tolist())
    team_sel = st.selectbox("Team", teams, index=0, key="edit_team")

    filtro_nome = st.text_input("Filtro nome (opzionale)", value="", key="edit_name_filter").strip().lower()

    df_team = df_db[df_db["team_raw"] == team_sel].copy()
    if filtro_nome:
        df_team = df_team[df_team["player_name"].fillna("").str.lower().str.contains(filtro_nome)]

    if df_team.empty:
        st.warning("Nessun giocatore trovato con questi filtri.")
        return

    df_team = df_team.sort_values(by=["jersey_number", "player_name"], ascending=[True, True])

    # opzioni: usiamo id come chiave stabile
    options = df_team["id"].tolist()

    def fmt_player(rid: int) -> str:
        r = df_team[df_team["id"] == rid].iloc[0]
        num = int(r["jersey_number"]) if pd.notna(r["jersey_number"]) else 0
        name = str(r["player_name"] or "").strip()
        role = str(r["role"] or "").strip()
        return f"{num:02d} — {name}  ({role})"

    player_id = st.selectbox("Giocatore", options, format_func=fmt_player, key="edit_player_id")

    rec = df_team[df_team["id"] == player_id].iloc[0].to_dict()
    st.caption(f"Record selezionato: id={rec['id']} | season={rec['season']} | team_norm={rec['team_norm']}")

    # campi editabili
    c3, c4 = st.columns(2)
    with c3:
        new_name = st.text_input("Nome giocatore", value=str(rec.get("player_name") or ""), key="edit_name")
        new_team_raw = st.text_input("Team (testo)", value=str(rec.get("team_raw") or ""), key="edit_team_raw")
    with c4:
        role_options = ["Alzatore", "Opposto", "Centrale", "Schiacciatore", "Libero"]
        current_role = str(rec.get("role") or "").strip()
        if current_role and current_role not in role_options:
            role_options = [current_role] + role_options
        new_role = st.selectbox("Ruolo", role_options, index=role_options.index(current_role) if current_role in role_options else 0, key="edit_role")
        new_num = st.number_input("Numero maglia", min_value=0, max_value=99, value=int(rec.get("jersey_number") or 0), step=1, key="edit_jersey")

    c5, c6 = st.columns(2)
    with c5:
        if st.button("Salva correzione", key="btn_roster_save"):
            team_raw_fixed = fix_team_name(new_team_raw)
            team_norm_fixed = team_norm(team_raw_fixed)
            now = datetime.now(timezone.utc).isoformat()

            with engine.begin() as conn:
                conn.execute(
                    text("""
                        UPDATE roster
                        SET team_raw = :team_raw,
                            team_norm = :team_norm,
                            jersey_number = :jersey_number,
                            player_name = :player_name,
                            role = :role,
                            created_at = :created_at
                        WHERE id = :id
                    """),
                    {
                        "team_raw": team_raw_fixed,
                        "team_norm": team_norm_fixed,
                        "jersey_number": int(new_num),
                        "player_name": str(new_name).strip(),
                        "role": str(new_role).strip(),
                        "created_at": now,
                        "id": int(rec["id"]),
                    }
                )
            st.success("Record aggiornato.")
            st.rerun()

    with c6:
        confirm_del = st.checkbox("Confermo: elimina questo record", value=False, key="confirm_del_roster")
        if st.button("Elimina record", disabled=not confirm_del, key="btn_roster_delete"):
            with engine.begin() as conn:
                conn.execute(text("DELETE FROM roster WHERE id = :id"), {"id": int(rec["id"])})
            st.success("Record eliminato.")
            st.rerun()


def render_sideout_players_by_role():
    st.header("Indici Side Out - Giocatori (per ruolo)")

    voce = st.radio(
        "Seleziona indice",
        ["SIDE OUT TOTALE", "SIDE OUT SPIN", "SIDE OUT FLOAT"],
        index=0,
        key="sop_voce",
    )

    # ===== RANGE GIORNATE =====
    with engine.begin() as conn:
        bounds = conn.execute(text("""
            SELECT MIN(round_number) AS min_r, MAX(round_number) AS max_r
            FROM matches
            WHERE round_number IS NOT NULL
        """)).mappings().first()

    min_r = int(bounds["min_r"] or 1)
    max_r = int(bounds["max_r"] or 1)

    c1, c2 = st.columns(2)
    with c1:
        from_round = st.number_input("Da giornata", min_value=min_r, max_value=max_r, value=min_r, step=1, key="sop_from")
    with c2:
        to_round = st.number_input("A giornata", min_value=min_r, max_value=max_r, value=max_r, step=1, key="sop_to")

    if from_round > to_round:
        st.error("Range non valido: 'Da giornata' deve essere <= 'A giornata'.")
        st.stop()

    # ===== RUOLI (multi) + MIN RICE =====
    with engine.begin() as conn:
        role_rows = conn.execute(text("""
            SELECT DISTINCT role
            FROM roster
            WHERE role IS NOT NULL AND TRIM(role) <> ''
            ORDER BY role
        """)).mappings().all()
    role_options = [r["role"] for r in role_rows] if role_rows else []

    sel_roles = st.multiselect(
        "Filtra per ruolo (selezione multipla)",
        options=role_options,
        default=role_options,
        key="sop_roles",
    )

    min_recv = st.number_input(
        "Numero minimo di ricezioni (sotto questo valore il giocatore non viene mostrato)",
        min_value=0,
        value=10,
        step=1,
        key="sop_minrecv",
    )

    st.info(
        "Si precisa che il ranking rappresenta la percentuale di side out ottenuta durante la ricezione da parte del giocatore indicato. "
        "Selezionando il titolo di ciascuna colonna, i nominativi saranno ordinati in base al parametro corrispondente."
    )

    # ===== MATCHES NEL RANGE =====
    with engine.begin() as conn:
        matches = conn.execute(text("""
            SELECT team_a, team_b, scout_text
            FROM matches
            WHERE round_number BETWEEN :from_round AND :to_round
        """), {"from_round": int(from_round), "to_round": int(to_round)}).mappings().all()

    if not matches:
        st.info("Nessun match nel range.")
        return

    # ===== ROSTER LOOKUP (team_norm + jersey -> name/role) =====
    # NB: usiamo team_norm per ridurre varianti di nome squadra
    def team_norm_key(name: str) -> str:
        return norm(name)

    with engine.begin() as conn:
        roster = conn.execute(text("""
            SELECT season, team_norm, jersey_number, player_name, role
            FROM roster
        """)).mappings().all()

    roster_map = {}
    for r in roster:
        tn = (r.get("team_norm") or "").strip().lower()
        jn = r.get("jersey_number")
        if tn and jn is not None:
            roster_map[(tn, int(jn))] = {
                "player_name": (r.get("player_name") or "").strip(),
                "role": (r.get("role") or "").strip(),
            }

    # ===== Parse rallies =====
    def parse_rallies(scout_text: str):
        if not scout_text:
            return []
        lines = [ln.strip() for ln in str(scout_text).splitlines() if ln and str(ln).strip()]
        rallies = []
        cur = []
        for raw in lines:
            c = code6(raw)
            if not c:
                continue
            if is_serve(c):
                if cur:
                    rallies.append(cur)
                cur = [c]
                continue
            if not cur:
                continue
            cur.append(c)
            if is_home_point(c) or is_away_point(c):
                rallies.append(cur)
                cur = []
        if cur:
            rallies.append(cur)
        return rallies

    def is_recv_code(c6: str, prefix: str) -> bool:
        # ricezione = RQ o RM
        return len(c6) >= 5 and c6[0] == prefix and c6[3:5] in ("RQ", "RM")

    def recv_kind_ok(c6: str) -> bool:
        # filtro indice: totale / spin / float
        # Qui "SPIN" = ricezioni RQ ; "FLOAT" = ricezioni RM (non battuta SQ/SM)
        if voce == "SIDE OUT TOTALE":
            return True
        if voce == "SIDE OUT SPIN":
            return len(c6) >= 5 and c6[3:5] == "RQ"
        if voce == "SIDE OUT FLOAT":
            return len(c6) >= 5 and c6[3:5] == "RM"
        return True
        if voce == "SIDE OUT SPIN":
            return is_home_spin(c6) or is_away_spin(c6)
        if voce == "SIDE OUT FLOAT":
            return is_home_float(c6) or is_away_float(c6)
        return True

    # ===== Accumulator per player =====
    # key = (team_norm, jersey_number)
    acc = {}

    def bump(team_name: str, jersey: int, win: bool, direct: bool):
        tn = team_norm_key(team_name)
        k = (tn, int(jersey))
        if k not in acc:
            acc[k] = {"team": team_name, "jersey": int(jersey), "recv_att": 0, "so_win": 0, "so_dir": 0}
        acc[k]["recv_att"] += 1
        if win:
            acc[k]["so_win"] += 1
        if direct:
            acc[k]["so_dir"] += 1

    # ===== Scan matches =====
    for m in matches:
        team_a = m.get("team_a") or ""
        team_b = m.get("team_b") or ""
        rallies = parse_rallies(m.get("scout_text") or "")

        for r in rallies:
            # ricezione home/away (c'è una sola ricezione per rally, prendiamo la prima che troviamo)
            home_recv = next((x for x in r if is_recv_code(x, "*") and recv_kind_ok(x)), None)
            away_recv = next((x for x in r if is_recv_code(x, "a") and recv_kind_ok(x)), None)

            home_point = any(is_home_point(x) for x in r)
            away_point = any(is_away_point(x) for x in r)

            if home_recv:
                # jersey number 2 cifre pos 1-2
                try:
                    jersey = int(home_recv[1:3])
                except Exception:
                    continue
                direct = bool(home_point and first_attack_after_reception_is_winner(r, "*"))
                bump(team_a, jersey, win=home_point, direct=direct)

            if away_recv:
                try:
                    jersey = int(away_recv[1:3])
                except Exception:
                    continue
                direct = bool(away_point and first_attack_after_reception_is_winner(r, "a"))
                bump(team_b, jersey, win=away_point, direct=direct)

    if not acc:
        st.info("Nessuna ricezione trovata nel range selezionato.")
        return

    # ===== Build dataframe =====
    rows = []
    for (tn, jersey), v in acc.items():
        info = roster_map.get((tn, jersey), {"player_name": f"N°{jersey:02d}", "role": ""})
        rows.append({
            "Nome giocatore": (info.get("player_name") or f"N°{jersey:02d}").strip(),
            "Ruolo": (info.get("role") or "").strip(),
            "Squadra": v["team"],
            "recv_att": int(v["recv_att"]),
            "so_win": int(v["so_win"]),
            "so_dir": int(v["so_dir"]),
        })

    df = pd.DataFrame(rows)

    # filtro ruoli
    if sel_roles:
        df = df[df["Ruolo"].isin(sel_roles)].copy()

    # filtro minimo ricezioni
    df = df[df["recv_att"] >= int(min_recv)].copy()

    if df.empty:
        st.info("Nessun giocatore soddisfa i filtri selezionati.")
        return

    # Tot ricezioni di squadra (per % Ply/Team)
    team_tot = df.groupby("Squadra", as_index=False)["recv_att"].sum().rename(columns={"recv_att": "team_recv"})
    df = df.merge(team_tot, on="Squadra", how="left")

    df["% Ply/Team"] = df.apply(lambda r: pct(int(r["recv_att"]), int(r["team_recv"])), axis=1)
    df["% di SO"] = df.apply(lambda r: pct(int(r["so_win"]), int(r["recv_att"])), axis=1)
    df["% di SO-d"] = df.apply(lambda r: pct(int(r["so_dir"]), int(r["recv_att"])), axis=1)

    # Dedup robusto: stessa persona può comparire due volte per errori roster -> raggruppo per Nome+Squadra
    df = df.groupby(["Nome giocatore", "Squadra"], as_index=False).agg({
        "Ruolo": "first",
        "recv_att": "sum",
        "so_win": "sum",
        "so_dir": "sum",
        "team_recv": "first",
    })
    df["% Ply/Team"] = df.apply(lambda r: pct(int(r["recv_att"]), int(r["team_recv"])), axis=1)
    df["% di SO"] = df.apply(lambda r: pct(int(r["so_win"]), int(r["recv_att"])), axis=1)
    df["% di SO-d"] = df.apply(lambda r: pct(int(r["so_dir"]), int(r["recv_att"])), axis=1)

    # Ranking fisso su % di SO (poi n ricezioni)
    df_rank = df.sort_values(by=["% di SO", "recv_att"], ascending=[False, False]).reset_index(drop=True)
    df_rank.insert(0, "Ranking", range(1, len(df_rank) + 1))

    out = df_rank[[
        "Ranking",
        "Nome giocatore",
        "Squadra",
        "recv_att",
        "% Ply/Team",
        "% di SO",
        "% di SO-d",
    ]].rename(columns={"recv_att": "N° ricezioni fatte"}).copy()

    # ===== Styling =====
    def highlight_perugia(row):
        return ["background-color: #fff3cd; font-weight: 800;" if "perugia" in str(row["Squadra"]).lower() else "" for _ in row]

    styled = (
        out.style
          .apply(highlight_perugia, axis=1)
          .format({
              "Ranking": "{:.0f}",
              "N° ricezioni fatte": "{:.0f}",
              "% Ply/Team": "{:.1f}",
              "% di SO": "{:.1f}",
              "% di SO-d": "{:.1f}",
          })
          .set_table_styles([
              {"selector": "th", "props": [("font-size", "22px"), ("text-align", "left"), ("padding", "10px 12px")]},
              {"selector": "td", "props": [("font-size", "21px"), ("padding", "10px 12px")]},
          ])
          .set_properties(subset=["% di SO"], **{"font-weight": "900", "background-color": "#e8f5e9"})
    )

    st.dataframe(styled, width="stretch", hide_index=True)

# =========================
# MAIN
# =========================
init_db()

st.sidebar.title("Volley App")
page = st.sidebar.radio(
    "Vai a:",
    [
        "Home",
        "Import DVW (solo staff)",
        "Import Ruoli (solo staff)",
        "Indici Side Out - Squadre",
        "Indici Fase Break – Squadre",
        "GRAFICI 4 Quadranti",
        "Indici Side Out - Giocatori (per ruolo)",
        "Indici Break Point - Giocatori (per ruolo)",
        "Classifiche Fondamentali - Squadre",
        "Classifiche Fondamentali - Giocatori (per ruolo)",
        "Punti per Set",
    ],
)

ADMIN_MODE = st.sidebar.checkbox("Modalità staff (admin)", value=True)

if page == "Home":
    st.header("Home")
    st.info("Usa il menu a sinistra per navigare. L'import è riservato allo staff.")

elif page == "Import DVW (solo staff)":
    render_import(ADMIN_MODE)

elif page == "Import Ruoli (solo staff)":
    render_import_ruoli(ADMIN_MODE)

elif page == "Indici Side Out - Squadre":
    render_sideout_team()

elif page == "Indici Fase Break – Squadre":
    render_break_team()

elif page == "GRAFICI 4 Quadranti":
    render_grafici_4_quadranti()


elif page == "Indici Side Out - Giocatori (per ruolo)":
    render_sideout_players_by_role()

elif page == "Indici Break Point - Giocatori (per ruolo)":
    st.header(page)
    st.info("In costruzione.")


else:
    st.header(page)
    st.info("In costruzione.")
