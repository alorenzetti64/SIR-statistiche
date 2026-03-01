
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
        "Team",
        "sets_total",
        "Punti Team/Set",
        "Nome giocatore",
        "Set giocati",
        "Punti/Set Giocatore",
        "% Ply/Team",
        "pts_serve",
        "pts_attack",
        "pts_block",
    ]].rename(columns={
        "Team": "Nome Team",
        "sets_total": "Set giocati dal Team",
        "Punti Team/Set": "Punti per Set (Team)",
        "Set giocati": "Set giocati dal Giocatore",
        "Punti/Set Giocatore": "Punti per Set (Giocatore)",
        "% Ply/Team": "% Ply/Team",
        "pts_serve": "Punti in Battuta",
        "pts_attack": "Punti in Attacco",
        "pts_block": "Punti a Muro",
    }).copy()

    # Trasforma i punti per fondamentale in "per set giocati dal giocatore"
    if "Set giocati dal Giocatore" in out.columns and out["Set giocati dal Giocatore"].notna().any():
        denom = out["Set giocati dal Giocatore"].replace(0, pd.NA).astype("float")
        for col in ["Punti in Battuta", "Punti in Attacco", "Punti a Muro"]:
            if col in out.columns:
                out[col] = (out[col].astype("float") / denom).fillna(0.0).round(1)

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
# UI: BREAK POINT - GIOCATORI (per ruolo)
# =========================
def render_break_players_by_role():
    st.header("Indici Break Point - Giocatori (per ruolo)")

    # ===== RANGE GIORNATE =====
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
        from_round = st.number_input("Da giornata", min_value=min_r, max_value=max_r, value=min_r, step=1, key="bppl_from")
    with c2:
        to_round = st.number_input("A giornata", min_value=min_r, max_value=max_r, value=max_r, step=1, key="bppl_to")

    if from_round > to_round:
        st.error("Range non valido.")
        st.stop()

    # ===== ROSTER (ruoli) =====
    season = st.text_input("Stagione roster (deve coincidere con quella importata)", value="2025-26", key="bppl_season")

    with engine.begin() as conn:
        roster_rows = conn.execute(text("""
            SELECT team_raw, team_norm, jersey_number, player_name, role, created_at
            FROM roster
            WHERE season = :season
        """), {"season": season}).mappings().all()

    if not roster_rows:
        st.warning("Roster vuoto per questa stagione: importa prima i ruoli (pagina Import Ruoli).")
        return

    df_roster = pd.DataFrame(roster_rows)
    df_roster["created_at"] = df_roster["created_at"].fillna("")
    df_roster = (
        df_roster.sort_values(by=["team_norm", "jersey_number", "created_at"])
                 .drop_duplicates(subset=["team_norm", "jersey_number"], keep="last")
    )

    roles_all = sorted(df_roster["role"].dropna().unique().tolist())
    roles_sel = st.multiselect("Filtro ruoli (puoi selezionarne quanti vuoi)", options=roles_all, default=roles_all, key="bppl_roles")

    min_serves = st.number_input(
        "Numero minimo di battute (sotto questo numero il giocatore non appare)",
        min_value=0, max_value=500, value=10, step=1, key="bppl_min_serves"
    )

    st.info(
        "Si precisa che il ranking rappresenta la percentuale di Break Point ottenuta durante la battuta da parte del giocatore indicato. "
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

    def parse_rallies(scout_text: str):
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

    agg = {}
    team_agg = {}

    def ensure_player(team_raw: str, num: int):
        tnorm = team_norm(team_raw)
        key = (tnorm, num)
        if key not in agg:
            agg[key] = {
                "team_norm": tnorm,
                "Squadra": team_raw,
                "N°": num,
                "serves": 0,
                "bp_win": 0,
                "aces": 0,
                "errors": 0,
                "played_serves": 0,
                "played_bp_win": 0,
            }
        return agg[key]

    def ensure_team(team_raw: str):
        tnorm = team_norm(team_raw)
        if tnorm not in team_agg:
            team_agg[tnorm] = {"team_norm": tnorm, "Squadra": team_raw, "serves": 0, "bp_win": 0}
        return team_agg[tnorm]

    for m in matches:
        team_a = fix_team_name(m.get("team_a") or "")
        team_b = fix_team_name(m.get("team_b") or "")

        rallies = parse_rallies(m.get("scout_text") or "")
        for rally in rallies:
            if not rally:
                continue
            first = rally[0]
            if not is_serve(first):
                continue

            home_served = first.startswith("*")
            away_served = first.startswith("a")

            home_point = any(is_home_point(x) for x in rally)
            away_point = any(is_away_point(x) for x in rally)

            if home_served:
                serving_team = team_a
                bp = 1 if home_point else 0
            elif away_served:
                serving_team = team_b
                bp = 1 if away_point else 0
            else:
                continue

            num = serve_player_number(first)
            if num is None:
                continue

            rec = ensure_player(serving_team, num)
            rec["serves"] += 1
            rec["bp_win"] += bp

            sgn = serve_sign(first)
            if sgn == "#":
                rec["aces"] += 1
            elif sgn == "=":
                rec["errors"] += 1

            if sgn not in ("#", "="):
                rec["played_serves"] += 1
                rec["played_bp_win"] += bp

            t = ensure_team(serving_team)
            t["serves"] += 1
            t["bp_win"] += bp

    df = pd.DataFrame(list(agg.values()))
    if df.empty:
        st.info("Nessun dato (battute) nel range selezionato.")
        return

    df = df.merge(
        df_roster[["team_norm", "jersey_number", "player_name", "role"]],
        left_on=["team_norm", "N°"],
        right_on=["team_norm", "jersey_number"],
        how="left",
    ).drop(columns=["jersey_number"])

    df.rename(columns={"player_name": "Nome giocatore", "role": "Ruolo"}, inplace=True)
    df["Nome giocatore"] = df["Nome giocatore"].fillna("(non in roster)")
    df["Ruolo"] = df["Ruolo"].fillna("(non in roster)")

    if roles_sel:
        df = df[df["Ruolo"].isin(roles_sel)].copy()

    df = df[df["serves"] >= int(min_serves)].copy()
    if df.empty:
        st.info("Nessun giocatore supera il filtro del numero minimo di battute.")
        return

    df["% di BP"] = (100.0 * df["bp_win"] / df["serves"].replace({0: pd.NA})).fillna(0.0)
    df["% di BP giocato"] = (100.0 * df["played_bp_win"] / df["played_serves"].replace({0: pd.NA})).fillna(0.0)

    df_team = pd.DataFrame(list(team_agg.values()))
    df_team["% Team BP"] = (100.0 * df_team["bp_win"] / df_team["serves"].replace({0: pd.NA})).fillna(0.0)
    df = df.merge(df_team[["team_norm", "% Team BP"]], on="team_norm", how="left")
    df["Diff. Team"] = df["% di BP"].fillna(0.0) - df["% Team BP"].fillna(0.0)

    def safe_ratio(err, ace):
        try:
            err = float(err or 0)
            ace = float(ace or 0)
        except Exception:
            return 0.0
        if ace == 0:
            return float("inf") if err > 0 else 0.0
        return err / ace

    df["Ratio err/p.to"] = df.apply(lambda r: safe_ratio(r["errors"], r["aces"]), axis=1)

    df_rank = df.sort_values(by=["% di BP", "serves"], ascending=[False, False]).reset_index(drop=True)
    df_rank.insert(0, "Ranking", range(1, len(df_rank) + 1))

    out = df_rank[[
        "Ranking",
        "Nome giocatore",
        "Squadra",
        "serves",
        "% di BP",
        "Diff. Team",
        "Ratio err/p.to",
        "% di BP giocato",
    ]].rename(columns={"serves": "Battute", "Ratio err/p.to": "Err/P.ti"}).copy()

    # Ratio come stringa (1 decimale) per render uniforme in Streamlit
    def ratio_str(x):
        try:
            if x == float("inf"):
                return "∞"
            return f"{float(x):.1f}"
        except Exception:
            return ""

    out["Err/P.ti"] = out["Err/P.ti"].apply(ratio_str)


    def highlight_perugia(row):
        is_perugia = "perugia" in str(row["Squadra"]).lower()
        style = "background-color: #fff3cd; font-weight: 800;" if is_perugia else ""
        return [style] * len(row)

    styled = (
        out.style
          .apply(highlight_perugia, axis=1)
          .set_properties(subset=["% di BP"], **{"background-color": "#e7f5ff", "font-weight": "900"})
          .format({
              "Ranking": "{:.0f}",
              "N° di Battute Fatte": "{:.0f}",
              "% di BP": "{:.1f}",
              "Diff. Team": "{:.1f}",
              "% di BP giocato": "{:.1f}",
          })
    )
    styled = styled.set_table_styles([
        {"selector": "th", "props": [("font-size", "22px"), ("text-align", "left"), ("padding", "8px 10px")]},
        {"selector": "td", "props": [("font-size", "21px"), ("padding", "8px 10px")]},
    ])

    st.dataframe(styled, width="stretch", hide_index=True)


# =========================
# UI: CLASSIFICHE FONDAMENTALI - SQUADRE
# =========================
def render_fondamentali_team():
    st.header("Classifiche Fondamentali - Squadre")

    voce = st.radio(
        "Seleziona fondamentale",
        [
            "Battuta",
            "Ricezione",
            "Attacco",
            "Muro",
            "Difesa",
        ],
        index=0,
        key="fund_team_voce",
    )

    # ===== FILTRO RANGE GIORNATE =====
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
        from_round = st.number_input("Da giornata", min_value=min_r, max_value=max_r, value=min_r, step=1, key="fund_from")
    with c2:
        to_round = st.number_input("A giornata", min_value=min_r, max_value=max_r, value=max_r, step=1, key="fund_to")

    if from_round > to_round:
        st.error("Range non valido: 'Da giornata' deve essere <= 'A giornata'.")
        st.stop()


    st.info("Sezione in costruzione. Ho già impostato il menu dei fondamentali: ora decidiamo insieme le metriche e le tabelle per ciascuna voce.")

    # Placeholder strutturale per sviluppo step-by-step (senza patch sparse)
    if voce == "Battuta":
        st.subheader("Battuta")

        # ===== FILTRI TIPO BATTUTA (SQ / SM) =====
        cbt1, cbt2 = st.columns(2)
        with cbt1:
            use_spin = st.checkbox("Battuta SPIN", value=True, key="fund_srv_spin")
        with cbt2:
            use_float = st.checkbox("Battuta FLOAT", value=True, key="fund_srv_float")

        # Regola: se spunti entrambe -> tutte; se spunti una -> solo quella; se nessuna -> tutte (fallback)
        if not use_spin and not use_float:
            use_spin = True
            use_float = True

        allowed_types = set()
        if use_spin:
            allowed_types.add("SQ")
        if use_float:
            allowed_types.add("SM")

        st.caption(
            "L’efficienza della battuta è calcolata in questo modo: "
            "(Punti + ½ P.*0,8 + Pos.*0,45 + Esc*0,3 + Neg*0,15 – Err) / Tot. "
            "Dove i coefficienti sono le % medie di B.Point con quel tipo di ricezione del campionato."
        )

        # ===== MATCHES NEL RANGE =====
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
            st.info("Nessun match nel range selezionato.")
            return

        def parse_rallies(scout_text: str):
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

        def fix_team(name: str) -> str:
            # riusa le stesse regole già usate altrove
            return fix_team_name(name)

        agg = {}  # team -> counts

        def ensure(team: str):
            if team not in agg:
                agg[team] = {
                    "Team": team,
                    "Tot": 0,
                    "Punti": 0,
                    "Half": 0,
                    "Err": 0,
                    "Pos": 0,
                    "Esc": 0,
                    "Neg": 0,
                }
            return agg[team]

        def serve_type(c6: str) -> str:
            return c6[3:5] if c6 and len(c6) >= 5 else ""

        def serve_sign_local(c6: str) -> str:
            return c6[5] if c6 and len(c6) >= 6 else ""

        for r in rows:
            ta = fix_team(r.get("team_a") or "")
            tb = fix_team(r.get("team_b") or "")
            rallies = parse_rallies(r.get("scout_text") or "")
            for rally in rallies:
                if not rally:
                    continue
                first = rally[0]
                if not is_serve(first):
                    continue

                stype = serve_type(first)
                if stype not in allowed_types:
                    continue

                # serving team
                if first.startswith("*"):
                    team = ta
                elif first.startswith("a"):
                    team = tb
                else:
                    continue

                rec = ensure(team)
                rec["Tot"] += 1

                sgn = serve_sign_local(first)
                if sgn == "#":
                    rec["Punti"] += 1
                elif sgn == "/":
                    rec["Half"] += 1
                elif sgn == "=":
                    rec["Err"] += 1
                elif sgn == "+":
                    rec["Pos"] += 1
                elif sgn == "!":
                    rec["Esc"] += 1
                elif sgn == "-":
                    rec["Neg"] += 1

        df = pd.DataFrame(list(agg.values()))
        if df.empty or df["Tot"].sum() == 0:
            st.info("Nessuna battuta trovata per i filtri selezionati.")
            return

        # percentuali su Tot
        def pct(num, den):
            return (100.0 * num / den) if den else 0.0

        df["Punti%"] = df.apply(lambda r: pct(r["Punti"], r["Tot"]), axis=1)
        df["Half%"]  = df.apply(lambda r: pct(r["Half"],  r["Tot"]), axis=1)
        df["Err%"]   = df.apply(lambda r: pct(r["Err"],   r["Tot"]), axis=1)
        df["Pos%"]   = df.apply(lambda r: pct(r["Pos"],   r["Tot"]), axis=1)
        df["Esc%"]   = df.apply(lambda r: pct(r["Esc"],   r["Tot"]), axis=1)
        df["Neg%"]   = df.apply(lambda r: pct(r["Neg"],   r["Tot"]), axis=1)

        # Eff = (Punti + Half*0.8 + Pos*0.45 + Esc*0.3 + Neg*0.15 - Err)/Tot
        # Nota: qui Punti/Half/... sono conteggi (non %)
        df["EFF"] = df.apply(
            lambda r: (
                (
                    r["Punti"]
                    + r["Half"] * 0.8
                    + r["Pos"] * 0.45
                    + r["Esc"] * 0.3
                    + r["Neg"] * 0.15
                    - r["Err"]
                )
                / r["Tot"]
                * 100.0
            ) if r["Tot"] else 0.0,
            axis=1
        )

        # Ordina per eff decrescente
        df = df.sort_values(by=["EFF", "Tot"], ascending=[False, False]).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))

        out = df[[
            "Rank", "Team", "Tot", "EFF",
            "Punti%", "Half%", "Err%", "Pos%", "Esc%", "Neg%"
        ]].rename(columns={
            "Rank": "Rank",
            "Team": "Team",
            "Tot": "Tot",
            "EFF": "EFF",
            "Punti%": "Punti",
            "Half%": "½ P",
            "Err%": "Err.",
            "Pos%": "Pos",
            "Esc%": "Esc",
            "Neg%": "Neg",
        }).copy()

        # stile: Perugia + evidenzia EFF
        def highlight_perugia(row):
            is_perugia = "perugia" in str(row["Team"]).lower()
            style = "background-color: #fff3cd; font-weight: 800;" if is_perugia else ""
            return [style] * len(row)

        styled = (
            out.style
              .apply(highlight_perugia, axis=1)
              .set_properties(subset=["EFF"], **{"background-color": "#e7f5ff", "font-weight": "900"})
              .format({
                  "Rank": "{:.0f}",
                  "Tot": "{:.0f}",
                  "EFF": "{:.1f}",
                  "Punti": "{:.1f}",
                  "½ P": "{:.1f}",
                  "Err.": "{:.1f}",
                  "Pos": "{:.1f}",
                  "Esc": "{:.1f}",
                  "Neg": "{:.1f}",
              })
              .set_table_styles([
                  {"selector": "th", "props": [("font-size", "22px"), ("text-align", "left"), ("padding", "8px 10px")]},
                  {"selector": "td", "props": [("font-size", "21px"), ("padding", "8px 10px")]},
              ])
        )

        st.dataframe(styled, width="stretch", hide_index=True)


    elif voce == "Ricezione":
        st.subheader("Ricezione")

        # ===== FILTRI TIPO BATTUTA AVVERSARIA (SQ / SM) =====
        cbt1, cbt2 = st.columns(2)
        with cbt1:
            use_spin = st.checkbox("Ricezione SPIN", value=True, key="fund_rec_spin")
        with cbt2:
            use_float = st.checkbox("Ricezione FLOAT", value=True, key="fund_rec_float")

        # Regola: entrambe -> tutte; una sola -> solo quella; nessuna -> tutte (fallback)
        if not use_spin and not use_float:
            use_spin = True
            use_float = True

        allowed_types = set()
        if use_spin:
            allowed_types.add("SQ")
        if use_float:
            allowed_types.add("SM")

        st.caption(
            "L’efficienza della ricezione è calcolata in questo modo: "
            "(Ok*0,77 + Escl*0,55 + Neg*0,38 – Mez*0,8 - Err) / Tot * 100. "
            "Dove i coefficienti sono le % medie di Side Out con quel tipo di ricezione del campionato."
        )

        # ===== MATCHES NEL RANGE =====
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
            st.info("Nessun match nel range selezionato.")
            return

        def parse_rallies(scout_text: str):
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

        def serve_type(first: str) -> str:
            return first[3:5] if first and len(first) >= 5 else ""

        def first_reception(rally: list[str], prefix: str) -> str | None:
            # prima ricezione (RQ/RM) della squadra che riceve
            for c in rally:
                if len(c) >= 6 and c[0] == prefix and c[3:5] in ("RQ", "RM"):
                    return c
            return None

        agg = {}  # team -> counts

        def ensure(team: str):
            if team not in agg:
                agg[team] = {
                    "Team": team,
                    "Tot": 0,
                    "Perf": 0,   # '#'
                    "Pos": 0,    # '+'
                    "Escl": 0,   # '!'
                    "Neg": 0,    # '-'
                    "Mez": 0,    # '/'
                    "Err": 0,    # '='
                }
            return agg[team]

        for r in rows:
            ta = fix_team_name(r.get("team_a") or "")
            tb = fix_team_name(r.get("team_b") or "")

            rallies = parse_rallies(r.get("scout_text") or "")
            for rally in rallies:
                if not rally:
                    continue
                first = rally[0]
                if not is_serve(first):
                    continue

                stype = serve_type(first)
                if stype not in allowed_types:
                    continue

                home_served = first.startswith("*")
                away_served = first.startswith("a")

                if home_served:
                    recv_team = tb
                    recv_prefix = "a"
                elif away_served:
                    recv_team = ta
                    recv_prefix = "*"
                else:
                    continue

                rece = first_reception(rally, recv_prefix)
                if not rece:
                    continue

                sign = rece[5]
                rec = ensure(recv_team)
                rec["Tot"] += 1

                if sign == "#":
                    rec["Perf"] += 1
                elif sign == "+":
                    rec["Pos"] += 1
                elif sign == "!":
                    rec["Escl"] += 1
                elif sign == "-":
                    rec["Neg"] += 1
                elif sign == "/":
                    rec["Mez"] += 1
                elif sign == "=":
                    rec["Err"] += 1
                else:
                    # altri segni non conteggiati (eventuali)
                    pass

        df = pd.DataFrame(list(agg.values()))
        if df.empty or df["Tot"].sum() == 0:
            st.info("Nessuna ricezione trovata per i filtri selezionati.")
            return

        def pct(num, den):
            return (100.0 * num / den) if den else 0.0

        df["Perf%"] = df.apply(lambda r: pct(r["Perf"], r["Tot"]), axis=1)
        df["Pos%"]  = df.apply(lambda r: pct(r["Pos"],  r["Tot"]), axis=1)
        df["Ok%"]   = df["Perf%"] + df["Pos%"]
        df["Escl%"] = df.apply(lambda r: pct(r["Escl"], r["Tot"]), axis=1)
        df["Neg%"]  = df.apply(lambda r: pct(r["Neg"],  r["Tot"]), axis=1)
        df["Mez%"]  = df.apply(lambda r: pct(r["Mez"],  r["Tot"]), axis=1)
        df["Err%"]  = df.apply(lambda r: pct(r["Err"],  r["Tot"]), axis=1)

        # Eff = (Ok*0.77 + Escl*0.55 + Neg*0.38 – Mez*0.8 - Err)/Tot*100
        # Qui usiamo conteggi (Ok_cnt etc.) e poi *100 per renderlo percentuale
        df["Ok_cnt"] = df["Perf"] + df["Pos"]

        df["EFF"] = df.apply(
            lambda r: (
                (
                    r["Ok_cnt"] * 0.77
                    + r["Escl"] * 0.55
                    + r["Neg"] * 0.38
                    - r["Mez"] * 0.8
                    - r["Err"]
                )
                / r["Tot"]
                * 100.0
            ) if r["Tot"] else 0.0,
            axis=1
        )

        df = df.sort_values(by=["EFF", "Tot"], ascending=[False, False]).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))

        out = df[[
            "Rank", "Team", "Tot", "EFF",
            "Perf%", "Pos%", "Ok%", "Escl%", "Neg%", "Mez%", "Err%"
        ]].rename(columns={
            "Rank": "Rank",
            "Team": "Team",
            "Tot": "Tot",
            "EFF": "EFF",
            "Perf%": "Perf",
            "Pos%": "Pos",
            "Ok%": "OK",
            "Escl%": "Escl",
            "Neg%": "Neg",
            "Mez%": "Mez",
            "Err%": "Err",
        }).copy()

        def highlight_perugia(row):
            is_perugia = "perugia" in str(row["Team"]).lower()
            style = "background-color: #fff3cd; font-weight: 800;" if is_perugia else ""
            return [style] * len(row)

        styled = (
            out.style
              .apply(highlight_perugia, axis=1)
              .set_properties(subset=["EFF"], **{"background-color": "#e7f5ff", "font-weight": "900"})
              .format({
                  "Rank": "{:.0f}",
                  "Tot": "{:.0f}",
                  "EFF": "{:.1f}",
                  "Perf": "{:.1f}",
                  "Pos": "{:.1f}",
                  "OK": "{:.1f}",
                  "Escl": "{:.1f}",
                  "Neg": "{:.1f}",
                  "Mez": "{:.1f}",
                  "Err": "{:.1f}",
              })
              .set_table_styles([
                  {"selector": "th", "props": [("font-size", "22px"), ("text-align", "left"), ("padding", "8px 10px")]},
                  {"selector": "td", "props": [("font-size", "21px"), ("padding", "8px 10px")]},
              ])
        )

        st.dataframe(styled, width="stretch", hide_index=True)


    elif voce == "Attacco":
        st.subheader("Attacco")

        copt1, copt2 = st.columns(2)
        with copt1:
            use_after_recv = st.checkbox("Attacco dopo Ricezione", value=True, key="fund_att_so")
        with copt2:
            use_transition = st.checkbox("Attacco di Transizione", value=True, key="fund_att_tr")

        # Regola: entrambe -> tutti; una sola -> solo quel tipo; nessuna -> tutti (fallback)
        if not use_after_recv and not use_transition:
            use_after_recv = True
            use_transition = True

        st.caption("L’efficienza dell’attacco è calcolata in questo modo: (Punti – Murate - Errori) / Tot * 100.")

        # ===== MATCHES NEL RANGE =====
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
            st.info("Nessun match nel range selezionato.")
            return

        def parse_rallies(scout_text: str):
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

        def first_attack_idx_after_serve(rally: list[str]) -> int | None:
            # primo attacco della rally (home o away) dopo la battuta
            for i, c in enumerate(rally[1:], start=1):
                if len(c) >= 6 and c[3] == "A" and c[0] in ("*", "a"):
                    return i
            return None

        def is_attack_code(c: str) -> bool:
            return len(c) >= 6 and c[3] == "A" and c[0] in ("*", "a")

        def attack_sign(c: str) -> str:
            return c[5] if c and len(c) >= 6 else ""

        agg = {}
        def ensure(team: str):
            if team not in agg:
                agg[team] = {
                    "Team": team,
                    "Tot": 0,
                    "Punti": 0,
                    "Pos": 0,
                    "Escl": 0,
                    "Neg": 0,
                    "Mur": 0,  # '/'
                    "Err": 0,  # '='
                }
            return agg[team]

        for r in rows:
            ta = fix_team_name(r.get("team_a") or "")
            tb = fix_team_name(r.get("team_b") or "")

            rallies = parse_rallies(r.get("scout_text") or "")
            for rally in rallies:
                if not rally:
                    continue
                if not is_serve(rally[0]):
                    continue

                fa = first_attack_idx_after_serve(rally)
                if fa is None:
                    continue

                for i in range(fa, len(rally)):
                    c = rally[i]
                    if not is_attack_code(c):
                        continue

                    is_first_attack = (i == fa)

                    if is_first_attack and not use_after_recv:
                        continue
                    if (not is_first_attack) and not use_transition:
                        continue

                    # team side by prefix
                    team = ta if c[0] == "*" else tb
                    rec = ensure(team)
                    rec["Tot"] += 1

                    s = attack_sign(c)
                    if s == "#":
                        rec["Punti"] += 1
                    elif s == "+":
                        rec["Pos"] += 1
                    elif s == "!":
                        rec["Escl"] += 1
                    elif s == "-":
                        rec["Neg"] += 1
                    elif s == "/":
                        rec["Mur"] += 1
                    elif s == "=":
                        rec["Err"] += 1

        df = pd.DataFrame(list(agg.values()))
        if df.empty or df["Tot"].sum() == 0:
            st.info("Nessun attacco trovato per i filtri selezionati.")
            return

        def pct(num, den):
            return (100.0 * num / den) if den else 0.0

        df["Punti%"] = df.apply(lambda r: pct(r["Punti"], r["Tot"]), axis=1)
        df["Pos%"]   = df.apply(lambda r: pct(r["Pos"],   r["Tot"]), axis=1)
        df["Escl%"]  = df.apply(lambda r: pct(r["Escl"],  r["Tot"]), axis=1)
        df["Neg%"]   = df.apply(lambda r: pct(r["Neg"],   r["Tot"]), axis=1)
        df["Mur%"]   = df.apply(lambda r: pct(r["Mur"],   r["Tot"]), axis=1)
        df["Err%"]   = df.apply(lambda r: pct(r["Err"],   r["Tot"]), axis=1)
        df["KO%"]    = df["Mur%"] + df["Err%"]

        # Eff = (Punti - KO)/Tot*100 -> using counts (Punti - (Mur+Err))/Tot*100
        df["EFF"] = df.apply(
            lambda r: ((r["Punti"] - (r["Mur"] + r["Err"])) / r["Tot"] * 100.0) if r["Tot"] else 0.0,
            axis=1
        )

        df = df.sort_values(by=["EFF", "Tot"], ascending=[False, False]).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))

        out = df[[
            "Rank", "Team", "Tot", "EFF",
            "Punti%", "Pos%", "Escl%", "Neg%", "Mur%", "Err%", "KO%"
        ]].rename(columns={
            "Rank": "Rank",
            "Team": "Team",
            "Tot": "Tot",
            "EFF": "Eff",
            "Punti%": "Punti",
            "Pos%": "Pos",
            "Escl%": "Escl",
            "Neg%": "Neg",
            "Mur%": "Mur",
            "Err%": "Err",
            "KO%": "KO",
        }).copy()

        def highlight_perugia(row):
            is_perugia = "perugia" in str(row["Team"]).lower()
            style = "background-color: #fff3cd; font-weight: 800;" if is_perugia else ""
            return [style] * len(row)

        styled = (
            out.style
              .apply(highlight_perugia, axis=1)
              .set_properties(subset=["Eff"], **{"background-color": "#e7f5ff", "font-weight": "900"})
              .format({
                  "Rank": "{:.0f}",
                  "Tot": "{:.0f}",
                  "Eff": "{:.1f}",
                  "Punti": "{:.1f}",
                  "Pos": "{:.1f}",
                  "Escl": "{:.1f}",
                  "Neg": "{:.1f}",
                  "Mur": "{:.1f}",
                  "Err": "{:.1f}",
                  "KO": "{:.1f}",
              })
              .set_table_styles([
                  {"selector": "th", "props": [("font-size", "22px"), ("text-align", "left"), ("padding", "8px 10px")]},
                  {"selector": "td", "props": [("font-size", "21px"), ("padding", "8px 10px")]},
              ])
        )

        st.dataframe(styled, width="stretch", hide_index=True)


    elif voce == "Muro":
        st.subheader("Muro")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            opt_neg = st.checkbox("Muro dopo Battuta negativa", value=True, key="fund_blk_neg")
        with c2:
            opt_exc = st.checkbox("Muro dopo Battuta Esclamativa", value=True, key="fund_blk_exc")
        with c3:
            opt_pos = st.checkbox("Muro dopo Battuta Positiva", value=True, key="fund_blk_pos")
        with c4:
            opt_tr  = st.checkbox("Muro di transizione", value=True, key="fund_blk_tr")

        # Regola: se nessuna spuntata -> tutte (fallback)
        if not (opt_neg or opt_exc or opt_pos or opt_tr):
            opt_neg = opt_exc = opt_pos = opt_tr = True

        st.caption(
            "L’efficienza del muro è calcolata in questo modo: "
            "(Vincenti*2 + Positivi*0,7 + Negativi*0,07 + Coperte*0,15 - Invasioni - ManiOut) / Tot * 100."
        )

        # ===== MATCHES NEL RANGE =====
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
            st.info("Nessun match nel range selezionato.")
            return

        def parse_rallies(scout_text: str):
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
            return c6[5] if c6 and len(c6) >= 6 else ""

        def is_attack(c6: str) -> bool:
            return len(c6) >= 6 and c6[3] == "A" and c6[0] in ("*", "a")

        def first_attack_idx(rally: list[str], attacker_prefix: str) -> int | None:
            for i in range(1, len(rally)):
                c = rally[i]
                if is_attack(c) and c[0] == attacker_prefix:
                    return i
            return None

        def first_block_after_idx(rally: list[str], start_i: int):
            for j in range(start_i + 1, len(rally)):
                c = rally[j]
                if len(c) >= 6 and c[3] == "B" and c[0] in ("*", "a"):
                    return j, c
            return None

        agg = {}
        def ensure(team: str):
            if team not in agg:
                agg[team] = {
                    "Team": team,
                    "Tot": 0,
                    "Perf": 0,
                    "Pos": 0,
                    "Neg": 0,
                    "Cop": 0,
                    "Inv": 0,
                    "Err": 0,
                }
            return agg[team]

        for r in rows:
            ta = fix_team_name(r.get("team_a") or "")
            tb = fix_team_name(r.get("team_b") or "")

            rallies = parse_rallies(r.get("scout_text") or "")
            for rally in rallies:
                if not rally or not is_serve(rally[0]):
                    continue

                first = rally[0]
                sgn = serve_sign(first)

                if first.startswith("*"):
                    recv_team = tb
                    recv_prefix = "a"
                    blk_team = ta
                    blk_prefix = "*"
                elif first.startswith("a"):
                    recv_team = ta
                    recv_prefix = "*"
                    blk_team = tb
                    blk_prefix = "a"
                else:
                    continue

                fa = first_attack_idx(rally, recv_prefix)
                if fa is None:
                    continue

                fb = first_block_after_idx(rally, fa)
                is_after_first_attack = False
                block_code = None
                if fb:
                    _, bc = fb
                    if bc[0] == blk_prefix:
                        is_after_first_attack = True
                        block_code = bc

                # Transizione: tutti i tocchi muro della squadra al muro che NON sono quel "primo muro dopo primo attacco"
                if opt_tr:
                    for c in rally[1:]:
                        if len(c) >= 6 and c[3] == "B" and c[0] == blk_prefix:
                            if is_after_first_attack and block_code is not None and c == block_code:
                                continue
                            rec = ensure(blk_team)
                            rec["Tot"] += 1
                            sign = c[5]
                            if sign == "#":
                                rec["Perf"] += 1
                            elif sign == "+":
                                rec["Pos"] += 1
                            elif sign == "-":
                                rec["Neg"] += 1
                            elif sign == "!":
                                rec["Cop"] += 1
                            elif sign == "/":
                                rec["Inv"] += 1
                            elif sign == "=":
                                rec["Err"] += 1

                # Dopo primo attacco, filtrato per segno battuta
                if is_after_first_attack and block_code is not None:
                    if (sgn == "-" and opt_neg) or (sgn == "!" and opt_exc) or (sgn == "+" and opt_pos):
                        rec = ensure(blk_team)
                        rec["Tot"] += 1
                        sign = block_code[5]
                        if sign == "#":
                            rec["Perf"] += 1
                        elif sign == "+":
                            rec["Pos"] += 1
                        elif sign == "-":
                            rec["Neg"] += 1
                        elif sign == "!":
                            rec["Cop"] += 1
                        elif sign == "/":
                            rec["Inv"] += 1
                        elif sign == "=":
                            rec["Err"] += 1

        df = pd.DataFrame(list(agg.values()))
        if df.empty or df["Tot"].sum() == 0:
            st.info("Nessun muro trovato per i filtri selezionati.")
            return

        def pct(num, den):
            return (100.0 * num / den) if den else 0.0

        df["Perf%"] = df.apply(lambda r: pct(r["Perf"], r["Tot"]), axis=1)
        df["Pos%"]  = df.apply(lambda r: pct(r["Pos"],  r["Tot"]), axis=1)
        df["Neg%"]  = df.apply(lambda r: pct(r["Neg"],  r["Tot"]), axis=1)
        df["Cop%"]  = df.apply(lambda r: pct(r["Cop"],  r["Tot"]), axis=1)
        df["Inv%"]  = df.apply(lambda r: pct(r["Inv"],  r["Tot"]), axis=1)
        df["Err%"]  = df.apply(lambda r: pct(r["Err"],  r["Tot"]), axis=1)

        df["EFF"] = df.apply(
            lambda r: (
                (r["Perf"] * 2.0 + r["Pos"] * 0.7 + r["Neg"] * 0.07 + r["Cop"] * 0.15 - r["Inv"] - r["Err"])
                / r["Tot"]
                * 100.0
            ) if r["Tot"] else 0.0,
            axis=1
        )

        df = df.sort_values(by=["EFF", "Tot"], ascending=[False, False]).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))

        out = df[[
            "Rank", "Team", "Tot", "EFF",
            "Perf%", "Pos%", "Neg%", "Cop%", "Inv%", "Err%"
        ]].rename(columns={
            "Rank": "Rank",
            "Team": "Team",
            "Tot": "Tot",
            "EFF": "Eff",
            "Perf%": "Perf",
            "Pos%": "Pos",
            "Neg%": "Neg",
            "Cop%": "Cop",
            "Inv%": "Inv",
            "Err%": "Err",
        }).copy()

        def highlight_perugia(row):
            is_perugia = "perugia" in str(row["Team"]).lower()
            style = "background-color: #fff3cd; font-weight: 800;" if is_perugia else ""
            return [style] * len(row)

        styled = (
            out.style
              .apply(highlight_perugia, axis=1)
              .set_properties(subset=["Eff"], **{"background-color": "#e7f5ff", "font-weight": "900"})
              .format({
                  "Rank": "{:.0f}",
                  "Tot": "{:.0f}",
                  "Eff": "{:.1f}",
                  "Perf": "{:.1f}",
                  "Pos": "{:.1f}",
                  "Neg": "{:.1f}",
                  "Cop": "{:.1f}",
                  "Inv": "{:.1f}",
                  "Err": "{:.1f}",
              })
              .set_table_styles([
                  {"selector": "th", "props": [("font-size", "22px"), ("text-align", "left"), ("padding", "8px 10px")]},
                  {"selector": "td", "props": [("font-size", "21px"), ("padding", "8px 10px")]},
              ])
        )

        st.dataframe(styled, width="stretch", hide_index=True)


    elif voce == "Difesa":
        st.subheader("Difesa")
        st.write("👉 Prossimo step: ranking squadre per difese positive e conversione a break point.")


# =========================
# UI: CLASSIFICHE FONDAMENTALI - GIOCATORI (per ruolo)
# =========================
def render_fondamentali_players():
    st.header("Classifiche Fondamentali - Giocatori (per ruolo)")

    fondamentale = st.radio(
        "Seleziona fondamentale",
        ["Battuta", "Ricezione", "Attacco", "Muro", "Difesa"],
        index=0,
        key="fund_pl_fond",
    )

    # ===== RANGE GIORNATE =====
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
        from_round = st.number_input("Da giornata", min_value=min_r, max_value=max_r, value=min_r, step=1, key="fund_pl_from")
    with c2:
        to_round = st.number_input("A giornata", min_value=min_r, max_value=max_r, value=max_r, step=1, key="fund_pl_to")

    if from_round > to_round:
        st.error("Range non valido: 'Da giornata' deve essere <= 'A giornata'.")
        st.stop()

    # ===== ROSTER / RUOLI =====
    season = st.text_input("Stagione roster", value="2025-26", key="fund_pl_season")

    with engine.begin() as conn:
        roster_rows = conn.execute(text("""
            SELECT team_raw, team_norm, jersey_number, player_name, role, created_at
            FROM roster
            WHERE season = :season
        """), {"season": season}).mappings().all()

    if not roster_rows:
        st.warning("Roster vuoto per questa stagione: importa prima i ruoli (pagina Import Ruoli).")
        return

    df_roster = pd.DataFrame(roster_rows)
    df_roster["created_at"] = df_roster["created_at"].fillna("")
    df_roster = (
        df_roster.sort_values(by=["team_norm", "jersey_number", "created_at"])
                 .drop_duplicates(subset=["team_norm", "jersey_number"], keep="last")
    )

    roles_all = sorted(df_roster["role"].dropna().unique().tolist())
    roles_sel = st.multiselect(
        "Filtra per ruolo (selezione multipla)",
        options=roles_all,
        default=roles_all,
        key="fund_pl_roles",
    )

    min_hits = st.number_input(
        "Numero minimo di colpi (sotto questo valore il giocatore non viene mostrato)",
        min_value=0,
        max_value=1000,
        value=10,
        step=1,
        key="fund_pl_min_hits",
    )

    # ===== MATCHES NEL RANGE =====
    with engine.begin() as conn:
        matches = conn.execute(text("""
            SELECT team_a, team_b, scout_text
            FROM matches
            WHERE round_number BETWEEN :from_round AND :to_round
        """), {"from_round": int(from_round), "to_round": int(to_round)}).mappings().all()

    if not matches:
        st.info("Nessun match nel range selezionato.")
        return

    def pct(num, den):
        return (100.0 * num / den) if den else 0.0

    def highlight_perugia(row):
        is_perugia = "perugia" in str(row.get("Squadra", "")).lower()
        style = "background-color: #fff3cd; font-weight: 800;" if is_perugia else ""
        return [style] * len(row)

    # =======================
    # BATTUTA
    # =======================
    if fondamentale == "Battuta":
        cb1, cb2 = st.columns(2)
        with cb1:
            use_spin = st.checkbox("Battuta SPIN", value=True, key="fund_pl_srv_spin")
        with cb2:
            use_float = st.checkbox("Battuta FLOAT", value=True, key="fund_pl_srv_float")

        if not use_spin and not use_float:
            use_spin = True
            use_float = True

        allowed_types = set()
        if use_spin:
            allowed_types.add("SQ")
        if use_float:
            allowed_types.add("SM")

        st.caption(
            "L’efficienza della battuta è calcolata in questo modo: "
            "(Punti + ½ P.*0,8 + Pos.*0,45 + Esc*0,3 + Neg*0,15 – Err) / Tot * 100. "
            "Dove i coefficienti sono le % medie di B.Point con quel tipo di ricezione del campionato."
        )

        agg = {}

        def ensure(team_raw: str, num: int):
            tnorm = team_norm(team_raw)
            key = (tnorm, num)
            if key not in agg:
                agg[key] = {
                    "team_norm": tnorm,
                    "Squadra": team_raw,
                    "N°": num,
                    "Tot": 0,
                    "Punti": 0,
                    "Half": 0,
                    "Err": 0,
                    "Pos": 0,
                    "Esc": 0,
                    "Neg": 0,
                }
            return agg[key]

        def serve_type(c6: str) -> str:
            return c6[3:5] if c6 and len(c6) >= 5 else ""

        def serve_sign_local(c6: str) -> str:
            return c6[5] if c6 and len(c6) >= 6 else ""

        for m in matches:
            ta = fix_team_name(m.get("team_a") or "")
            tb = fix_team_name(m.get("team_b") or "")
            scout_text = m.get("scout_text") or ""
            if not scout_text:
                continue

            for raw in str(scout_text).splitlines():
                raw = raw.strip()
                if not raw or raw[0] not in ("*", "a"):
                    continue
                c6 = code6(raw)
                if not c6 or not is_serve(c6):
                    continue

                stype = serve_type(c6)
                if stype not in allowed_types:
                    continue

                team = ta if c6[0] == "*" else tb

                num = serve_player_number(c6)
                if num is None:
                    continue

                rec = ensure(team, num)
                rec["Tot"] += 1

                sgn = serve_sign_local(c6)
                if sgn == "#":
                    rec["Punti"] += 1
                elif sgn == "/":
                    rec["Half"] += 1
                elif sgn == "=":
                    rec["Err"] += 1
                elif sgn == "+":
                    rec["Pos"] += 1
                elif sgn == "!":
                    rec["Esc"] += 1
                elif sgn == "-":
                    rec["Neg"] += 1

        df = pd.DataFrame(list(agg.values()))
        if df.empty:
            st.info("Nessuna battuta trovata per i filtri selezionati.")
            return

        df = df.merge(
            df_roster[["team_norm", "jersey_number", "player_name", "role", "team_raw"]],
            left_on=["team_norm", "N°"],
            right_on=["team_norm", "jersey_number"],
            how="left",
        ).drop(columns=["jersey_number"])

        df.rename(columns={"player_name": "Nome giocatore", "role": "Ruolo"}, inplace=True)
        df["Nome giocatore"] = df["Nome giocatore"].fillna(df["N°"].apply(lambda x: f"N°{int(x):02d}"))
        df["Ruolo"] = df["Ruolo"].fillna("(non in roster)")
        df["Squadra"] = df["team_raw"].fillna(df["Squadra"])
        df = df.drop(columns=["team_raw"])

        if roles_sel:
            df = df[df["Ruolo"].isin(roles_sel)].copy()

        df = df[df["Tot"] >= int(min_hits)].copy()
        if df.empty:
            st.info("Nessun giocatore supera il filtro del numero minimo di colpi.")
            return

        df["Punti%"] = df.apply(lambda r: pct(r["Punti"], r["Tot"]), axis=1)
        df["Half%"]  = df.apply(lambda r: pct(r["Half"],  r["Tot"]), axis=1)
        df["Err%"]   = df.apply(lambda r: pct(r["Err"],   r["Tot"]), axis=1)
        df["Pos%"]   = df.apply(lambda r: pct(r["Pos"],   r["Tot"]), axis=1)
        df["Esc%"]   = df.apply(lambda r: pct(r["Esc"],   r["Tot"]), axis=1)
        df["Neg%"]   = df.apply(lambda r: pct(r["Neg"],   r["Tot"]), axis=1)

        df["EFF"] = df.apply(
            lambda r: (
                (r["Punti"] + r["Half"] * 0.8 + r["Pos"] * 0.45 + r["Esc"] * 0.3 + r["Neg"] * 0.15 - r["Err"])
                / r["Tot"]
                * 100.0
            ) if r["Tot"] else 0.0,
            axis=1
        )

        df = df.sort_values(by=["EFF", "Tot"], ascending=[False, False]).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))

        out = df[[
            "Rank", "Nome giocatore", "Squadra", "Tot", "EFF",
            "Punti%", "Half%", "Err%", "Pos%", "Esc%", "Neg%"
        ]].rename(columns={
            "Tot": "Tot",
            "EFF": "EFF",
            "Punti%": "Punti",
            "Half%": "½ P",
            "Err%": "Err.",
            "Pos%": "Pos",
            "Esc%": "Esc",
            "Neg%": "Neg",
        }).copy()

        styled = (
            out.style
              .apply(highlight_perugia, axis=1)
              .set_properties(subset=["EFF"], **{"background-color": "#e7f5ff", "font-weight": "900"})
              .format({
                  "Rank": "{:.0f}",
                  "Tot": "{:.0f}",
                  "EFF": "{:.1f}",
                  "Punti": "{:.1f}",
                  "½ P": "{:.1f}",
                  "Err.": "{:.1f}",
                  "Pos": "{:.1f}",
                  "Esc": "{:.1f}",
                  "Neg": "{:.1f}",
              })
              .set_table_styles([
                  {"selector": "th", "props": [("font-size", "22px"), ("text-align", "left"), ("padding", "8px 10px")]},
                  {"selector": "td", "props": [("font-size", "21px"), ("padding", "8px 10px")]},
              ])
        )
        st.dataframe(styled, width="stretch", hide_index=True)
        return

    # =======================
    # RICEZIONE
    # =======================
    if fondamentale == "Ricezione":
        cb1, cb2 = st.columns(2)
        with cb1:
            use_spin = st.checkbox("Ricezione SPIN", value=True, key="fund_pl_rec_spin")
        with cb2:
            use_float = st.checkbox("Ricezione FLOAT", value=True, key="fund_pl_rec_float")

        if not use_spin and not use_float:
            use_spin = True
            use_float = True

        allowed_types = set()
        if use_spin:
            allowed_types.add("SQ")
        if use_float:
            allowed_types.add("SM")

        st.caption(
            "L’efficienza della ricezione è calcolata in questo modo: "
            "(Ok*0,77 + Escl*0,55 + Neg*0,38 – Mez*0,8 - Err) / Tot * 100. "
            "Dove i coefficienti sono le % medie di Side Out con quel tipo di ricezione del campionato."
        )

        def parse_rallies(scout_text: str):
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

        def serve_type(first: str) -> str:
            return first[3:5] if first and len(first) >= 5 else ""

        def first_reception(rally: list[str], prefix: str):
            for c in rally:
                if len(c) >= 6 and c[0] == prefix and c[3:5] in ("RQ", "RM"):
                    return c
            return None

        agg = {}

        def ensure(team_raw: str, num: int):
            tnorm = team_norm(team_raw)
            key = (tnorm, num)
            if key not in agg:
                agg[key] = {
                    "team_norm": tnorm,
                    "Squadra": team_raw,
                    "N°": num,
                    "Tot": 0,
                    "Perf": 0,
                    "Pos": 0,
                    "Escl": 0,
                    "Neg": 0,
                    "Mez": 0,
                    "Err": 0,
                }
            return agg[key]

        for m in matches:
            ta = fix_team_name(m.get("team_a") or "")
            tb = fix_team_name(m.get("team_b") or "")
            rallies = parse_rallies(m.get("scout_text") or "")
            for rally in rallies:
                if not rally or not is_serve(rally[0]):
                    continue
                first = rally[0]
                stype = serve_type(first)
                if stype not in allowed_types:
                    continue

                if first.startswith("*"):
                    recv_team = tb
                    recv_prefix = "a"
                else:
                    recv_team = ta
                    recv_prefix = "*"

                rece = first_reception(rally, recv_prefix)
                if not rece:
                    continue

                num = serve_player_number(rece)
                if num is None:
                    continue

                sign = rece[5]
                rec = ensure(recv_team, num)
                rec["Tot"] += 1
                if sign == "#":
                    rec["Perf"] += 1
                elif sign == "+":
                    rec["Pos"] += 1
                elif sign == "!":
                    rec["Escl"] += 1
                elif sign == "-":
                    rec["Neg"] += 1
                elif sign == "/":
                    rec["Mez"] += 1
                elif sign == "=":
                    rec["Err"] += 1

        df = pd.DataFrame(list(agg.values()))
        if df.empty:
            st.info("Nessuna ricezione trovata per i filtri selezionati.")
            return

        df = df.merge(
            df_roster[["team_norm", "jersey_number", "player_name", "role", "team_raw"]],
            left_on=["team_norm", "N°"],
            right_on=["team_norm", "jersey_number"],
            how="left",
        ).drop(columns=["jersey_number"])

        df.rename(columns={"player_name": "Nome giocatore", "role": "Ruolo"}, inplace=True)
        df["Nome giocatore"] = df["Nome giocatore"].fillna(df["N°"].apply(lambda x: f"N°{int(x):02d}"))
        df["Ruolo"] = df["Ruolo"].fillna("(non in roster)")
        df["Squadra"] = df["team_raw"].fillna(df["Squadra"])
        df = df.drop(columns=["team_raw"])

        if roles_sel:
            df = df[df["Ruolo"].isin(roles_sel)].copy()

        df = df[df["Tot"] >= int(min_hits)].copy()
        if df.empty:
            st.info("Nessun giocatore supera il filtro del numero minimo di colpi.")
            return

        df["Perf%"] = df.apply(lambda r: pct(r["Perf"], r["Tot"]), axis=1)
        df["Pos%"]  = df.apply(lambda r: pct(r["Pos"],  r["Tot"]), axis=1)
        df["OK%"]   = df["Perf%"] + df["Pos%"]
        df["Escl%"] = df.apply(lambda r: pct(r["Escl"], r["Tot"]), axis=1)
        df["Neg%"]  = df.apply(lambda r: pct(r["Neg"],  r["Tot"]), axis=1)
        df["Mez%"]  = df.apply(lambda r: pct(r["Mez"],  r["Tot"]), axis=1)
        df["Err%"]  = df.apply(lambda r: pct(r["Err"],  r["Tot"]), axis=1)

        ok_cnt = df["Perf"] + df["Pos"]
        df["EFF"] = ((ok_cnt * 0.77 + df["Escl"] * 0.55 + df["Neg"] * 0.38 - df["Mez"] * 0.8 - df["Err"]) / df["Tot"]) * 100.0

        df = df.sort_values(by=["EFF", "Tot"], ascending=[False, False]).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))

        out = df[[
            "Rank", "Nome giocatore", "Squadra", "Tot", "EFF",
            "Perf%", "Pos%", "OK%", "Escl%", "Neg%", "Mez%", "Err%"
        ]].rename(columns={
            "Tot": "Tot",
            "EFF": "Eff",
            "Perf%": "Perf",
            "Pos%": "Pos",
            "OK%": "OK",
            "Escl%": "Escl",
            "Neg%": "Neg",
            "Mez%": "Mez",
            "Err%": "Err",
        }).copy()

        styled = (
            out.style
              .apply(highlight_perugia, axis=1)
              .set_properties(subset=["Eff"], **{"background-color": "#e7f5ff", "font-weight": "900"})
              .format({
                  "Rank": "{:.0f}",
                  "Tot": "{:.0f}",
                  "Eff": "{:.1f}",
                  "Perf": "{:.1f}",
                  "Pos": "{:.1f}",
                  "OK": "{:.1f}",
                  "Escl": "{:.1f}",
                  "Neg": "{:.1f}",
                  "Mez": "{:.1f}",
                  "Err": "{:.1f}",
              })
              .set_table_styles([
                  {"selector": "th", "props": [("font-size", "22px"), ("text-align", "left"), ("padding", "8px 10px")]},
                  {"selector": "td", "props": [("font-size", "21px"), ("padding", "8px 10px")]},
              ])
        )
        st.dataframe(styled, width="stretch", hide_index=True)
        return

    
    # =======================
    # ATTACCO
    # =======================
    if fondamentale == "Attacco":
        copt1, copt2 = st.columns(2)
        with copt1:
            use_after_recv = st.checkbox("Attacco dopo Ricezione", value=True, key="fund_pl_att_so")
        with copt2:
            use_transition = st.checkbox("Attacco di Transizione", value=True, key="fund_pl_att_tr")

        if not use_after_recv and not use_transition:
            use_after_recv = True
            use_transition = True

        st.caption("L’efficienza dell’attacco è calcolata in questo modo: (Punti – Murate - Errori) / Tot * 100.")

        def parse_rallies(scout_text: str):
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

        def is_attack_code(c: str) -> bool:
            return len(c) >= 6 and c[3] == "A" and c[0] in ("*", "a")

        def attack_sign(c: str) -> str:
            return c[5] if c and len(c) >= 6 else ""

        def first_attack_idx_after_serve(rally: list[str]) -> int | None:
            for i, c in enumerate(rally[1:], start=1):
                if is_attack_code(c):
                    return i
            return None

        agg = {}

        def ensure(team_raw: str, num: int):
            tnorm = team_norm(team_raw)
            key = (tnorm, num)
            if key not in agg:
                agg[key] = {
                    "team_norm": tnorm,
                    "Squadra": team_raw,
                    "N°": num,
                    "Tot": 0,
                    "Punti": 0,
                    "Pos": 0,
                    "Escl": 0,
                    "Neg": 0,
                    "Mur": 0,
                    "Err": 0,
                }
            return agg[key]

        for m in matches:
            ta = fix_team_name(m.get("team_a") or "")
            tb = fix_team_name(m.get("team_b") or "")
            rallies = parse_rallies(m.get("scout_text") or "")

            for rally in rallies:
                if not rally or not is_serve(rally[0]):
                    continue

                fa = first_attack_idx_after_serve(rally)
                if fa is None:
                    continue

                for i in range(fa, len(rally)):
                    c = rally[i]
                    if not is_attack_code(c):
                        continue

                    is_first = (i == fa)
                    if is_first and not use_after_recv:
                        continue
                    if (not is_first) and not use_transition:
                        continue

                    team_raw = ta if c[0] == "*" else tb
                    num = serve_player_number(c)
                    if num is None:
                        continue

                    rec = ensure(team_raw, num)
                    rec["Tot"] += 1
                    s = attack_sign(c)
                    if s == "#":
                        rec["Punti"] += 1
                    elif s == "+":
                        rec["Pos"] += 1
                    elif s == "!":
                        rec["Escl"] += 1
                    elif s == "-":
                        rec["Neg"] += 1
                    elif s == "/":
                        rec["Mur"] += 1
                    elif s == "=":
                        rec["Err"] += 1

        df = pd.DataFrame(list(agg.values()))
        if df.empty:
            st.info("Nessun attacco trovato per i filtri selezionati.")
            return

        df = df.merge(
            df_roster[["team_norm", "jersey_number", "player_name", "role", "team_raw"]],
            left_on=["team_norm", "N°"],
            right_on=["team_norm", "jersey_number"],
            how="left",
        ).drop(columns=["jersey_number"])

        df.rename(columns={"player_name": "Nome giocatore", "role": "Ruolo"}, inplace=True)
        df["Nome giocatore"] = df["Nome giocatore"].fillna(df["N°"].apply(lambda x: f"N°{int(x):02d}"))
        df["Ruolo"] = df["Ruolo"].fillna("(non in roster)")
        df["Squadra"] = df["team_raw"].fillna(df["Squadra"])
        df = df.drop(columns=["team_raw"])

        if roles_sel:
            df = df[df["Ruolo"].isin(roles_sel)].copy()

        df = df[df["Tot"] >= int(min_hits)].copy()
        if df.empty:
            st.info("Nessun giocatore supera il filtro del numero minimo di colpi.")
            return

        df["Punti%"] = df.apply(lambda r: pct(r["Punti"], r["Tot"]), axis=1)
        df["Pos%"]   = df.apply(lambda r: pct(r["Pos"],   r["Tot"]), axis=1)
        df["Escl%"]  = df.apply(lambda r: pct(r["Escl"],  r["Tot"]), axis=1)
        df["Neg%"]   = df.apply(lambda r: pct(r["Neg"],   r["Tot"]), axis=1)
        df["Mur%"]   = df.apply(lambda r: pct(r["Mur"],   r["Tot"]), axis=1)
        df["Err%"]   = df.apply(lambda r: pct(r["Err"],   r["Tot"]), axis=1)
        df["KO%"]    = df["Mur%"] + df["Err%"]

        df["EFF"] = df.apply(
            lambda r: ((r["Punti"] - (r["Mur"] + r["Err"])) / r["Tot"] * 100.0) if r["Tot"] else 0.0,
            axis=1
        )

        df = df.sort_values(by=["EFF", "Tot"], ascending=[False, False]).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))

        out = df[[
            "Rank", "Nome giocatore", "Squadra", "Tot", "EFF",
            "Punti%", "Pos%", "Escl%", "Neg%", "Mur%", "Err%", "KO%"
        ]].rename(columns={
            "Tot": "Tot",
            "EFF": "Eff",
            "Punti%": "Punti",
            "Pos%": "Pos",
            "Escl%": "Escl",
            "Neg%": "Neg",
            "Mur%": "Mur",
            "Err%": "Err",
            "KO%": "KO",
        }).copy()

        styled = (
            out.style
              .apply(highlight_perugia, axis=1)
              .set_properties(subset=["Eff"], **{"background-color": "#e7f5ff", "font-weight": "900"})
              .format({
                  "Rank": "{:.0f}",
                  "Tot": "{:.0f}",
                  "Eff": "{:.1f}",
                  "Punti": "{:.1f}",
                  "Pos": "{:.1f}",
                  "Escl": "{:.1f}",
                  "Neg": "{:.1f}",
                  "Mur": "{:.1f}",
                  "Err": "{:.1f}",
                  "KO": "{:.1f}",
              })
              .set_table_styles([
                  {"selector": "th", "props": [("font-size", "22px"), ("text-align", "left"), ("padding", "8px 10px")]},
                  {"selector": "td", "props": [("font-size", "21px"), ("padding", "8px 10px")]},
              ])
        )
        st.dataframe(styled, width="stretch", hide_index=True)
        return

    
    # =======================
    # MURO
    # =======================
    if fondamentale == "Muro":
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            opt_neg = st.checkbox("Muro dopo Battuta negativa", value=True, key="fund_pl_blk_neg")
        with c2:
            opt_exc = st.checkbox("Muro dopo Battuta Esclamativa", value=True, key="fund_pl_blk_exc")
        with c3:
            opt_pos = st.checkbox("Muro dopo Battuta Positiva", value=True, key="fund_pl_blk_pos")
        with c4:
            opt_tr  = st.checkbox("Muro di transizione", value=True, key="fund_pl_blk_tr")

        if not (opt_neg or opt_exc or opt_pos or opt_tr):
            opt_neg = opt_exc = opt_pos = opt_tr = True

        st.caption(
            "L’efficienza del muro è calcolata in questo modo: "
            "(Vincenti*2 + Positivi*0,7 + Negativi*0,07 + Coperte*0,15 - Invasioni - ManiOut) / Tot * 100."
        )

        def parse_rallies(scout_text: str):
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
            return c6[5] if c6 and len(c6) >= 6 else ""

        def is_attack(c6: str) -> bool:
            return len(c6) >= 6 and c6[3] == "A" and c6[0] in ("*", "a")

        def first_attack_idx(rally: list[str], attacker_prefix: str):
            for i in range(1, len(rally)):
                c = rally[i]
                if is_attack(c) and c[0] == attacker_prefix:
                    return i
            return None

        def first_block_after_idx(rally: list[str], start_i: int):
            for j in range(start_i + 1, len(rally)):
                c = rally[j]
                if len(c) >= 6 and c[3] == "B" and c[0] in ("*", "a"):
                    return j, c
            return None

        agg = {}  # (team_norm, num) -> counts

        def ensure(team_raw: str, num: int):
            tnorm = team_norm(team_raw)
            key = (tnorm, num)
            if key not in agg:
                agg[key] = {
                    "team_norm": tnorm,
                    "Squadra": team_raw,
                    "N°": num,
                    "Tot": 0,
                    "Perf": 0,
                    "Pos": 0,
                    "Neg": 0,
                    "Cop": 0,
                    "Inv": 0,
                    "Err": 0,
                }
            return agg[key]

        def add_block(team_raw: str, num: int, block_code: str):
            rec = ensure(team_raw, num)
            rec["Tot"] += 1
            sign = block_code[5] if len(block_code) >= 6 else ""
            if sign == "#":
                rec["Perf"] += 1
            elif sign == "+":
                rec["Pos"] += 1
            elif sign == "-":
                rec["Neg"] += 1
            elif sign == "!":
                rec["Cop"] += 1
            elif sign == "/":
                rec["Inv"] += 1
            elif sign == "=":
                rec["Err"] += 1

        for m in matches:
            ta = fix_team_name(m.get("team_a") or "")
            tb = fix_team_name(m.get("team_b") or "")
            rallies = parse_rallies(m.get("scout_text") or "")

            for rally in rallies:
                if not rally or not is_serve(rally[0]):
                    continue

                first = rally[0]
                sgn = serve_sign(first)

                if first.startswith("*"):
                    recv_team = tb
                    recv_prefix = "a"
                    blk_team = ta
                    blk_prefix = "*"
                else:
                    recv_team = ta
                    recv_prefix = "*"
                    blk_team = tb
                    blk_prefix = "a"

                fa = first_attack_idx(rally, recv_prefix)
                if fa is None:
                    continue

                fb = first_block_after_idx(rally, fa)
                is_after_first_attack = False
                block_code = None
                if fb:
                    _, bc = fb
                    if bc[0] == blk_prefix:
                        is_after_first_attack = True
                        block_code = bc

                # Transizione: tutti i tocchi muro della squadra al muro che NON sono quel "primo muro dopo primo attacco"
                if opt_tr:
                    for c in rally[1:]:
                        if len(c) >= 6 and c[3] == "B" and c[0] == blk_prefix:
                            if is_after_first_attack and block_code is not None and c == block_code:
                                continue
                            num = serve_player_number(c)
                            if num is None:
                                continue
                            add_block(blk_team, num, c)

                # Dopo primo attacco, filtrato per segno battuta
                if is_after_first_attack and block_code is not None:
                    if (sgn == "-" and opt_neg) or (sgn == "!" and opt_exc) or (sgn == "+" and opt_pos):
                        num = serve_player_number(block_code)
                        if num is None:
                            continue
                        add_block(blk_team, num, block_code)

        df = pd.DataFrame(list(agg.values()))
        if df.empty:
            st.info("Nessun muro trovato per i filtri selezionati.")
            return

        df = df.merge(
            df_roster[["team_norm", "jersey_number", "player_name", "role", "team_raw"]],
            left_on=["team_norm", "N°"],
            right_on=["team_norm", "jersey_number"],
            how="left",
        ).drop(columns=["jersey_number"])

        df.rename(columns={"player_name": "Nome giocatore", "role": "Ruolo"}, inplace=True)
        df["Nome giocatore"] = df["Nome giocatore"].fillna(df["N°"].apply(lambda x: f"N°{int(x):02d}"))
        df["Ruolo"] = df["Ruolo"].fillna("(non in roster)")
        df["Squadra"] = df["team_raw"].fillna(df["Squadra"])
        df = df.drop(columns=["team_raw"])

        if roles_sel:
            df = df[df["Ruolo"].isin(roles_sel)].copy()

        df = df[df["Tot"] >= int(min_hits)].copy()
        if df.empty:
            st.info("Nessun giocatore supera il filtro del numero minimo di colpi.")
            return

        df["Perf%"] = df.apply(lambda r: pct(r["Perf"], r["Tot"]), axis=1)
        df["Pos%"]  = df.apply(lambda r: pct(r["Pos"],  r["Tot"]), axis=1)
        df["Neg%"]  = df.apply(lambda r: pct(r["Neg"],  r["Tot"]), axis=1)
        df["Cop%"]  = df.apply(lambda r: pct(r["Cop"],  r["Tot"]), axis=1)
        df["Inv%"]  = df.apply(lambda r: pct(r["Inv"],  r["Tot"]), axis=1)
        df["Err%"]  = df.apply(lambda r: pct(r["Err"],  r["Tot"]), axis=1)

        df["EFF"] = df.apply(
            lambda r: (
                (r["Perf"] * 2.0 + r["Pos"] * 0.7 + r["Neg"] * 0.07 + r["Cop"] * 0.15 - r["Inv"] - r["Err"])
                / r["Tot"]
                * 100.0
            ) if r["Tot"] else 0.0,
            axis=1
        )

        df = df.sort_values(by=["EFF", "Tot"], ascending=[False, False]).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))

        out = df[[
            "Rank", "Nome giocatore", "Squadra", "Tot", "EFF",
            "Perf%", "Pos%", "Neg%", "Cop%", "Inv%", "Err%"
        ]].rename(columns={
            "Tot": "Tot",
            "EFF": "Eff",
            "Perf%": "Perf",
            "Pos%": "Pos",
            "Neg%": "Neg",
            "Cop%": "Cop",
            "Inv%": "Inv",
            "Err%": "Err",
        }).copy()

        styled = (
            out.style
              .apply(highlight_perugia, axis=1)
              .set_properties(subset=["Eff"], **{"background-color": "#e7f5ff", "font-weight": "900"})
              .format({
                  "Rank": "{:.0f}",
                  "Tot": "{:.0f}",
                  "Eff": "{:.1f}",
                  "Perf": "{:.1f}",
                  "Pos": "{:.1f}",
                  "Neg": "{:.1f}",
                  "Cop": "{:.1f}",
                  "Inv": "{:.1f}",
                  "Err": "{:.1f}",
              })
              .set_table_styles([
                  {"selector": "th", "props": [("font-size", "22px"), ("text-align", "left"), ("padding", "8px 10px")]},
                  {"selector": "td", "props": [("font-size", "21px"), ("padding", "8px 10px")]},
              ])
        )
        st.dataframe(styled, width="stretch", hide_index=True)
        return

    
    # =======================
    # DIFESA
    # =======================
    if fondamentale == "Difesa":
        # stato iniziale
        if "fund_pl_def_total" not in st.session_state:
            st.session_state.fund_pl_def_total = True
        for k in ("dt", "dq", "dm", "dh"):
            kk = f"fund_pl_def_{k}"
            if kk not in st.session_state:
                st.session_state[kk] = False

        def _toggle_total():
            if st.session_state.fund_pl_def_total:
                st.session_state.fund_pl_def_dt = False
                st.session_state.fund_pl_def_dq = False
                st.session_state.fund_pl_def_dm = False
                st.session_state.fund_pl_def_dh = False

        def _toggle_specific():
            if (st.session_state.fund_pl_def_dt or st.session_state.fund_pl_def_dq or
                st.session_state.fund_pl_def_dm or st.session_state.fund_pl_def_dh):
                st.session_state.fund_pl_def_total = False
            if (not st.session_state.fund_pl_def_total and
                not st.session_state.fund_pl_def_dt and not st.session_state.fund_pl_def_dq and
                not st.session_state.fund_pl_def_dm and not st.session_state.fund_pl_def_dh):
                st.session_state.fund_pl_def_total = True

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.checkbox("Difesa su palla Spinta", key="fund_pl_def_dt", on_change=_toggle_specific)
        with c2:
            st.checkbox("Difesa su 1°tempo", key="fund_pl_def_dq", on_change=_toggle_specific)
        with c3:
            st.checkbox("Difesa su PIPE", key="fund_pl_def_dm", on_change=_toggle_specific)
        with c4:
            st.checkbox("Difesa su H-ball", key="fund_pl_def_dh", on_change=_toggle_specific)
        with c5:
            st.checkbox("Difesa TOTALE", key="fund_pl_def_total", on_change=_toggle_total)

        st.caption(
            "L’efficienza della difesa è calcolata in questo modo: "
            "(Buone*2 + Coperture*0,5 + Negative*0,4 + OverTheNet*0,3 – Errori) / Tot * 100. "
            "È un indice motivatore."
        )

        def parse_rallies(scout_text: str):
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

        def is_defense(c6: str) -> bool:
            return len(c6) >= 6 and c6[3] == "D" and c6[0] in ("*", "a")

        def def_type(c6: str) -> str:
            return c6[3:5] if c6 and len(c6) >= 5 else ""

        def def_sign(c6: str) -> str:
            return c6[5] if c6 and len(c6) >= 6 else ""

        def defense_selected(c6: str) -> bool:
            if st.session_state.fund_pl_def_total:
                return True

            t = def_type(c6)
            s = def_sign(c6)

            if st.session_state.fund_pl_def_dt and t == "DT":
                return s != "!"
            if st.session_state.fund_pl_def_dq and t == "DQ":
                return s != "!"
            if st.session_state.fund_pl_def_dm and t == "DM":
                return s != "!"
            if st.session_state.fund_pl_def_dh and t == "DH":
                return s != "!"
            return False

        agg = {}  # (team_norm, num) -> counts

        def ensure(team_raw: str, num: int):
            tnorm = team_norm(team_raw)
            key = (tnorm, num)
            if key not in agg:
                agg[key] = {
                    "team_norm": tnorm,
                    "Squadra": team_raw,
                    "N°": num,
                    "Tot": 0,
                    "Perf": 0,  # '+'
                    "Cop": 0,   # '!'
                    "Neg": 0,   # '-'
                    "Over": 0,  # '/'
                    "Err": 0,   # '='
                }
            return agg[key]

        for m in matches:
            ta = fix_team_name(m.get("team_a") or "")
            tb = fix_team_name(m.get("team_b") or "")
            rallies = parse_rallies(m.get("scout_text") or "")
            for rally in rallies:
                if not rally:
                    continue
                for c in rally[1:]:
                    if not is_defense(c):
                        continue
                    if not defense_selected(c):
                        continue

                    team_raw = ta if c[0] == "*" else tb
                    num = serve_player_number(c)
                    if num is None:
                        continue

                    rec = ensure(team_raw, num)
                    rec["Tot"] += 1
                    s = def_sign(c)
                    if s == "+":
                        rec["Perf"] += 1
                    elif s == "!":
                        rec["Cop"] += 1
                    elif s == "-":
                        rec["Neg"] += 1
                    elif s == "/":
                        rec["Over"] += 1
                    elif s == "=":
                        rec["Err"] += 1

        df = pd.DataFrame(list(agg.values()))
        if df.empty:
            st.info("Nessuna difesa trovata per i filtri selezionati.")
            return

        df = df.merge(
            df_roster[["team_norm", "jersey_number", "player_name", "role", "team_raw"]],
            left_on=["team_norm", "N°"],
            right_on=["team_norm", "jersey_number"],
            how="left",
        ).drop(columns=["jersey_number"])

        df.rename(columns={"player_name": "Nome giocatore", "role": "Ruolo"}, inplace=True)
        df["Nome giocatore"] = df["Nome giocatore"].fillna(df["N°"].apply(lambda x: f"N°{int(x):02d}"))
        df["Ruolo"] = df["Ruolo"].fillna("(non in roster)")
        df["Squadra"] = df["team_raw"].fillna(df["Squadra"])
        df = df.drop(columns=["team_raw"])

        if roles_sel:
            df = df[df["Ruolo"].isin(roles_sel)].copy()

        df = df[df["Tot"] >= int(min_hits)].copy()
        if df.empty:
            st.info("Nessun giocatore supera il filtro del numero minimo di colpi.")
            return

        df["Perf%"] = df.apply(lambda r: pct(r["Perf"], r["Tot"]), axis=1)
        df["Cop%"]  = df.apply(lambda r: pct(r["Cop"],  r["Tot"]), axis=1)
        df["Neg%"]  = df.apply(lambda r: pct(r["Neg"],  r["Tot"]), axis=1)
        df["Over%"] = df.apply(lambda r: pct(r["Over"], r["Tot"]), axis=1)
        df["Err%"]  = df.apply(lambda r: pct(r["Err"],  r["Tot"]), axis=1)

        df["EFF"] = df.apply(
            lambda r: (
                (r["Perf"] * 2.0 + r["Cop"] * 0.5 + r["Neg"] * 0.4 + r["Over"] * 0.3 - r["Err"])
                / r["Tot"]
                * 100.0
            ) if r["Tot"] else 0.0,
            axis=1
        )

        df = df.sort_values(by=["EFF", "Tot"], ascending=[False, False]).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))

        out = df[[
            "Rank", "Nome giocatore", "Squadra", "Tot", "EFF",
            "Perf%", "Cop%", "Neg%", "Over%", "Err%"
        ]].rename(columns={
            "Tot": "Tot",
            "EFF": "Eff",
            "Perf%": "Perf",
            "Cop%": "Cop",
            "Neg%": "Neg",
            "Over%": "Over",
            "Err%": "Err",
        }).copy()

        styled = (
            out.style
              .apply(highlight_perugia, axis=1)
              .set_properties(subset=["Eff"], **{"background-color": "#e7f5ff", "font-weight": "900"})
              .format({
                  "Rank": "{:.0f}",
                  "Tot": "{:.0f}",
                  "Eff": "{:.1f}",
                  "Perf": "{:.1f}",
                  "Cop": "{:.1f}",
                  "Neg": "{:.1f}",
                  "Over": "{:.1f}",
                  "Err": "{:.1f}",
              })
              .set_table_styles([
                  {"selector": "th", "props": [("font-size", "22px"), ("text-align", "left"), ("padding", "8px 10px")]},
                  {"selector": "td", "props": [("font-size", "21px"), ("padding", "8px 10px")]},
              ])
        )
        st.dataframe(styled, width="stretch", hide_index=True)
        return

    st.info("In costruzione: per ora sono complete le tabelle Battuta e Ricezione (giocatori).")

# =========================
# UI: PUNTI PER SET (per ruolo) + fasi
# =========================
def render_points_per_set():
    st.header("Punti per Set")
    st.sidebar.caption("BUILD: PUNTI_PER_SET_V11 (POINTS/SET ALL)")

    # ===== FILTRO RANGE GIORNATE =====
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
        from_round = st.number_input("Da giornata", min_value=min_r, max_value=max_r, value=min_r, step=1, key="pps_from")
    with c2:
        to_round = st.number_input("A giornata", min_value=min_r, max_value=max_r, value=max_r, step=1, key="pps_to")

    if from_round > to_round:
        st.error("Range non valido: 'Da giornata' deve essere <= 'A giornata'.")
        st.stop()

    # ===== FILTRO RUOLI (da roster) =====
    season = st.text_input("Stagione roster", value="2025-26", key="pps_season")
    with engine.begin() as conn:
        roster_rows = conn.execute(text("""
            SELECT team_raw, team_norm, jersey_number, player_name, role, created_at
            FROM roster
            WHERE season = :season
        """), {"season": season}).mappings().all()

    if not roster_rows:
        st.warning("Roster vuoto per questa stagione: importa prima i ruoli (pagina Import Ruoli).")
        return

    df_roster = pd.DataFrame(roster_rows)
    df_roster["created_at"] = df_roster["created_at"].fillna("")
    df_roster = (
        df_roster.sort_values(by=["team_norm", "jersey_number", "created_at"])
                 .drop_duplicates(subset=["team_norm", "jersey_number"], keep="last")
    )

    roles_all = sorted(df_roster["role"].dropna().unique().tolist())
    roles_sel = st.multiselect(
        "Filtri ruoli (puoi selezionarne quanti vuoi)",
        options=roles_all,
        default=roles_all,
        key="pps_roles",
    )

    # ===== FASI (anche entrambe) =====
    cf1, cf2 = st.columns(2)
    with cf1:
        use_sideout = st.checkbox("Fase Side Out", value=True, key="pps_so")
    with cf2:
        use_break = st.checkbox("Fase Break", value=True, key="pps_bp")

    if not use_sideout and not use_break:
        use_sideout = True
        use_break = True

    st.caption("I punti considerati sono solo: **Battuta (#)**, **Attacco (#)**, **Muro (#)**.")

    # ===== MATCHES =====
    with engine.begin() as conn:
        matches = conn.execute(text("""
            SELECT id AS match_id, team_a, team_b, scout_text
            FROM matches
            WHERE round_number BETWEEN :from_round AND :to_round
        """), {"from_round": int(from_round), "to_round": int(to_round)}).mappings().all()

    if not matches:
        st.info("Nessun match nel range selezionato.")
        return

    SET_RE = re.compile(r"\*\*(\d)set\b", re.IGNORECASE)

    def parse_rallies_and_sets(scout_text: str):
        if not scout_text:
            return [], set()

        lines = [ln.strip() for ln in str(scout_text).splitlines() if ln and ln.strip()]
        current_set = None
        sets_seen = set()
        rallies = []
        current = []
        current_set_for_rally = None

        for ln in lines:
            mm = SET_RE.search(ln)
            if mm:
                if current:
                    rallies.append((current_set_for_rally, current))
                    current = []
                    current_set_for_rally = None
                current_set = int(mm.group(1))
                sets_seen.add(current_set)
                continue

            if current_set is None:
                continue

            c6 = code6(ln)
            if not c6 or c6[0] not in ("*", "a"):
                continue

            if is_serve(c6):
                if current:
                    rallies.append((current_set_for_rally, current))
                current = [c6]
                current_set_for_rally = current_set
                continue

            if not current:
                continue

            current.append(c6)

            if is_home_point(c6) or is_away_point(c6):
                rallies.append((current_set_for_rally, current))
                current = []
                current_set_for_rally = None

        if current:
            rallies.append((current_set_for_rally, current))

        return rallies, sets_seen

    def player_num_from_code(c6: str):
        mm = re.match(r"^[\*a](\d{2})", c6)
        return int(mm.group(1)) if mm else None

    def is_player_action_any(c6: str) -> bool:
        if len(c6) < 5 or c6[0] not in ("*", "a"):
            return False
        if c6[3:5] in ("SQ", "SM", "RQ", "RM"):
            return True
        if len(c6) >= 4 and c6[3] in ("A", "B", "D"):
            return True
        return False

    def point_kind(c6: str):
        if len(c6) < 6 or c6[5] != "#":
            return None
        if c6[3:5] in ("SQ", "SM"):
            return "serve"
        if c6[3] == "A":
            return "attack"
        if c6[3] == "B":
            return "block"
        return None

    def is_error_forced_point(c6: str) -> bool:
        if len(c6) < 6:
            return False
        if c6[3:5] in ("SQ", "SM") and c6[5] == "=":
            return True
        if c6[3] == "A" and c6[5] == "=":
            return True
        if c6[3] == "B" and c6[5] == "/":
            return True
        return False

    players = {}
    team_tot = {}

    def ensure_player(team_raw: str, num: int):
        tnorm = team_norm(team_raw)
        key = (tnorm, num)
        if key not in players:
            players[key] = {
                "team_norm": tnorm,
                "Nome Team": team_raw,
                "N°": num,
                "sets": set(),  # (match_id, set_no)
                "pts_total": 0,
                "pts_serve": 0,
                "pts_attack": 0,
                "pts_block": 0,
            }
        return players[key]

    def ensure_team(team_raw: str):
        tnorm = team_norm(team_raw)
        if tnorm not in team_tot:
            team_tot[tnorm] = {
                "team_norm": tnorm,
                "Nome Team": team_raw,
                "sets_total": 0,
                "pts_total": 0,
                "err_avv_total": 0,
            }
        return team_tot[tnorm]

    for m in matches:
        match_id = int(m.get("match_id") or 0)
        team_a = fix_team_name(m.get("team_a") or "")
        team_b = fix_team_name(m.get("team_b") or "")

        rallies, sets_seen = parse_rallies_and_sets(m.get("scout_text") or "")
        n_sets = len(sets_seen) if sets_seen else 0

        if n_sets:
            ensure_team(team_a)["sets_total"] += n_sets
            ensure_team(team_b)["sets_total"] += n_sets

        for set_no, rally in rallies:
            if not rally or set_no is None:
                continue
            first = rally[0]
            if not is_serve(first):
                continue

            serve_prefix = first[0]

            # set giocati giocatore: >=1 azione nel set (match_id, set_no)
            for c in rally:
                if not is_player_action_any(c):
                    continue
                num = player_num_from_code(c)
                if num is None:
                    continue
                team_raw = team_a if c[0] == "*" else team_b
                ensure_player(team_raw, num)["sets"].add((match_id, int(set_no)))

            # winner
            home_won = any(is_home_point(x) for x in rally)
            away_won = any(is_away_point(x) for x in rally)
            if home_won or away_won:
                if home_won:
                    win_prefix = "*"
                    win_team = team_a
                    lose_prefix = "a"
                else:
                    win_prefix = "a"
                    win_team = team_b
                    lose_prefix = "*"

                is_break_team = (serve_prefix == win_prefix)
                is_sideout_team = (serve_prefix != win_prefix)
                if (is_sideout_team and use_sideout) or (is_break_team and use_break):
                    if any((x[0] == lose_prefix and is_error_forced_point(x)) for x in rally):
                        ensure_team(win_team)["err_avv_total"] += 1

            # punti giocatore
            for c in rally:
                kind = point_kind(c)
                if kind is None:
                    continue
                num = player_num_from_code(c)
                if num is None:
                    continue

                player_prefix = c[0]
                is_break = (serve_prefix == player_prefix)
                is_sideout = (serve_prefix != player_prefix)

                if (is_sideout and not use_sideout) or (is_break and not use_break):
                    continue

                team_raw = team_a if player_prefix == "*" else team_b
                rec = ensure_player(team_raw, num)
                rec["pts_total"] += 1
                if kind == "serve":
                    rec["pts_serve"] += 1
                elif kind == "attack":
                    rec["pts_attack"] += 1
                elif kind == "block":
                    rec["pts_block"] += 1

                ensure_team(team_raw)["pts_total"] += 1

    df_players = pd.DataFrame(list(players.values()))
    if df_players.empty:
        st.info("Nessun dato punti trovato nel range selezionato.")
        return

    df_players = df_players.merge(
        df_roster[["team_norm", "jersey_number", "player_name", "role", "team_raw"]],
        left_on=["team_norm", "N°"],
        right_on=["team_norm", "jersey_number"],
        how="left",
    ).drop(columns=["jersey_number"])

    df_players.rename(columns={"player_name": "Nome giocatore", "role": "Ruolo"}, inplace=True)
    df_players["Nome giocatore"] = df_players["Nome giocatore"].fillna(df_players["N°"].apply(lambda x: f"N°{int(x):02d}"))
    df_players["Ruolo"] = df_players["Ruolo"].fillna("(non in roster)")
    df_players["Nome Team"] = df_players["team_raw"].fillna(df_players["Nome Team"])
    df_players = df_players.drop(columns=["team_raw"])

    if roles_sel:
        df_players = df_players[df_players["Ruolo"].isin(roles_sel)].copy()

    df_players["Set giocati dal Giocatore"] = df_players["sets"].apply(lambda s: len(s) if isinstance(s, set) else 0)

    df_team = pd.DataFrame(list(team_tot.values()))
    df_team["Punti per Set (Team)"] = df_team.apply(lambda r: (r["pts_total"] / r["sets_total"]) if r["sets_total"] else 0.0, axis=1)
    df_team["Errori Avv/Set"] = df_team.apply(lambda r: (r["err_avv_total"] / r["sets_total"]) if r["sets_total"] else 0.0, axis=1)

    df_players = df_players.merge(
        df_team[["team_norm", "sets_total", "Punti per Set (Team)", "Errori Avv/Set"]],
        on="team_norm",
        how="left",
    )

    df_players["Punti per Set (Giocatore)"] = df_players.apply(
        lambda r: (r["pts_total"] / r["Set giocati dal Giocatore"]) if r["Set giocati dal Giocatore"] else 0.0,
        axis=1
    )

    df_players["% Ply/Team"] = df_players.apply(
        lambda r: (100.0 * r["Punti per Set (Giocatore)"] / r["Punti per Set (Team)"]) if (r["Punti per Set (Team)"] and r["Punti per Set (Team)"] > 0) else 0.0,
        axis=1
    )

    df_rank = df_players.sort_values(
        by=["Punti per Set (Giocatore)", "Set giocati dal Giocatore"],
        ascending=[False, False]
    ).reset_index(drop=True)
    df_rank.insert(0, "Ranking", range(1, len(df_rank) + 1))

    out = df_rank[[
        "Ranking",
        "Nome Team",
        "sets_total",
        "Punti per Set (Team)",
        "Errori Avv/Set",
        "Nome giocatore",
        "Set giocati dal Giocatore",
        "Punti per Set (Giocatore)",
        "% Ply/Team",
        "pts_serve",
        "pts_attack",
        "pts_block",
    ]].rename(columns={
        "sets_total": "Set giocati dal Team",
        "pts_serve": "Punti in Battuta",
        "pts_attack": "Punti in Attacco",
        "pts_block": "Punti a Muro",
    }).copy()

    # --- Trasforma Punti in Battuta/Attacco/Muro in valori PER SET del giocatore ---
    denom = pd.to_numeric(out["Set giocati dal Giocatore"], errors="coerce").replace(0, pd.NA)
    for col in ["Punti in Battuta", "Punti in Attacco", "Punti a Muro"]:
        out[col] = (pd.to_numeric(out[col], errors="coerce") / denom).astype(float).fillna(0.0)

    # formatter 1 decimale senza zeri finali
    def _fmt1(x):
        try:
            if x is None:
                return ""
            v = float(x)
            s = f"{v:.1f}"
            return s.rstrip("0").rstrip(".")
        except Exception:
            return x

    def highlight_perugia(row):
        is_perugia = "perugia" in str(row.get("Nome Team", "")).lower()
        style = "background-color: #fff3cd; font-weight: 800;" if is_perugia else ""
        return [style] * len(row)

    styled = (
        out.style
          .apply(highlight_perugia, axis=1)
          .format({
              "Punti per Set (Team)": _fmt1,
              "Errori Avv/Set": _fmt1,
              "Punti per Set (Giocatore)": _fmt1,
              "% Ply/Team": _fmt1,
              "Punti in Battuta": _fmt1,
              "Punti in Attacco": _fmt1,
              "Punti a Muro": _fmt1,
          })
    )

    st.dataframe(styled, use_container_width=True, hide_index=True)


# =========================
# UI: HOME DASHBOARD (3 TAB)
# =========================
def render_home_dashboard():
    st.header("Home – Dashboard")

    # ===== RANGE GIORNATE =====
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
        from_round = st.number_input("Da giornata", min_value=min_r, max_value=max_r, value=min_r, step=1, key="home_from")
    with c2:
        to_round = st.number_input("A giornata", min_value=min_r, max_value=max_r, value=max_r, step=1, key="home_to")

    if from_round > to_round:
        st.error("Range non valido: 'Da giornata' deve essere <= 'A giornata'.")
        st.stop()

    # ===== SQUADRA (default Perugia) =====
    with engine.begin() as conn:
        teams = conn.execute(text("""
            SELECT DISTINCT team_a AS t FROM matches
            UNION
            SELECT DISTINCT team_b AS t FROM matches
        """)).mappings().all()
    team_list = sorted([fix_team_name(r["t"] or "") for r in teams if (r.get("t") or "").strip()])
    default_team = None
    for t in team_list:
        if "perugia" in t.lower():
            default_team = t
            break
    if not default_team and team_list:
        default_team = team_list[0]

    team_focus = st.selectbox("Squadra", team_list, index=(team_list.index(default_team) if default_team in team_list else 0), key="home_team")

    # ===== DATA (matches in range) =====
    with engine.begin() as conn:
        matches = conn.execute(text("""
            SELECT id AS match_id, team_a, team_b, scout_text
            FROM matches
            WHERE round_number BETWEEN :from_round AND :to_round
        """), {"from_round": int(from_round), "to_round": int(to_round)}).mappings().all()

    if not matches:
        st.info("Nessun match nel range selezionato.")
        return

    SET_RE = re.compile(r"\*\*(\d)set\b", re.IGNORECASE)

    def parse_rallies(scout_text: str):
        if not scout_text:
            return []
        lines = [ln.strip() for ln in str(scout_text).splitlines() if ln and ln.strip()]
        current_set = None
        rallies = []
        current = []

        for ln in lines:
            m = SET_RE.search(ln)
            if m:
                current_set = int(m.group(1))
                continue

            if current_set is None:
                continue

            c6 = code6(ln)
            if not c6 or c6[0] not in ("*", "a"):
                continue

            if is_serve(c6):
                if current:
                    rallies.append(current)
                current = [c6]
                continue

            if not current:
                continue
            current.append(c6)

            if is_home_point(c6) or is_away_point(c6):
                rallies.append(current)
                current = []

        if current:
            rallies.append(current)
        return rallies

    def serve_sign(c6: str) -> str:
        return c6[5] if c6 and len(c6) >= 6 else ""

    def serve_type(c6: str) -> str:
        return c6[3:5] if c6 and len(c6) >= 5 else ""

    def is_rece(c6: str) -> bool:
        return len(c6) >= 6 and c6[3:5] in ("RQ", "RM") and c6[0] in ("*", "a")

    def rece_sign(c6: str) -> str:
        return c6[5] if c6 and len(c6) >= 6 else ""

    def is_attack_code(c6: str) -> bool:
        return len(c6) >= 6 and c6[0] in ("*", "a") and c6[3] == "A"

    def is_block_code(c6: str) -> bool:
        return len(c6) >= 6 and c6[0] in ("*", "a") and c6[3] == "B"

    def is_def_code(c6: str) -> bool:
        return len(c6) >= 6 and c6[0] in ("*", "a") and c6[3] == "D"

    def team_of(prefix: str, team_a: str, team_b: str) -> str:
        return team_a if prefix == "*" else team_b

    # ===== Aggregazioni per team =====
    T = {}

    def ensure_team(team_raw: str):
        tn = team_norm(team_raw)
        if tn not in T:
            T[tn] = {
                "team_norm": tn,
                "Team": fix_team_name(team_raw),
                "sets_total": 0,

                "so_att": 0, "so_win": 0,
                "so_spin_att": 0, "so_spin_win": 0,
                "so_float_att": 0, "so_float_win": 0,
                "so_dir_win": 0,
                "so_play_att": 0, "so_play_win": 0,
                "so_good_att": 0, "so_good_win": 0,
                "so_exc_att": 0, "so_exc_win": 0,
                "so_neg_att": 0, "so_neg_win": 0,

                "bp_att": 0, "bp_win": 0,
                "bp_play_att": 0, "bp_play_win": 0,
                "bp_neg_att": 0, "bp_neg_win": 0,
                "bp_exc_att": 0, "bp_exc_win": 0,
                "bp_pos_att": 0, "bp_pos_win": 0,
                "bp_half_att": 0, "bp_half_win": 0,
                "bt_ace": 0, "bt_err": 0,

                "srv_tot": 0, "srv_hash": 0, "srv_half": 0, "srv_pos": 0, "srv_exc": 0, "srv_neg": 0, "srv_err": 0,

                "rec_tot": 0, "rec_hash": 0, "rec_pos": 0, "rec_exc": 0, "rec_neg": 0, "rec_half": 0, "rec_err": 0,

                "att_tot": 0, "att_hash": 0, "att_pos": 0, "att_exc": 0, "att_neg": 0, "att_blk": 0, "att_err": 0,
                "att_first_tot": 0, "att_first_hash": 0, "att_first_blk": 0, "att_first_err": 0,
                "att_tr_tot": 0, "att_tr_hash": 0, "att_tr_blk": 0, "att_tr_err": 0,

                "blk_tot": 0, "blk_hash": 0, "blk_pos": 0, "blk_neg": 0, "blk_cov": 0, "blk_inv": 0, "blk_err": 0,

                "def_tot": 0, "def_pos": 0, "def_cov": 0, "def_neg": 0, "def_over": 0, "def_err": 0,
            }
        return T[tn]

    # Sets per match
    for m in matches:
        ta = fix_team_name(m.get("team_a") or "")
        tb = fix_team_name(m.get("team_b") or "")
        scout_text = m.get("scout_text") or ""
        sets_seen = set(int(x) for x in SET_RE.findall(scout_text))
        n_sets = len(sets_seen)
        if n_sets:
            ensure_team(ta)["sets_total"] += n_sets
            ensure_team(tb)["sets_total"] += n_sets

    for m in matches:
        ta = fix_team_name(m.get("team_a") or "")
        tb = fix_team_name(m.get("team_b") or "")
        rallies = parse_rallies(m.get("scout_text") or "")
        for r in rallies:
            if not r or not is_serve(r[0]):
                continue
            first = r[0]
            s_prefix = first[0]
            s_team = team_of(s_prefix, ta, tb)
            rcv_prefix = "a" if s_prefix == "*" else "*"
            rcv_team = team_of(rcv_prefix, ta, tb)
            sgn = serve_sign(first)
            stype = serve_type(first)

            home_won = any(is_home_point(x) for x in r)
            away_won = any(is_away_point(x) for x in r)
            s_team_won = (home_won and s_prefix == "*") or (away_won and s_prefix == "a")
            rcv_team_won = (home_won and rcv_prefix == "*") or (away_won and rcv_prefix == "a")

            # SideOut (ricevente)
            if any(is_rece(x) and x[0] == rcv_prefix for x in r):
                t = ensure_team(rcv_team)
                t["so_att"] += 1
                if rcv_team_won:
                    t["so_win"] += 1

            if stype == "SQ":
                t = ensure_team(rcv_team)
                t["so_spin_att"] += 1
                if rcv_team_won:
                    t["so_spin_win"] += 1
            if stype == "SM":
                t = ensure_team(rcv_team)
                t["so_float_att"] += 1
                if rcv_team_won:
                    t["so_float_win"] += 1

            rece = None
            for x in r:
                if is_rece(x) and x[0] == rcv_prefix:
                    rece = x
                    break

            if rece:
                rs = rece_sign(rece)
                t = ensure_team(rcv_team)
                if rs in ("#", "+", "!", "-"):
                    t["so_play_att"] += 1
                    if rcv_team_won:
                        t["so_play_win"] += 1
                if rs in ("#", "+"):
                    t["so_good_att"] += 1
                    if rcv_team_won:
                        t["so_good_win"] += 1
                if rs == "!":
                    t["so_exc_att"] += 1
                    if rcv_team_won:
                        t["so_exc_win"] += 1
                if rs == "-":
                    t["so_neg_att"] += 1
                    if rcv_team_won:
                        t["so_neg_win"] += 1

            first_att = None
            for x in r:
                if is_attack_code(x) and x[0] == rcv_prefix:
                    first_att = x
                    break
            if first_att and len(first_att) >= 6 and first_att[5] == "#" and rcv_team_won:
                ensure_team(rcv_team)["so_dir_win"] += 1

            # Break (battitore)
            bt = ensure_team(s_team)
            bt["bp_att"] += 1
            if s_team_won:
                bt["bp_win"] += 1

            if sgn not in ("#", "="):
                bt["bp_play_att"] += 1
                if s_team_won:
                    bt["bp_play_win"] += 1

            if sgn == "-":
                bt["bp_neg_att"] += 1
                if s_team_won:
                    bt["bp_neg_win"] += 1
            if sgn == "!":
                bt["bp_exc_att"] += 1
                if s_team_won:
                    bt["bp_exc_win"] += 1
            if sgn == "+":
                bt["bp_pos_att"] += 1
                if s_team_won:
                    bt["bp_pos_win"] += 1
            if sgn == "/":
                bt["bp_half_att"] += 1
                if s_team_won:
                    bt["bp_half_win"] += 1

            if sgn == "#":
                bt["bt_ace"] += 1
            if sgn == "=":
                bt["bt_err"] += 1

            # Serve distribution
            bt["srv_tot"] += 1
            if sgn == "#":
                bt["srv_hash"] += 1
            elif sgn == "/":
                bt["srv_half"] += 1
            elif sgn == "+":
                bt["srv_pos"] += 1
            elif sgn == "!":
                bt["srv_exc"] += 1
            elif sgn == "-":
                bt["srv_neg"] += 1
            elif sgn == "=":
                bt["srv_err"] += 1

            # Reception distribution
            if rece:
                rt = ensure_team(rcv_team)
                rt["rec_tot"] += 1
                rs = rece_sign(rece)
                if rs == "#":
                    rt["rec_hash"] += 1
                elif rs == "+":
                    rt["rec_pos"] += 1
                elif rs == "!":
                    rt["rec_exc"] += 1
                elif rs == "-":
                    rt["rec_neg"] += 1
                elif rs == "/":
                    rt["rec_half"] += 1
                elif rs == "=":
                    rt["rec_err"] += 1

            # Attack + first vs transition
            first_attack_index = None
            for i, x in enumerate(r[1:], start=1):
                if is_attack_code(x):
                    first_attack_index = i
                    break

            for i, x in enumerate(r[1:], start=1):
                if not is_attack_code(x):
                    continue
                at = ensure_team(team_of(x[0], ta, tb))
                at["att_tot"] += 1
                sign = x[5]
                if sign == "#":
                    at["att_hash"] += 1
                elif sign == "+":
                    at["att_pos"] += 1
                elif sign == "!":
                    at["att_exc"] += 1
                elif sign == "-":
                    at["att_neg"] += 1
                elif sign == "/":
                    at["att_blk"] += 1
                elif sign == "=":
                    at["att_err"] += 1

                if first_attack_index is not None:
                    if i == first_attack_index:
                        at["att_first_tot"] += 1
                        if sign == "#":
                            at["att_first_hash"] += 1
                        if sign == "/":
                            at["att_first_blk"] += 1
                        if sign == "=":
                            at["att_first_err"] += 1
                    elif i > first_attack_index:
                        at["att_tr_tot"] += 1
                        if sign == "#":
                            at["att_tr_hash"] += 1
                        if sign == "/":
                            at["att_tr_blk"] += 1
                        if sign == "=":
                            at["att_tr_err"] += 1

            # Block
            for x in r[1:]:
                if not is_block_code(x):
                    continue
                bl = ensure_team(team_of(x[0], ta, tb))
                bl["blk_tot"] += 1
                sign = x[5]
                if sign == "#":
                    bl["blk_hash"] += 1
                elif sign == "+":
                    bl["blk_pos"] += 1
                elif sign == "-":
                    bl["blk_neg"] += 1
                elif sign == "!":
                    bl["blk_cov"] += 1
                elif sign == "/":
                    bl["blk_inv"] += 1
                elif sign == "=":
                    bl["blk_err"] += 1

            # Defense
            for x in r[1:]:
                if not is_def_code(x):
                    continue
                df = ensure_team(team_of(x[0], ta, tb))
                df["def_tot"] += 1
                sign = x[5]
                if sign == "+":
                    df["def_pos"] += 1
                elif sign == "!":
                    df["def_cov"] += 1
                elif sign == "-":
                    df["def_neg"] += 1
                elif sign == "/":
                    df["def_over"] += 1
                elif sign == "=":
                    df["def_err"] += 1

    dfT = pd.DataFrame(list(T.values()))
    if dfT.empty:
        st.info("Nessun dato nel range selezionato.")
        return

    def _pct(num, den):
        return (100.0 * num / den) if den else 0.0

    def build_df(value_series: pd.Series, higher_is_better: bool = True):
        df = pd.DataFrame({"Team": dfT["Team"], "Value": value_series})
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce").fillna(0.0)
        df = df.sort_values(by="Value", ascending=not higher_is_better).reset_index(drop=True)
        df["Rank"] = range(1, len(df) + 1)
        df.attrs["higher_is_better"] = higher_is_better
        return df

    def render_card(title: str, df: pd.DataFrame, fmt: str = "{:.1f}", higher_is_better: bool | None = None):
        # higher_is_better: se None, prova a leggere da df.attrs
        hib = higher_is_better
        if hib is None:
            hib = bool(df.attrs.get("higher_is_better", True))

        # trova riga team
        team_row = None
        for _, r in df.iterrows():
            if norm(r["Team"]) == norm(team_focus):
                team_row = r
                break
            if "perugia" in norm(team_focus) and "perugia" in norm(r["Team"]):
                team_row = r
                break
        if team_row is None:
            team_row = df.iloc[0] if not df.empty else None

        val = float(team_row["Value"]) if team_row is not None else 0.0
        rk = int(team_row["Rank"]) if team_row is not None else 0

        # delta dal 3° posto
        third_val = float(df.iloc[2]["Value"]) if len(df) >= 3 else float(df.iloc[-1]["Value"])
        # per metrica "più alto è meglio": delta = val - third
        # per "più basso è meglio": delta = third - val (positivo = sei davanti, negativo = dietro)
        delta = (val - third_val) if hib else (third_val - val)

        # colore in base al rank
        if rk and rk <= 3:
            color = "#2f9e44"   # green
        elif rk and rk <= 6:
            color = "#f08c00"   # orange
        else:
            color = "#495057"   # gray

        st.markdown(f"**{title}**")

        # valore + rank + delta (stile 'da panchina')
        delta_txt = f"{delta:+.1f} vs 3°"
        # se 'lower is better' e delta==inf (es. Err/Pti), gestisci
        if not (delta == delta and abs(delta) != float("inf")):
            delta_txt = "—"

        st.markdown(
            f"""
            <div style="display:flex; align-items:baseline; justify-content:space-between; gap:12px; padding:8px 10px; border:1px solid #e9ecef; border-radius:12px;">
              <div>
                <div style="font-size:34px; font-weight:900; color:{color}; line-height:1;">{fmt.format(val)}</div>
                <div style="font-size:14px; color:#868e96; margin-top:2px;">{team_focus} • Rank {rk}/12</div>
              </div>
              <div style="text-align:right;">
                <div style="font-size:14px; color:#868e96;">gap podio</div>
                <div style="font-size:18px; font-weight:800; color:#343a40;">{delta_txt}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        top3 = df.head(3).copy()
        top3["Value"] = top3["Value"].astype(float).round(1)
        st.dataframe(top3[["Rank", "Team", "Value"]], use_container_width=True, hide_index=True)

    tab_so, tab_bp, tab_eff = st.tabs(["SIDE OUT", "BREAK", "EFFICIENZE"])

    with tab_so:
        cols = st.columns(3)
        metrics = [
            ("Side Out TOTALE", build_df(dfT.apply(lambda r: _pct(r["so_win"], r["so_att"]), axis=1))),
            ("Side Out SPIN", build_df(dfT.apply(lambda r: _pct(r["so_spin_win"], r["so_spin_att"]), axis=1))),
            ("Side Out FLOAT", build_df(dfT.apply(lambda r: _pct(r["so_float_win"], r["so_float_att"]), axis=1))),
            ("Side Out DIRETTO", build_df(dfT.apply(lambda r: _pct(r["so_dir_win"], r["so_att"]), axis=1))),
            ("Side Out GIOCATO", build_df(dfT.apply(lambda r: _pct(r["so_play_win"], r["so_play_att"]), axis=1))),
            ("Side Out con RICE BUONA", build_df(dfT.apply(lambda r: _pct(r["so_good_win"], r["so_good_att"]), axis=1))),
            ("Side Out con RICE ESCLAMATIVA", build_df(dfT.apply(lambda r: _pct(r["so_exc_win"], r["so_exc_att"]), axis=1))),
            ("Side Out con RICE NEGATIVA", build_df(dfT.apply(lambda r: _pct(r["so_neg_win"], r["so_neg_att"]), axis=1))),
        ]
        for i, (title, dfm) in enumerate(metrics):
            with cols[i % 3]:
                render_card(title, dfm)

    with tab_bp:
        cols = st.columns(3)

        def ratio_err_ace(r):
            ace = r["bt_ace"]
            err = r["bt_err"]
            return (err / ace) if ace else float("inf")

        metrics = [
            ("BREAK TOTALE", build_df(dfT.apply(lambda r: _pct(r["bp_win"], r["bp_att"]), axis=1))),
            ("BREAK GIOCATO", build_df(dfT.apply(lambda r: _pct(r["bp_play_win"], r["bp_play_att"]), axis=1))),
            ("BREAK con BT. NEGATIVA", build_df(dfT.apply(lambda r: _pct(r["bp_neg_win"], r["bp_neg_att"]), axis=1))),
            ("BREAK con BT. ESCLAMATIVA", build_df(dfT.apply(lambda r: _pct(r["bp_exc_win"], r["bp_exc_att"]), axis=1))),
            ("BREAK con BT. POSITIVA", build_df(dfT.apply(lambda r: _pct(r["bp_pos_win"], r["bp_pos_att"]), axis=1))),
            ("BREAK con BT. 1/2 PUNTO", build_df(dfT.apply(lambda r: _pct(r["bp_half_win"], r["bp_half_att"]), axis=1))),
            ("BT punto/errore/ratio (Err/Pti)", build_df(dfT.apply(lambda r: ratio_err_ace(r), axis=1), higher_is_better=False)),
        ]

        for i, (title, dfm) in enumerate(metrics):
            with cols[i % 3]:
                if "Err/Pti" in title:
                    render_card(title, dfm, fmt="{:.2f}", higher_is_better=False)
                else:
                    render_card(title, dfm)

    with tab_eff:
        st.subheader("Filtri Efficienze")
        cflag = st.columns(3)
        with cflag[0]:
            st.checkbox("Battuta SPIN", value=True, key="home_srv_spin")
            st.checkbox("Battuta FLOAT", value=True, key="home_srv_float")
        with cflag[1]:
            st.checkbox("Ricezione SPIN", value=True, key="home_rec_spin")
            st.checkbox("Ricezione FLOAT", value=True, key="home_rec_float")
        with cflag[2]:
            att_first = st.checkbox("Attacco dopo Ricezione", value=True, key="home_att_first")
            att_tr = st.checkbox("Attacco di Transizione", value=True, key="home_att_tr")

        if not att_first and not att_tr:
            att_first = att_tr = True

        def eff_serve(r):
            tot = r["srv_tot"]
            if not tot:
                return 0.0
            return ((r["srv_hash"] + r["srv_half"]*0.8 + r["srv_pos"]*0.45 + r["srv_exc"]*0.3 + r["srv_neg"]*0.15 - r["srv_err"]) / tot) * 100.0

        def eff_rece(r):
            tot = r["rec_tot"]
            if not tot:
                return 0.0
            ok = r["rec_hash"] + r["rec_pos"]
            return ((ok*0.77 + r["rec_exc"]*0.55 + r["rec_neg"]*0.38 - r["rec_half"]*0.8 - r["rec_err"]) / tot) * 100.0

        def eff_att_total(r):
            tot = r["att_tot"]
            if not tot:
                return 0.0
            ko = r["att_blk"] + r["att_err"]
            return ((r["att_hash"] - ko) / tot) * 100.0

        def eff_att_first(r):
            tot = r["att_first_tot"]
            if not tot:
                return 0.0
            ko = r["att_first_blk"] + r["att_first_err"]
            return ((r["att_first_hash"] - ko) / tot) * 100.0

        def eff_att_tr(r):
            tot = r["att_tr_tot"]
            if not tot:
                return 0.0
            ko = r["att_tr_blk"] + r["att_tr_err"]
            return ((r["att_tr_hash"] - ko) / tot) * 100.0

        def eff_block(r):
            tot = r["blk_tot"]
            if not tot:
                return 0.0
            return ((r["blk_hash"]*2 + r["blk_pos"]*0.7 + r["blk_neg"]*0.07 + r["blk_cov"]*0.15 - r["blk_inv"] - r["blk_err"]) / tot) * 100.0

        def eff_def(r):
            tot = r["def_tot"]
            if not tot:
                return 0.0
            return ((r["def_pos"]*2 + r["def_cov"]*0.5 + r["def_neg"]*0.4 + r["def_over"]*0.3 - r["def_err"]) / tot) * 100.0

        cols = st.columns(3)
        eff_cards = [
            ("Efficienza BATTUTA", build_df(dfT.apply(lambda r: eff_serve(r), axis=1))),
            ("Efficienza RICEZIONE", build_df(dfT.apply(lambda r: eff_rece(r), axis=1))),
        ]

        if att_first and att_tr:
            eff_cards.append(("Efficienza ATTACCO (Tot)", build_df(dfT.apply(lambda r: eff_att_total(r), axis=1))))
        elif att_first:
            eff_cards.append(("Efficienza ATTACCO (Dopo Ricezione)", build_df(dfT.apply(lambda r: eff_att_first(r), axis=1))))
        else:
            eff_cards.append(("Efficienza ATTACCO (Transizione)", build_df(dfT.apply(lambda r: eff_att_tr(r), axis=1))))

        eff_cards += [
            ("Efficienza MURO TOTALE", build_df(dfT.apply(lambda r: eff_block(r), axis=1))),
            ("Efficienza DIFESA TOTALE", build_df(dfT.apply(lambda r: eff_def(r), axis=1))),
        ]

        for i, (title, dfm) in enumerate(eff_cards):
            with cols[i % 3]:
                render_card(title, dfm, fmt="{:.1f}")


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
    render_home_dashboard()

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
    render_break_players_by_role()


elif page == "Classifiche Fondamentali - Squadre":
    render_fondamentali_team()

elif page == "Classifiche Fondamentali - Giocatori (per ruolo)":
    render_fondamentali_players()

elif page == "Punti per Set":
    render_points_per_set()

else:
    st.header(page)
    st.info("In costruzione.")
