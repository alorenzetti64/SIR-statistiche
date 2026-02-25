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
    # requisito: codice giornata intorno al 13° carattere (proviamo index 12 e 13)
    for start in (12, 13):
        if len(filename) >= start + 3:
            code = filename[start : start + 3]
            if re.match(r"^[AR]\d{2}$", code):
                return code[0], int(code[1:3])

    # fallback: cerca pattern in tutto il nome (se qualcuno rinomina male)
    m = re.search(r"([AR]\d{2})", filename)
    if m:
        code = m.group(1)
        return code[0], int(code[1:3])

    raise ValueError("Codice giornata non trovato nel filename (atteso A01 / R06).")


def parse_dvw_minimal(dvw_text: str) -> dict:
    """
    Estrae solo season, competition, team_a, team_b.
    ASSUNZIONE coerente col tuo uso attuale: nel [3TEAMS] la prima è la squadra con codici '*'
    (casa) e la seconda è la squadra con codici 'a' (ospite).
    """
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

    return {
        "season": season,
        "competition": competition,
        "team_a": team_a,
        "team_b": team_b,
    }


def extract_scout_lines(dvw_text: str) -> list[str]:
    """
    Prende le righe DOPO [3SCOUT] fino alla prossima sezione [..]
    e tiene solo righe che iniziano con '*' o 'a' (dopo strip caratteri controllo).
    """
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

        # rimuovi eventuali caratteri di controllo iniziali
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
    # *24RQ# / *16RM! ecc.
    return len(c6) >= 5 and c6[0] == "*" and c6[3:5] in ("RQ", "RM")


def is_away_rece(c6: str) -> bool:
    # a24RQ# / a16RM! ecc.
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
    # *14SM- / a01SQ! ecc.
    return len(c6) >= 5 and c6[0] in ("*", "a") and c6[3:5] in ("SQ", "SM")


def is_home_point(c6: str) -> bool:
    return c6.startswith("*p")


def is_away_point(c6: str) -> bool:
    return c6.startswith("ap")


def pct(wins: int, attempts: int) -> float:
    return (wins / attempts * 100.0) if attempts else 0.0


def compute_counts_from_scout(scout_lines: list[str]) -> dict:
    # segmenta rally: da battuta (SQ/SM) a codice punto (*p / ap)
    rallies: list[list[str]] = []
    current: list[str] = []

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
            continue  # ignora righe finché non arriva la prima battuta

        current.append(c)

        if is_home_point(c) or is_away_point(c):
            rallies.append(current)
            current = []

    # conteggi
    so_home_attempts = so_home_wins = 0
    so_away_attempts = so_away_wins = 0

    bp_home_attempts = bp_home_wins = 0
    bp_away_attempts = bp_away_wins = 0

    so_spin_home_attempts = so_spin_home_wins = 0
    so_spin_away_attempts = so_spin_away_wins = 0

    so_float_home_attempts = so_float_home_wins = 0
    so_float_away_attempts = so_float_away_wins = 0

    for r in rallies:
        first = r[0]
        home_served = first.startswith("*")
        away_served = first.startswith("a")

        home_point = any(is_home_point(x) for x in r)
        away_point = any(is_away_point(x) for x in r)

        home_rece = any(is_home_rece(x) for x in r)
        away_rece = any(is_away_rece(x) for x in r)

        home_spin = any(is_home_spin(x) for x in r)
        away_spin = any(is_away_spin(x) for x in r)

        home_float = any(is_home_float(x) for x in r)
        away_float = any(is_away_float(x) for x in r)

        # SideOut totale
        if home_rece:
            so_home_attempts += 1
            if home_point:
                so_home_wins += 1

        if away_rece:
            so_away_attempts += 1
            if away_point:
                so_away_wins += 1

        # SideOut SPIN (RQ)
        if home_spin:
            so_spin_home_attempts += 1
            if home_point:
                so_spin_home_wins += 1

        if away_spin:
            so_spin_away_attempts += 1
            if away_point:
                so_spin_away_wins += 1

        # SideOut FLOAT (RM)
        if home_float:
            so_float_home_attempts += 1
            if home_point:
                so_float_home_wins += 1

        if away_float:
            so_float_away_attempts += 1
            if away_point:
                so_float_away_wins += 1

        # Break
        if home_served:
            bp_home_attempts += 1
            if home_point:
                bp_home_wins += 1

        if away_served:
            bp_away_attempts += 1
            if away_point:
                bp_away_wins += 1

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
                so_float_away_wins INTEGER
            )
        """))

        # migration: aggiungi colonne se mancanti
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
        ]
        for col, coltype in cols_to_add:
            try:
                conn.execute(text(f"ALTER TABLE matches ADD COLUMN {col} {coltype}"))
            except Exception:
                pass


# =========================
# UI: IMPORT + DELETE
# =========================
def show_last_imports(limit: int = 20):
    st.subheader("Ultimi import")
    with engine.begin() as conn:
        rows = conn.execute(text(f"""
            SELECT id, created_at, filename, phase, round_number, team_a, team_b, competition,
                   so_home_attempts, so_home_wins, so_away_attempts, so_away_wins,
                   sideout_home_pct, sideout_away_pct,
                   so_spin_home_attempts, so_spin_home_wins, so_spin_away_attempts, so_spin_away_wins,
                   so_float_home_attempts, so_float_home_wins, so_float_away_attempts, so_float_away_wins
            FROM matches
            ORDER BY id DESC
            LIMIT {limit}
        """)).mappings().all()

    df = pd.DataFrame(rows)
    if df.empty:
        st.info("Nessun import ancora.")
        return
    st.dataframe(df, width="stretch", hide_index=True)


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

    # Mostra sempre gli ultimi import
    show_last_imports(limit=20)

    # Pannello DELETE sempre visibile
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
         so_float_home_attempts, so_float_home_wins, so_float_away_attempts, so_float_away_wins
        )
        VALUES
        (:filename, :phase, :round_number, :season, :competition, :team_a, :team_b,
         :n_azioni, :preview, :scout_text, :match_key, :created_at,

         :so_home_attempts, :so_home_wins, :so_away_attempts, :so_away_wins,
         :sideout_home_pct, :sideout_away_pct,

         :bp_home_attempts, :bp_home_wins, :bp_away_attempts, :bp_away_wins,
         :break_home_pct, :break_away_pct,

         :so_spin_home_attempts, :so_spin_home_wins, :so_spin_away_attempts, :so_spin_away_wins,
         :so_float_home_attempts, :so_float_home_wins, :so_float_away_attempts, :so_float_away_wins
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
            so_float_away_wins = excluded.so_float_away_wins
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

                        **{k: int(v) if isinstance(v, bool) is False and isinstance(v, int) else v for k, v in counts.items()},
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
# UI: SIDEOUT TEAM
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

        df = df[["squadra", "% S.O.", "n° ricezioni", "n° Side Out"]].copy()
        df.insert(0, "Rank", range(1, len(df) + 1))

        styled = (
            df.style
              .apply(highlight_perugia, axis=1)
              .format({
                  "Rank": "{:.0f}",
                  "% S.O.": "{:.1f}",
                  "n° ricezioni": "{:.0f}",
                  "n° Side Out": "{:.0f}",
              })
              .set_table_styles([
                  {"selector": "th", "props": [("font-size", "24px"), ("text-align", "left"), ("padding", "10px 12px")]},
                  {"selector": "td", "props": [("font-size", "23px"), ("padding", "10px 12px")]},
              ])
        )

        st.dataframe(styled, width="stretch", hide_index=True)

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

        styled = (
            df_spin.style
              .apply(highlight_perugia, axis=1)
              .format({
                  "Rank": "{:.0f}",
                  "% S.O. SPIN": "{:.1f}",
                  "n° ricezioni SPIN": "{:.0f}",
                  "n° Side Out SPIN": "{:.0f}",
                  "% SPIN su TOT": "{:.1f}",
              })
              .set_table_styles([
                  {"selector": "th", "props": [("font-size", "24px"), ("text-align", "left"), ("padding", "10px 12px")]},
                  {"selector": "td", "props": [("font-size", "23px"), ("padding", "10px 12px")]},
              ])
        )

        st.dataframe(styled, width="stretch", hide_index=True)

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

        styled = (
            df_float.style
              .apply(highlight_perugia, axis=1)
              .format({
                  "Rank": "{:.0f}",
                  "% S.O. FLOAT": "{:.1f}",
                  "n° ricezioni FLOAT": "{:.0f}",
                  "n° Side Out FLOAT": "{:.0f}",
                  "% FLOAT su TOT": "{:.1f}",
              })
              .set_table_styles([
                  {"selector": "th", "props": [("font-size", "24px"), ("text-align", "left"), ("padding", "10px 12px")]},
                  {"selector": "td", "props": [("font-size", "23px"), ("padding", "10px 12px")]},
              ])
        )

        st.dataframe(styled, width="stretch", hide_index=True)

    else:
        st.info(f"In costruzione: **{voce}**")


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
        "Indici Side Out - Squadre",
        "Indici Break Point - Squadre",
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

elif page == "Indici Side Out - Squadre":
    render_sideout_team()

elif page == "Indici Break Point - Squadre":
    st.header("Indici Break Point - Squadre")
    st.info("In costruzione.")

elif page == "Indici Side Out - Giocatori (per ruolo)":
    st.header("Indici Side Out - Giocatori (per ruolo)")
    st.info("In costruzione.")

elif page == "Indici Break Point - Giocatori (per ruolo)":
    st.header("Indici Break Point - Giocatori (per ruolo)")
    st.info("In costruzione.")

elif page == "Classifiche Fondamentali - Squadre":
    st.header("Classifiche Fondamentali - Squadre")
    st.info("In costruzione.")

elif page == "Classifiche Fondamentali - Giocatori (per ruolo)":
    st.header("Classifiche Fondamentali - Giocatori (per ruolo)")
    st.info("In costruzione.")

elif page == "Punti per Set":
    st.header("Punti per Set")
    st.info("In costruzione.")