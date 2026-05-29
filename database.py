"""
database.py - SQLite database operations for the Score Analyzer.

Handles creation, insertion, retrieval, and deletion of game score history.
"""

import sqlite3
import os
from datetime import datetime
from typing import List, Tuple, Optional

# --- Configuration ---
DB_NAME = "score_analyzer.db"
DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, DB_NAME)


def get_connection() -> sqlite3.Connection:
    """Create and return a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Access columns by name
    return conn


def setup_db() -> None:
    """
    Initialize the database and create the 'game_history' table if it
    doesn't already exist.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS game_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_input_text TEXT NOT NULL,
            multiplier_score REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def insert_scores(raw_text: str, scores: List[float]) -> None:
    """
    Insert a list of multiplier scores into the database.

    Parameters
    ----------
    raw_text : str
        The original raw text pasted by the user.
    scores : List[float]
        The parsed list of multiplier scores to insert.
    """
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat(timespec="seconds")
    rows = [(raw_text, score, now) for score in scores]
    cursor.executemany(
        "INSERT INTO game_history (raw_input_text, multiplier_score, timestamp) "
        "VALUES (?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def fetch_all_scores() -> List[float]:
    """
    Retrieve all multiplier scores from the database, ordered by id (ascending).

    Returns
    -------
    List[float]
        A list of all multiplier scores in chronological order.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT multiplier_score FROM game_history ORDER BY id ASC"
    )
    rows = cursor.fetchall()
    conn.close()
    return [row["multiplier_score"] for row in rows]


def fetch_recent_scores(limit: int = 10) -> List[Tuple[int, float, str]]:
    """
    Retrieve the most recent scores from the database.

    Parameters
    ----------
    limit : int
        Number of recent records to fetch (default: 10).

    Returns
    -------
    List[Tuple[int, float, str]]
        A list of tuples: (id, multiplier_score, timestamp).
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, multiplier_score, timestamp FROM game_history "
        "ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [(row["id"], row["multiplier_score"], row["timestamp"]) for row in rows]


def get_total_count() -> int:
    """Return the total number of score records in the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as cnt FROM game_history")
    count = cursor.fetchone()["cnt"]
    conn.close()
    return count


def clear_database() -> None:
    """Delete all records from the game_history table."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM game_history")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    # Quick test when run directly
    setup_db()
    print(f"Database initialized at: {DB_PATH}")
    print(f"Total records: {get_total_count()}")