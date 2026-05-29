"""
main.py - Entry point for the Aviator Score Analyzer & Predictor.

Run this script to launch the desktop GUI application.
"""

import sys
import os

# Ensure the project root is on sys.path for clean imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui import create_gui


def main() -> None:
    """
    Application entry point.
    Initializes the database and launches the Tkinter GUI.
    """
    from database import setup_db

    # Ensure the database and tables exist
    setup_db()

    # Launch the GUI
    create_gui()


if __name__ == "__main__":
    main()