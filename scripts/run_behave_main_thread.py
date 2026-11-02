"""Runs the behave suite with GTK on the main thread, which macOS requires.

Usage: python scripts/run_behave_main_thread.py [behave args...]
"""
import os
import sys

# Spawned worker processes re-import this file, so only the parent may start behave.
if(__name__ == "__main__"):
    sys.path.insert(0, os.getcwd())
    from features import gtk_main
    from behave.__main__ import main
    sys.exit(gtk_main.serve(main))
