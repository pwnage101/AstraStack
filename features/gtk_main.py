"""Runs each scenario's GTK main loop, on the process's main thread when the launcher asks for it.

macOS only allows GUI work on the main thread, so scripts/run_behave_main_thread.py moves behave onto a worker thread
and calls serve() to hand the main thread to GTK. Plain `behave` keeps running GTK on a separate thread.
"""
import queue
import traceback
from threading import Event, Thread
from typing import Callable
from gi import require_version
require_version('Gtk', '3.0')
# Importing Gtk initializes it, so this module must first be imported from the main thread when serve() is used.
from gi.repository import Gtk

_tasks: "queue.Queue[Callable[[], None]]" = queue.Queue()
_serving = False

def start(task: Callable[[], None]) -> Event:
    """Runs a task that builds the UI and then runs the GTK main loop.

    Args:
        task: Function that builds the UI and ends by calling Gtk.main().

    Returns:
        An event that is set once the task has returned.
    """
    done = Event()
    def run() -> None:
        try:
            task()
        finally:
            done.set()
    if(_serving):
        _tasks.put(run)
    else:
        Thread(target=run, args=()).start()
    return done

def serve(target: Callable[[], int]) -> int:
    """Runs target on a worker thread while the main thread runs the tasks given to start().

    Args:
        target: Function that runs the behave suite and returns its exit status.

    Returns:
        The exit status returned by target, or 1 if it raised.
    """
    global _serving
    _serving = True
    status = []
    worker = Thread(target=lambda: status.append(target()), args=())
    worker.start()
    while(worker.is_alive()):
        try:
            task = _tasks.get(timeout=0.1)
        except queue.Empty:
            continue
        try:
            task()
        except Exception:
            traceback.print_exc()
    return status[0] if status else 1
