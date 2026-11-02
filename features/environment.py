import os
import sys
import shutil
import tempfile
from itertools import count
from threading import Event
from behave.model import Feature, Scenario
from behave.runner import Context
from gi import require_version
require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib
sys.path.insert(0, os.getcwd())

from UI import UI
from Globals import g
from features import gtk_main

CPUS = os.environ.get("ASTRASTACK_TEST_CPUS")
SHARDS = int(os.environ.get("ASTRASTACK_TEST_SHARDS", 1))
SHARD_INDEX = int(os.environ.get("ASTRASTACK_TEST_SHARD_INDEX", 0))
OUTPUTS = os.environ.get("ASTRASTACK_TEST_OUTPUTS")
GTK_TIMEOUT = 30

def before_all(context: Context) -> None:
    if(OUTPUTS is not None):
        os.makedirs(OUTPUTS, exist_ok=True)
        context.testOutputs = os.path.abspath(OUTPUTS)
    else:
        context.testOutputs = tempfile.mkdtemp(prefix="astrastack-test-")
    context.scenarioIndex = count()
    if(sys.platform == "darwin" and not gtk_main.isServing()):
        raise RuntimeError("On macOS, run the suite with `python scripts/run_behave_main_thread.py` instead of `behave`")

def before_feature(context: Context, feature: Feature) -> None:
    for scenario in feature.walk_scenarios():
        index = next(context.scenarioIndex)
        if(index % SHARDS != SHARD_INDEX):
            scenario.skip()

def before_scenario(context: Context, scenario: Scenario) -> None:
    g.TESTING = True
    mapped = Event()
    def run() -> None:
        g.ui = UI()
        g.ui.builder.get_object("wavelets").set_expanded(True)
        g.ui.builder.get_object("deblur").set_expanded(True)
        g.ui.builder.get_object("colors").set_expanded(True)
        if(CPUS is not None):
            g.ui.cpus.set_value(int(CPUS))
        g.ui.window.connect("map-event", lambda *args: mapped.set())
        Gtk.main()
    context.gtkDone = gtk_main.start(run)
    assert mapped.wait(GTK_TIMEOUT), "The main window was not mapped"

def after_scenario(context: Context, scenario: Scenario) -> None:
    if(not hasattr(context, "gtkDone")):
        return
    GLib.idle_add(g.ui.window.close)
    assert context.gtkDone.wait(GTK_TIMEOUT), "The GTK main loop did not exit"

def after_all(context: Context) -> None:
    if(OUTPUTS is None):
        shutil.rmtree(context.testOutputs, ignore_errors=True)
