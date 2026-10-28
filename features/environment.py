import os
import sys
import time
import shutil
import tempfile
from itertools import count
from threading import Event, Thread
from behave.model import Scenario
from behave.runner import Context
from gi import require_version
require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib
sys.path.insert(0, os.getcwd())

from UI import UI
from Globals import g

CPUS = os.environ.get("ASTRASTACK_TEST_CPUS")
SHARD = os.environ.get("ASTRASTACK_TEST_SHARD")
GTK_TIMEOUT = 30

def before_all(context: Context) -> None:
    context.testOutputs = tempfile.mkdtemp(prefix="astrastack-test-")
    context.scenarioIndex = count()

def before_scenario(context: Context, scenario: Scenario) -> None:
    index = next(context.scenarioIndex)
    if(SHARD is not None):
        shard, shards = (int(n) for n in SHARD.split("/"))
        if(index % shards != shard):
            scenario.skip(f"Runs in shard {index % shards}/{shards}")
            return
    g.TESTING = True
    g.ui = UI()
    g.ui.builder.get_object("wavelets").set_expanded(True)
    g.ui.builder.get_object("deblur").set_expanded(True)
    g.ui.builder.get_object("colors").set_expanded(True)
    if(CPUS is not None):
        g.ui.cpus.set_value(int(CPUS))
    mapped = Event()
    g.ui.window.connect("map-event", lambda *args: mapped.set())
    context.gtkThread = Thread(target=Gtk.main, args=())
    context.gtkThread.start()
    assert mapped.wait(GTK_TIMEOUT), "The main window was not mapped"

def after_scenario(context: Context, scenario: Scenario) -> None:
    if(not hasattr(context, "gtkThread")):
        return
    # Let processing finish and close the window from the GTK thread, so no GTK work overlaps the next scenario's UI
    ui = g.ui
    while(ui.processThread is not None and ui.processThread.is_alive()):
        time.sleep(0.01)
    def close() -> None:
        # Dialogs are toplevels GTK keeps alive until destroyed; a hidden file chooser left from an earlier scenario
        # keeps updating its path bar, which can make GTK abort (gtkpathbar.c set_button_image_get_info_cb)
        for dialog in (ui.openDialog, ui.saveDialog):
            dialog.destroy()
        ui.window.close()
    GLib.idle_add(close)
    context.gtkThread.join(GTK_TIMEOUT)
    assert not context.gtkThread.is_alive(), "The GTK main loop did not exit"

def after_all(context: Context) -> None:
    shutil.rmtree(context.testOutputs, ignore_errors=True)
