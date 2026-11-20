import copy

class Object():
    pass


g = Object()

g.TESTING = False
g.ui = None

# Returns a deep copy of the globals for worker processes, without the UI
def cloneGlobals() -> Object:
    # The memo maps it to None, so g itself is never changed while other threads read it
    return copy.deepcopy(g, memo={id(g.ui): None})
