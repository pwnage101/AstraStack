import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ColorSpace

class Object():
    def actualColor(self):
        return (self.colorMode or self.guessedColorMode)

    # Returns the working space of the loaded file's working data, which has no primaries once demosaiced
    def actualWorkingSpace(self) -> "ColorSpace.WorkingSpace":
        import ColorSpace
        from Video import Video
        if(self.actualColor() >= Video.COLOR_RGGB):
            return ColorSpace.UNTAGGED
        return self.workingSpace

g = Object()

g.TESTING = False
g.ui = None
g.decodeParams = None
g.workingSpace = None

# Returns a deep copy of the globals for worker processes, without the UI
def cloneGlobals() -> Object:
    # The memo maps it to None, so g itself is never changed while other threads read it
    return copy.deepcopy(g, memo={id(g.ui): None})
