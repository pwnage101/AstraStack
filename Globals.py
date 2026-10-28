import copy

class Object():
    def actualColor(self):
        return (self.colorMode or self.guessedColorMode)
        
    # Returns the input's color description, None when there is none or the color mode isn't RGB
    # (grayscale image files keep theirs, they are display encoded like color ones)
    def actualColorSpace(self):
        from Video import Video
        if(self.actualColor() == Video.COLOR_RGB or
           (self.actualColor() == Video.COLOR_GRAYSCALE and self.colorSpaceFromImage)):
            return self.colorSpace
        return None

g = Object()

g.TESTING = False
g.ui = None
g.pool = None

def cloneGlobals():
    ui = g.ui
    pool = g.pool
    g.ui = None
    g.pool = None
    gCopy = copy.deepcopy(g)
    g.ui = ui
    g.pool = pool
    return gCopy
    

