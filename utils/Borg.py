# coding=utf-8
class Borg:
    """
    Base Class for Singleton-Like objects.
    For more information, refer to
    http://code.activestate.com/recipes/66531-singleton-we-dont-need-no-stinkin-singleton-the-bo/
    """
    __shared_state = {}

    def __init__(self):
        self.__dict__ = self.__shared_state
