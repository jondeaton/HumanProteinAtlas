#!/usr/bin/env python
"""
File: sample
Date: 10/19/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import imageio

class Sample:
    def __init__(self, id, ):
        self.id = id
        self._red = None
        self._green = None
        self._blue = None
        self._yellow = None

        @property
        def red(self):
            if self._red is not None:
                return self._red
            self.red = imageio.lo

        @property
        def green(self):
            pass

        @property
        def blue(self):
            pass

        @property
        def yellow(self):
            pass
