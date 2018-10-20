#!/usr/bin/env python
"""
File: sample
Date: 10/19/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import imageio
import numpy as np
from enum import Enum


class Color(Enum):
    red = 0
    yellow = 1
    green = 2
    blue = 3


colors = {
    'red': Color.red,
    'yellow': Color.yellow,
    'green': Color.green,
    'blue': Color.blue
}

color_names = {color: name for name, color in colors.items()}

default_color_ordering = {color: idx for idx, color in enumerate(colors.values())}


class Sample:
    def __init__(self, sample_id, images_dir, cache=False,
                 color_ordering=default_color_ordering):
        # public
        self.id = sample_id
        self.color_ordering = color_ordering

        # private
        self._images_dir = images_dir
        self._cache = cache

        self._image_locations = dict()
        self._images = dict()
        self._combined = None

        self._shape = None

    @property
    def red(self):
        return self.image(Color.red)

    @property
    def yellow(self):
        return self.image(Color.yellow)

    @property
    def green(self):
        return self.image(Color.green)

    @property
    def blue(self):
        return self.image(Color.blue)

    @property
    def combined(self):
        """

        the first time that we call this all the images
        which we may have loaded into the "_images" dictionary
        get moved into the combined numpy array and are accessed
        from that location in stead thereafter

        :return:
        """
        if self._combined is not None:
            return self._combined

        combined = np.empty((4,) + self.shape)

        for color in list(Color):

            if color in self._images:
                img = self._images[color]
                del self._images[color]
            else:
                img = self._load_image(color)

            idx = self.color_ordering[color]
            combined[idx] = img

        if self._cache:
            self._combined = combined

        return combined

    def image(self, color):
        assert isinstance(color, Color)

        if self._combined is not None:
            idx = self.color_ordering[color]
            return self._combined[idx]

        if color in self._images:
            return self._images[color]

        img = self._load_image(color)

        if self._cache:
            self._images[color] = img
        return img

    def drop_cache(self):
        self._images = dict()
        self._image_locations = dict()
        self._combined = None

    def show_single(self, color):
        # debugging
        if isinstance(color, str):
            color = colors[color]
        import matplotlib.pyplot as plt
        plt.imshow(self.image(color))

    def show(self):
        # for debugging
        import matplotlib.pyplot as plt
        comb = np.empty((3,) + self.shape)
        comb[0] = self.red
        comb[1] = self.green
        comb[2] = self.blue

        comb[0] += self.yellow / 2
        comb[1] += self.yellow / 2

        plt.imshow(comb)

    @property
    def image_locations(self):
        if self._image_locations:
            return self._image_locations

        self._image_locations = dict()
        for color in list(Color):
            self._image_locations[color] = self.location(color)
        return self._image_locations

    def location(self, color):
        color_name = color_names[color]
        return "{id}_{color}.png".format(id=self.id, color=color_name)

    @property
    def shape(self):
        if self._shape is None:
            self._shape = self.red.shape
        return self._shape

    def _load_image(self, color):
        png_file = self.image_locations[color]
        img = imageio.imread(png_file)
        if self._shape is None:
            self._shape = img.shape
        return img
