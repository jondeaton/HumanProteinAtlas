#!/usr/bin/env python
"""
File: sample
Date: 10/19/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os
import imageio
import numpy as np
from enum import Enum
from HumanProteinAtlas import Organelle


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
    def __init__(self, sample_id, labels, images_dir, cache=False,
                 color_ordering=default_color_ordering):
        # public
        self.id = sample_id
        self.color_ordering = color_ordering
        self.labels = labels
        self._one_hot_labels = None

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
    def multi_channel(self):
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
            img = self.image(color)
            if color in self._images:
                del self._images[color]
            idx = self.color_ordering[color]
            combined[idx] = img

        if self._cache:
            self._combined = combined

        return combined

    def image(self, color):
        if isinstance(color, str):
            color = colors[color]

        assert isinstance(color, Color)

        if self._combined is not None:
            idx = self.color_ordering[color]
            return self._combined[idx]

        if color in self._images:
            return self._images[color]

        img = self._load_image(color) / 256

        if self._cache:
            self._images[color] = img
        return img

    @property
    def multi_hot_label(self):
        if self._one_hot_labels is None:
            a = np.array(self.labels)
            b = np.zeros((a.size, len(Organelle)))
            b[np.arange(a.size), a] = 1
            self._one_hot_labels = np.sum(b, axis=0)
        return self._one_hot_labels

    def drop_cache(self):
        self._images = dict()
        self._image_locations = dict()
        self._combined = None

    def show(self, color=None):
        # for debugging
        import matplotlib.pyplot as plt
        if color is None:
            comb = np.ones(self.shape + (3,))
            comb[:, :, 0] = self.red
            comb[:, :, 1] = self.green
            comb[:, :, 2] = self.blue

            comb[:, :, 0] += self.yellow
            comb[:, :, 1] += self.yellow

            plt.imshow(comb)
            plt.title("Sample: %s (combined)" % self.id)
            plt.show()

        else:
            if isinstance(color, str):
                color = colors[color]
            plt.imshow(self.image(color))
            plt.title("Sample: %s (%s)" % (self.id, colors[color]))
            plt.show()

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
        fname = "{id}_{color}.png".format(id=self.id, color=color_name)
        return os.path.join(self._images_dir, fname)

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
