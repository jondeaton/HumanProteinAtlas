#!/usr/bin/env python
"""
File: organelle
Date: 10/19/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

from enum import Enum

organelle_name = {
    0: "Nucleoplasm",
    1: "Nuclear membrane",
    2: "Nucleoli",
    3: "Nucleoli fibrillar center",
    4: "Nuclear speckles",
    5: "Nuclear bodies",
    6: "Endoplasmic reticulum",
    7: "Golgi apparatus",
    8: "Peroxisomes",
    9: "Endosomes",
    10: "Lysosomes",
    11: "Intermediate filaments",
    12: "Actin filaments",
    13: "Focal adhesion sites",
    14: "Microtubules",
    15: "Microtubule ends",
    16: "Cytokinetic bridge",
    17: "Mitotic spindle",
    18: "Microtubule organizing center",
    19: "Centrosome",
    20: "Lipid droplets",
    21: "Plasma membrane",
    22: "Cell junctions",
    23: "Mitochondria",
    24: "Aggresome",
    25: "Cytosol",
    26: "Cytoplasmic bodies",
    27: "Rods & rings"
}

class Organelle(Enum):
    nucleoplasm = 0
    nuclear_membrane = 1
    nucleoli = 2
    nucleoli_fibrillar_center = 3
    nuclear_speckles = 4
    nuclear_bodies = 5
    endoplasmic_reticulum = 6
    golgi_apparatus = 7
    peroxisomes = 8
    endosomes = 9
    lysosomes = 10
    intermediate_filaments = 11
    actin_filaments = 12
    focal_adhesion_sites = 13
    microtubules = 14
    microtubule_ends = 15
    Cytokinetic_bridge = 16
    mitotic_spindle = 17
    microtubule_organizing_center = 18
    centrosome = 19
    lipid_droplets = 20
    plasma_membrane = 21
    cell_junctions = 22
    mitochondria = 23
    aggresome = 24
    cytosol = 25
    cytoplasmic_bodies = 26
    rods_and_rings = 27