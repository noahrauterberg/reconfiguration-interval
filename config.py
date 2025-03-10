#
# Copyright (c) Tobias Pfandzelter. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import os
import numpy as np
import scipy.constants
import scipy.special

# whether to show animations of simulated constellations
ANIMATE = False

# whether to print debug information
DEBUG = False

# number of columns in the output terminal
TERM_SIZE = 80

# radius of earth in km -> this is the mean
EARTH_RADIUS = 6_371.0

# model to use for satellite orbits
# can be SGP4 or Kepler
MODEL = "SGP4"

# simulation interval in seconds
INTERVAL = 1

# total length of the simulation in seconds
STEPS = 5_760  # longest orbital period for Starlink shells + a bit to have a multiple of 15 seconds

# speed of light in km/s
C = scipy.constants.speed_of_light / 1_000.0

# output folders
__root = os.path.abspath(os.path.dirname(__file__)) if __file__ else "."
DISTANCES_DIR = os.path.join(__root, "distances-results")
os.makedirs(DISTANCES_DIR, exist_ok=True)
SAT_POSITIONS_DIR = os.path.join(__root, "sat-positions")
os.makedirs(SAT_POSITIONS_DIR, exist_ok=True)
GS_POSITIONS_DIR = os.path.join(__root, "gs-positions")
os.makedirs(GS_POSITIONS_DIR, exist_ok=True)
GSLS_DIR = os.path.join(__root, "gsls")
os.makedirs(GSLS_DIR, exist_ok=True)
PATHS_DIR = os.path.join(__root, "paths")
os.makedirs(PATHS_DIR, exist_ok=True)
LINKS_DIR = os.path.join(__root, "links")
os.makedirs(LINKS_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(__root, "results.csv")


# constellation shells to consider
SHELLS = [
    {  # 1,584 satellites
        "name": "st1",
        "pretty_name": "Starlink 1",
        "planes": 72,
        "sats": 22,
        "altitude": 550,
        "inc": 53.0,
    },
    # {  # 1,584 satellites
    #     "name": "st2",
    #     "pretty_name": "Starlink 2",
    #     "planes": 72,
    #     "sats": 22,
    #     "altitude": 540,
    #     "inc": 53.2,
    # },
    # {  # 720 satellites
    #     "name": "st3",
    #     "pretty_name": "Starlink 3",
    #     "planes": 36,
    #     "sats": 20,
    #     "altitude": 570,
    #     "inc": 70.0,
    # },
    # {  # 348 satellites
    #     "name": "st4",
    #     "pretty_name": "Starlink 4",
    #     "planes": 6,
    #     "sats": 58,
    #     "altitude": 560,
    #     "inc": 97.6,
    # },
    # {  # 172 satellites
    #     "name": "st5",
    #     "pretty_name": "Starlink 5",
    #     "planes": 4,
    #     "sats": 43,
    #     "altitude": 560,
    #     "inc": 97.6,
    # },
]

# generate the distances
for s in SHELLS:
    if DEBUG:
        print("Calculating distances for {}".format(s["pretty_name"]))

    # intra-plane distance
    s["D_M"] = (EARTH_RADIUS + s["altitude"]) * np.sqrt(
        2 * (1 - np.cos((2 * np.pi) / s["sats"]))
    )
    if DEBUG:
        print("Intra-plane distance: {}".format(s["D_M"]))
    # max inter distance
    s["D_N_max"] = (EARTH_RADIUS + s["altitude"]) * np.sqrt(
        2 * (1 - np.cos((2 * np.pi) / s["planes"]))
    )
    if DEBUG:
        print("Max inter-plane distance: {}".format(s["D_N_max"]))

    # mean inter distance
    s["D_N_mean"] = (
        (2 / np.pi)
        * (EARTH_RADIUS + s["altitude"])
        * np.sqrt(2 * (1 - np.cos((2 * np.pi) / s["planes"])))
        * scipy.special.ellipe(1 - (np.cos(np.deg2rad(s["inc"]))) ** 2)
    )
    if DEBUG:
        print("Mean inter-plane distance: {}".format(s["D_N_mean"]))
