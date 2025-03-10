#
# Copyright (c) Tobias Pfandzelter. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import tqdm
import os
import sys
import concurrent.futures
import numpy as np

import config
from simulation.simulation import Simulation
from simulation.groundstation import GroundStation

sys.path.append(os.path.abspath(os.getcwd()))


def run_simulation(
    steps: int,
    interval: float,
    planes: int,
    nodes: int,
    inc: float,
    altitude: int,
    name: str,
    groundstations: GroundStation,
    sat_links_dir: str,
    sat_positions_dir: str,
    gs_positions_dir: str,
):
    sat_links_dir = os.path.join(sat_links_dir, name)
    sat_positions_dir = os.path.join(sat_positions_dir, name)
    gs_positions_dir = os.path.join(gs_positions_dir, name)

    os.makedirs(sat_links_dir, exist_ok=True)
    os.makedirs(sat_positions_dir, exist_ok=True)
    os.makedirs(gs_positions_dir, exist_ok=True)

    # setup simulation
    semi_major = (
        int(altitude + config.EARTH_RADIUS) * 1000
    )  # semi-major axis in meters. Since we are only concerned with circular orbits, this is just the radius of the orbit = altitude + earth_radius
    s = Simulation(
        planes=planes,
        nodes_per_plane=nodes,
        inclination=inc,
        semi_major_axis=semi_major,
        model=config.MODEL,
        animate=config.ANIMATE,
        report_status=config.DEBUG,
        groundstations=groundstations,
    )

    # for each timestep, run simulation
    # for step in tqdm.trange(total_steps, desc="simulating {}".format(name)):
    for next_time in range(0, steps, interval):
        with open(
            os.path.join(sat_links_dir, "{}.csv".format(next_time)), "w"
        ) as sat_links, open(
            os.path.join(sat_positions_dir, "{}.csv".format(next_time)), "w"
        ) as sat_pos, open(
            os.path.join(gs_positions_dir, "{}.csv".format(next_time)), "w"
        ) as gs_pos:
            sat_links.write("a,b,distance\n")
            sat_pos.write("id,x,y,z\n")
            gs_pos.write("lat,long,x,y,z,max_gsl_dist\n")

            s.update_model(
                new_time=next_time,
                link_file=sat_links,
                positions_file=sat_pos,
                gs_file=gs_pos,
            )

    if s.animation is not None:
        s.animation.terminate()

    s.terminate()


if __name__ == "__main__":
    # Generate ground stations
    ground_stations = []

    for long in range(0, 60, 10):
        equator = GroundStation(
            f"equator_{long}",
            0,
            long,
            25,
        )
        ground_stations.append(equator)
        for lat in range(10, 60, 10):
            north = GroundStation(
                f"north_{lat}_{long}",
                lat,
                long,
                25,
            )
            south = GroundStation(
                f"south_{lat}_{long}",
                -lat,
                -long,
                25,
            )
            ground_stations.append(north)
            ground_stations.append(south)

    with concurrent.futures.ProcessPoolExecutor() as executor:

        for s in config.SHELLS:
            # run_simulation(
            #     config.STEPS,
            #     config.INTERVAL,
            #     int(s["planes"]),
            #     int(s["sats"]),
            #     float(s["inc"]),
            #     int(s["altitude"]),
            #     s["name"],
            #     ground_stations,
            #     config.DISTANCES_DIR,
            #     config.SAT_POSITIONS_DIR,
            #     config.GS_POSITIONS_DIR,
            # )
            executor.submit(
                run_simulation,
                config.STEPS,
                config.INTERVAL,
                int(s["planes"]),
                int(s["sats"]),
                float(s["inc"]),
                int(s["altitude"]),
                s["name"],
                ground_stations,
                config.DISTANCES_DIR,
                config.SAT_POSITIONS_DIR,
                config.GS_POSITIONS_DIR,
            )
