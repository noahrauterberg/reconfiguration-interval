import numpy as np
import pandas as pd
import config
import data_loader as dl
import networkx as nx
import matplotlib.pyplot as plt
import typing

INTERVAL_LENGTH = 15


def main():
    print("NOT YET IMPLEMENTED")


def load_interval(
    start_time: int,
) -> typing.Tuple[
    typing.Dict[str, pd.DataFrame],
    typing.Dict[str, pd.DataFrame],
    typing.Dict[str, pd.DataFrame],
]:
    """
    Load the satellite positions, ground station positions, and inter-satellite distances for the given interval.

    :param
    start_time: The start time of the interval.
    :return: A tuple containing the satellite positions, ground station positions, and inter-satellite distances.
    """
    ret_sat_pos = {}
    ret_gs_pos = {}
    ret_isl_dist = {}

    for time in range(start_time, start_time + INTERVAL_LENGTH):
        # sat_pos = dl.load_file(f"debug/sat_pos/{time}.csv")
        # gs_pos = dl.load_file(f"debug/gs_pos/{time}.csv")
        # isl_dist = dl.load_file(f"debug/isls/{time}.csv")
        sat_pos = dl.load_file(f"{config.SAT_POSITIONS_DIR}/st1/{time}.csv")
        gs_pos = dl.load_file(f"{config.GS_POSITIONS_DIR}/st1/{time}.csv")
        isl_dist = dl.load_file(f"{config.DISTANCES_DIR}/st1/{time}.csv")

        ret_sat_pos[time] = sat_pos
        ret_gs_pos[time] = gs_pos
        ret_isl_dist[time] = isl_dist

    return ret_sat_pos, ret_gs_pos, ret_isl_dist


def possible_gsls(
    sat_positions: pd.DataFrame, gs_positions: pd.DataFrame
) -> pd.DataFrame:
    """Returns a DataFrame with all possible GSLs between satellites and ground stations.

    Args:
        sat_positions (pd.DataFrame): Positions of the satellites, columns: 'id', 'x', 'y', 'z'
        gs_positions (pd.DataFrame): Positions of the ground stations, columns: 'name', 'x', 'y', 'z'

    Returns:
        pd.DataFrame: DataFrame with all possible GSLs, columns: 'satellite', 'ground_station', 'distance'
    """
    possible_gsls = []
    for _, sat in sat_positions.iterrows():
        for _, gs in gs_positions.iterrows():
            sat_pos = np.array([sat["x"], sat["y"], sat["z"]])
            gs_pos = np.array([gs["x"], gs["y"], gs["z"]])
            gsl_len = np.linalg.norm(gs_pos - sat_pos)
            if gsl_len <= max_gsl_dist:
                link = {
                    "satellite": sat["id"],
                    "ground_station": gs["name"],
                    "distance": gsl_len,
                }
                possible_gsls.append(link)

    return pd.DataFrame(possible_gsls)


def gsls_for_interval(
    sat_interval_positions: typing.Dict[str, pd.DataFrame],
    gs_interval_positions: typing.Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Determines one GSL for each GS with the lowest average distance over the whole interval.
    Assumes that keys for sat_interval_positions and gs_interval_positions are the same.

    Args:
        sat_interval_positions (typing.Dict[str, pd.DataFrame]): Dict containing the position information for all satellite over the whole interval in separate dataframes
        gs_interval_positions (typing.Dict[str, pd.DataFrame]): Dict containing the position information for all GSs over the whole interval in separate dataframes

    Returns:
        pd.DataFrame: DataDrame with GSLs that
    """
    # TODO: is this reasonable? Yes, because the global scheduler does not have any information on future communication the GS might initiate
    # To optimize this based on the known destination could very well be interesting to analyze
    timesteps = sat_interval_positions.keys()
    gsls = []
    for t in timesteps:
        sat_positions = sat_interval_positions[t]
        gs_positions = gs_interval_positions[t]
        gsls.append(possible_gsls(sat_positions, gs_positions))

    # This is only for debugging:
    if len(timesteps) == 1:
        return gsls[0]

    common_sat_gsls = set.intersection(
        *[set(zip(df["satellite"], df["ground_station"])) for df in gsls]
    )

    filtered_gsls = []
    for df in gsls:
        mask = df.apply(
            lambda row: (row["satellite"], row["ground_station"]) in common_sat_gsls,
            axis=1,
        )
        filtered_gsls.append(df[mask])
    common = pd.concat(filtered_gsls).sort_values(by=["satellite", "ground_station"])
    average_distances = (
        common.groupby(["satellite", "ground_station"])["distance"].mean().reset_index()
    )

    return average_distances.loc[
        average_distances.groupby("ground_station")["distance"].idxmin()
    ]


if __name__ == "__main__":
    max_gsl_dist = 1141384  # this is constant for now, but in reality depends on the altitude of the satellite

    sat_pos, gs_pos, isl_dist = load_interval(0)
    gsls = gsls_for_interval(sat_pos, gs_pos).sort_values(by=["ground_station"])
    gsls.to_csv("debug/gsls.csv", index=False)
