import numpy as np
import pandas as pd
import config
import networkx as nx
import matplotlib.pyplot as plt
import typing
import scipy.constants as const

INTERVAL_LENGTH = 15


def main():
    print("NOT YET IMPLEMENTED")


def paths_to_df(
    paths: dict[str, typing.List[str]],
    df: pd.DataFrame,
    G: nx.Graph,
    time: int,
):
    """Adds given paths to the given dataframe including a latency assumption and the given time.
    The latency is calculated by multiplying the total path length with the speed of light.

    Args:
        paths (dict[str, typing.List[str]]): Dictionary containing the paths from each source to the target with the source node as key
        df (pd.DataFrame): DataFrame to add the paths to
        G (nx.Graph): Graph to calculate the path length on
        time (int): Time of the paths
    """
    for source, path in paths.items():
        if path:
            distance = nx.path_weight(G, path, "weight")
            df.loc[len(df)] = {
                "source": source,
                "target": path[-1],
                "path": path,
                "latency": distance * const.c,
                "time": time,
            }
        else:
            df.loc[len(df)] = {
                "source": source,
                "target": "gs_0_0",
                "path": [],
                "latency": np.nan,
                "time": time,
            }


def shortest_paths(
    G: nx.Graph, sources: typing.List[str], target: str
) -> dict[str, typing.List[str]]:
    """Computes the shortest path from each source to the target.

    Args:
        G (nx.Graph): Graph to compute the shortest paths on
        sources (typing.List[str]): List of source nodes, most likely all (other) ground stations than the target
        target (str): Target node
    Returns:
        dict[str, typing.List[str]]: Dictionary containing the shortest path from each source to the target, key is the source node
    """
    paths = {}
    nodes_with_no_path = []
    for source in sources:
        try:
            path = nx.shortest_path(G, source=source, target=target, weight="weight")
            paths[source] = path
        except nx.NetworkXNoPath:
            nodes_with_no_path.append(source)
            paths[source] = []
    print(f"Couldn't find a path for the following nodes:\n{nodes_with_no_path}")
    return paths


def generate_graph(
    sat_positions: pd.DataFrame,
    gs_positions: pd.DataFrame,
    isl_distances: pd.DataFrame,
    gsls: pd.DataFrame,
) -> nx.Graph:
    """Generates a graph from the given dataframes.

    Args:
        sat_positions (pd.DataFrame): positions of the satellites, columns: 'id', 'x', 'y', 'z'
        gs_positions (pd.DataFrame): positions of the ground stations, columns: 'lat', 'long, 'x', 'y', 'z', 'max_gsl_dist'
        isl_distances (pd.DataFrame): inter-satellite links, columns: 'a', 'b', 'distance'
        gsls (pd.DataFrame): ground-satellite links, columns: 'satellite', 'ground_station', 'distance'

    Returns:
        nx.Graph: _description_
    """
    G = nx.Graph()
    for _, sat in sat_positions.iterrows():
        G.add_node(f"sat_{sat['id']}", pos=(sat["x"], sat["y"]))

    for _, gs in gs_positions.iterrows():
        G.add_node(f"gs_{gs['lat']}_{gs['long']}", pos=(gs["x"], gs["y"]))

    for _, isl in isl_distances.iterrows():
        G.add_edge(
            f"sat_{int(isl['a'])}", f"sat_{int(isl['b'])}", weight=isl["distance"]
        )

    for _, gsl in gsls.iterrows():
        G.add_edge(
            f"sat_{(gsl['satellite'])}",
            f"gs_{(gsl['ground_station'])}",
            weight=gsl["distance"],
        )

    return G


def load_interval(
    start_time: int,
    gs_positions_dir: str = f"{config.GS_POSITIONS_DIR}/st1",
    sat_positions_dir: str = f"{config.SAT_POSITIONS_DIR}/st1",
    isl_distances_dir: str = f"{config.DISTANCES_DIR}/st1",
    interval_length: int = INTERVAL_LENGTH,
) -> typing.Tuple[
    typing.Dict[str, pd.DataFrame],
    typing.Dict[str, pd.DataFrame],
    typing.Dict[str, pd.DataFrame],
]:
    """
    Load the satellite positions, ground station positions, and inter-satellite distances for the given interval.

    Args:
        start_time: The start time of the interval.
    Returns:
        A tuple containing the satellite positions, ground station positions, and inter-satellite distances for each timestep in the interval.
    """
    ret_sat_pos = {}
    ret_gs_pos = {}
    ret_isl_dist = {}

    for time in range(start_time, start_time + interval_length):
        gs_pos = pd.read_csv(f"{gs_positions_dir}/{time}.csv")
        sat_pos = pd.read_csv(f"{sat_positions_dir}/{time}.csv")
        isl_dist = pd.read_csv(f"{isl_distances_dir}/{time}.csv")

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
        gs_positions (pd.DataFrame): Positions of the ground stations, columns: 'lat', 'long, 'x', 'y', 'z', 'max_gsl_dist'

    Returns:
        pd.DataFrame: DataFrame with all possible GSLs, columns: 'satellite', 'ground_station', 'distance'
    """
    possible_gsls = []
    for _, sat in sat_positions.iterrows():
        for _, gs in gs_positions.iterrows():
            sat_pos = np.array([sat["x"], sat["y"], sat["z"]])
            gs_pos = np.array([gs["x"], gs["y"], gs["z"]])
            gsl_len = np.linalg.norm(gs_pos - sat_pos)
            if gsl_len <= gs["max_gsl_dist"]:
                link = {
                    "satellite": sat["id"],
                    "ground_station": f"{gs['lat']}_{gs['long']}",
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


def gs_pos_to_gs_list(gs_positions: pd.DataFrame) -> typing.List[str]:
    ret = []
    for _, gs in gs_positions.iterrows():
        ret.append(f"gs_{gs['lat']}_{gs['long']}")
    return ret


if __name__ == "__main__":
    total_steps = 10
    INTERVAL_LENGTH = 5

    df = pd.DataFrame(
        {
            "source": pd.Series(dtype=str),
            "target": pd.Series(dtype=str),
            "path": pd.Series(dtype=object),
            "latency": pd.Series(dtype=float),
            "time": pd.Series(dtype=int),
        }
    )
    for step in range(0, total_steps, INTERVAL_LENGTH):
        sat_pos, gs_pos, isl_dist = load_interval(step)
        gsls = gsls_for_interval(sat_pos, gs_pos)
        # gsls.to_csv("debug/gsls.csv")
        # gsls = pd.read_csv("debug/gsls.csv")

        for t in range(0, INTERVAL_LENGTH):
            sat_positions = sat_pos[step + t]
            gs_positions = gs_pos[step + t]
            isl_distances = isl_dist[step + t]
            # This is inaccurate as gsls only represents average distances over the whole interval
            # This should be relatively easy to fix as the GSLs are already calculated and just need to be returned by gsls_for_interval
            # TODO for future me, not a priority
            G = generate_graph(sat_positions, gs_positions, isl_distances, gsls)
            paths = shortest_paths(G, gs_pos_to_gs_list(gs_positions), "gs_0_0")
            paths_to_df(paths, df, G, t)
    df.to_csv("debug/paths.csv")
