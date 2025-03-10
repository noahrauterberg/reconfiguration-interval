import numpy as np
import pandas as pd
import config
import networkx as nx
import typing
import os
import scipy.constants as const

# from pycallgraph2 import PyCallGraph
# from pycallgraph2.output import GraphvizOutput

INTERVAL_LENGTH = 15
TOTAL_STEPS = 5_730  # one orbital period of shell 1


def main():
    # Collect dfs of individual timesteps to concatenate them into a single df
    paths_dfs = np.empty(TOTAL_STEPS, dtype=object)
    linkwise_dfs = np.empty(TOTAL_STEPS, dtype=object)

    for step in range(0, TOTAL_STEPS, INTERVAL_LENGTH):
        print(f"Processing step {step}")
        sat_pos, gs_pos, isl_dist = load_interval(step)
        gsls = gsls_for_interval(sat_pos, gs_pos)

        for t in range(step, step + INTERVAL_LENGTH):
            sat_positions = sat_pos[t]
            gs_positions = gs_pos[t]
            isl_distances = isl_dist[t]
            G = generate_graph(sat_positions, gs_positions, isl_distances, gsls[t])
            paths_Null_Island = shortest_paths(
                G, gs_pos_to_gs_list(gs_positions), "gs_0_0"
            )
            paths_lat_25 = shortest_paths(G, gs_pos_to_gs_list(gs_positions), "gs_25_0")
            paths_lat_50 = shortest_paths(G, gs_pos_to_gs_list(gs_positions), "gs_50_0")

            # Convert to dfs and save results
            paths_Null_Island_df = paths_to_df(
                paths_Null_Island, G, t, default_target="gs_0_0"
            )
            paths_lat_25_df = paths_to_df(paths_lat_25, G, t, default_target="gs_25_0")
            paths_lat_50_df = paths_to_df(paths_lat_50, G, t, default_target="gs_50_0")
            all_targets_df = pd.concat(
                [paths_Null_Island_df, paths_lat_25_df, paths_lat_50_df],
                ignore_index=True,
            )
            paths_dfs[t] = all_targets_df
            all_targets_df.to_pickle(
                os.path.join(config.PATHS_DIR, f"{t}.pckl", "w"), index=False
            )
            linkwise_df = linkwise_analysis(all_targets_df, isl_distances)
            linkwise_dfs[t] = linkwise_df
            linkwise_df.to_pickle(
                os.path.join(config.PATHS_DIR, f"{t}.pckl", "w"), index=False
            )

    paths = pd.concat(paths_dfs, ignore_index=True)
    paths.to_pickle(os.path.join(config.PATHS_DIR, "all_timesteps.pckl"), index=False)
    links = pd.concat(linkwise_dfs, ignore_index=True)
    links.to_pickle(os.path.join(config.LINKS_DIR, "all_timesteps.pckl"), index=False)


def paths_to_df(
    paths: dict[str, typing.List[str]],
    G: nx.Graph,
    time: int,
    default_target: str = "gs_0_0",
) -> pd.DataFrame:
    """Returns a DataFrame for the given paths including a latency assumption and the given time.
    The latency is calculated by multiplying the total path length with the speed of light.

    Args:
        paths (dict[str, typing.List[str]]): Dictionary containing the paths from each source to the target with the source node as key
        G (nx.Graph): Graph to calculate the path length on
        time (int): Time of the paths
        default_target (str): The default target to use if there is no path

    Returns:
        pd.Dataframe: Dataframe based on given paths, columns: source, target, path, latency, time
    """
    # object seems to be more memory efficient than U9 (restrcting string size to 9 chars)
    sources = np.empty(len(paths), dtype=object)
    targets = np.empty(len(paths), dtype=object)
    path_list = np.empty(len(paths), dtype=object)
    latencies = np.empty(len(paths), dtype=np.float64)
    times = np.full(len(paths), time, dtype=np.int16)
    for idx, (source, path) in enumerate(paths.items()):
        if path:
            distance = nx.path_weight(G, path, "weight")
            latency = distance * const.c

            sources[idx] = source
            targets[idx] = path[-1]
            path_list[idx] = path
            latencies[idx] = latency
        else:
            sources[idx] = source
            targets[idx] = default_target
            path_list[idx] = []
            latencies[idx] = np.nan
    return pd.DataFrame(
        {
            "source": sources,
            "target": targets,
            "path": path_list,
            "latency": latencies,
            "time": times,
        }
    )


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
    for source in sources:
        try:
            path = nx.shortest_path(G, source=source, target=target, weight="weight")
            paths[source] = path
        except nx.NetworkXNoPath:
            paths[source] = []
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
    typing.Dict[int, pd.DataFrame],
    typing.Dict[int, pd.DataFrame],
    typing.Dict[int, pd.DataFrame],
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
    sat_coordinates = sat_positions[["x", "y", "z"]].values
    gs_coordinates = gs_positions[["x", "y", "z"]].values

    sat_grid, gs_grid = np.meshgrid(
        np.arange(len(sat_positions)), np.arange(len(gs_positions)), indexing="ij"
    )

    distance_matrix = np.linalg.norm(
        sat_coordinates[sat_grid] - gs_coordinates[gs_grid], axis=2
    )
    max_dist_mask = distance_matrix <= gs_positions["max_gsl_dist"].values[gs_grid]
    valid_indices = np.where(max_dist_mask)

    return pd.DataFrame(
        {
            "satellite": sat_positions["id"].values[valid_indices[0]],
            "ground_station": [
                f"{gs_positions.loc[i, 'lat']}_{gs_positions.loc[i, 'long']}"
                for i in gs_positions.index[valid_indices[1]]
            ],
            "distance": distance_matrix[valid_indices],
        }
    )


def gsls_for_interval(
    sat_interval_positions: typing.Dict[int, pd.DataFrame],
    gs_interval_positions: typing.Dict[int, pd.DataFrame],
) -> typing.Dict[int, pd.DataFrame]:
    """Determines one GSL for each GS with the lowest average distance over the whole interval.
    Assumes that keys for sat_interval_positions and gs_interval_positions are the same.

    Args:
        sat_interval_positions (typing.Dict[int, pd.DataFrame]): Dict containing the position information for all satellite over the whole interval in separate dataframes
        gs_interval_positions (typing.Dict[int, pd.DataFrame]): Dict containing the position information for all GSs over the whole interval in separate dataframes

    Returns:
        typing.Dict[int, pd.DataFrame]: DataFrame containing per-time data for gsls "chosen by the global scheduler" (lowest avg distance)
    """
    # It is reasonable to assume that the global scheduler would optimize for distance rather than latency as it does not have any information on future communication the GS might initiate
    # Anyways, optimizing this based on the known destination could very well be interesting
    timesteps = list(sat_interval_positions.keys())
    gsls = [
        possible_gsls(sat_interval_positions[t], gs_interval_positions[t])
        for t in timesteps
    ]

    common_gsls = set.intersection(
        *[set(zip(df["satellite"], df["ground_station"])) for df in gsls]
    )

    filtered_gsls = [
        df[
            df.apply(
                lambda row: (row["satellite"], row["ground_station"]) in common_gsls,
                axis=1,
            )
        ]
        for df in gsls
    ]
    common = pd.concat(filtered_gsls).sort_values(by=["satellite", "ground_station"])
    average_distances = (
        common.groupby(["satellite", "ground_station"])["distance"].mean().reset_index()
    )

    lowest_avg_dist = average_distances.loc[
        average_distances.groupby("ground_station")["distance"].idxmin()
    ]

    ret = {}
    # Pairs of GSLs that were chosen by the global scheduler because they remain valid across the whole interval and have the lowest avg distance
    chose_gsls = set(
        zip(lowest_avg_dist["satellite"], lowest_avg_dist["ground_station"])
    )
    for idx, t in enumerate(timesteps):
        cur = filtered_gsls[idx]
        # Filter the dfs so that they only contain chosen gsls
        ret[t] = cur[
            cur.apply(
                lambda row: (row["satellite"], row["ground_station"]) in chose_gsls,
                axis=1,
            )
        ]
    return ret


def gs_pos_to_gs_list(gs_positions: pd.DataFrame) -> typing.List[str]:
    ret = []
    for _, gs in gs_positions.iterrows():
        ret.append(f"gs_{gs['lat']}_{gs['long']}")
    return ret


def linkwise_analysis(connections: pd.DataFrame, isls: pd.DataFrame) -> pd.DataFrame:
    """Performs a linkwise analysis on the given connections of the dataframe.
    This is limited to ISLs as GSLs are by definition only used by one GS and satellite.

    Args:
        connections (pd.DataFrame): DataFrame containing the connections to analyze, columns: "source", "target", "path", "latency", "time"
        isls (pd.DataFrame): DataFrame containing the ISLs, columns: "a", "b", "distance"

    Returns:
        pd.DataFrame: DataFrame containing the linkwise analysis, columns: "node_a", "node_b", "used_by"
    """
    linkwise = pd.DataFrame(
        {
            "node_a": isls["a"].values,
            "node_b": isls["b"].values,
            "used_by": 0,
        }
    )

    connections["path"] = connections["path"].apply(
        lambda p: np.array(
            [int(s.removeprefix("sat_")) for s in p if not s.startswith("gs")]
        )
    )

    for _, conn in connections.iterrows():
        path = conn["path"]
        # The range excludes the GSLs
        for i in range(1, len(path) - 2):
            cond_ab = (linkwise["node_a"] == path[i]) & (
                linkwise["node_b"] == path[i + 1]
            )
            cond_ba = (linkwise["node_a"] == path[i + 1]) & (
                linkwise["node_b"] == path[i]
            )
            mask = cond_ab | cond_ba
            linkwise.loc[mask, "used_by"] += 1

    return linkwise


if __name__ == "__main__":
    main()
    # with PyCallGraph(output=GraphvizOutput()):
    #     main()
