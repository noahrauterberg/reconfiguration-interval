#
# Copyright (c) 2025 Noah Rauterberg. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# TODO: copied from analysis.py for now
INTERVAL_LENGTH = 15
TOTAL_STEPS = 5_730  # one orbital period of shell 1
SERVER_GS = ["gs_0_0", "gs_25_0", "gs_50_0"]


def main():
    # Load data
    paths_df_1 = load_all_timesteps("paths_1")
    links_df_1 = load_all_timesteps("links_1")

    paths_df_15 = load_all_timesteps("paths_15")
    links_df_15 = load_all_timesteps("links_15")

    # Add path length
    paths_df_15["path_length"] = paths_df_15["path"].apply(calc_path_length)
    paths_df_1["path_length"] = paths_df_1["path"].apply(calc_path_length)

    # Add Latitude and Longitude columns
    paths_df_15["latitude"] = paths_df_15["source"].apply(extract_latitude)
    paths_df_15["longitude"] = paths_df_15["source"].apply(extract_longitude)
    paths_df_1["latitude"] = paths_df_1["source"].apply(extract_latitude)
    paths_df_1["longitude"] = paths_df_1["source"].apply(extract_longitude)

    path_stability(paths_1=paths_df_1, paths_15=paths_df_15)


def latency_cost(
    paths_1: pd.DataFrame, paths_15: pd.DataFrame, save_path: str = "latency-cost"
):
    # TOOD: implement
    raise ("Not yet implemented")


def path_stability(
    paths_1: pd.DataFrame, paths_15: pd.DataFrame, save_path: str = "path-stability"
):
    for gs in SERVER_GS:
        relevant_1 = paths_1.loc[paths_1["target"] == gs]
        relevant_15 = paths_15.loc[paths_15["target"] == gs]

        relevant_1["path_tuple"] = relevant_1["path"].apply(tuple)
        relevant_15["path_tuple"] = relevant_15["path"].apply(tuple)

        unique_paths_1 = relevant_1.group_by("source")["path_tuple"].nunique()
        unique_paths_15 = relevant_15.group_by("source")["path_tuple"].nunique()

        stability_metrics = pd.DataFrame(
            {
                "source": unique_paths_1.index.get_level_values("source"),
                "target": gs,
                "path_changes_15_sec": unique_paths_15.values,
                "path_changes_1_sec": unique_paths_1.values,
                "path_stability": unique_paths_15.values / unique_paths_1.values,
            }
        )

        coords = stability_metrics["source"].apply(extract_coords)
        stability_metrics[["latitude", "longitude"]] = coords.to_list()

        # TODO: This could be supplemented by the average path duration

        if save_path == None:
            continue
        plot_cdf(
            unique_paths_1,
            series2=unique_paths_15,
            title=f"CDF for the number of unique paths to {gs}",
            x_label="Number of unique paths",
            save_path=f"{save_path}-unique-paths-to-{gs}",
            label="1-sec interval",
            label2="15-sec interval",
        )
        plot_cdf(
            stability_metrics,
            title=f"CDF for the path stability to {gs}",
            x_label="Path Stability Ratio",
            save_path=f"{save_path}-stability-ratio-{gs}",
        )
        plot_hist(
            data_x=["longitude"],
            x_label="Longitude",
            data_y=["latitude"],
            y_label="Latitude",
            weight=["path_stability"],
            weight_label="Path Stability",
            title="Path Stability Distribution for Server-GS {gs}",
            save_path=f"{save_path}-stability-hist-{gs}",
        )


def shortest_path_to_longest_path(df):
    ratios = []
    pairs = df[["source", "target"]].drop_duplicates().values
    for source, target in pairs:
        relevant = df[(df["source"] == source) & (df["target"] == target)]
        min_latency = relevant["latency"].min()
        max_latency = relevant["latency"].max()
        ratios.append(
            {
                "source": source,
                "target": target,
                "min": min_latency,
                "max": max_latency,
                "ratio": min_latency / max_latency,
            }
        )
    return pd.DataFrame(ratios)


def plot_ratios(df, save_path: str = None) -> None:
    plt.figure(figsize=(12, 6))
    series = df["ratio"]
    sns.histplot(series, stat="proportion", cumulative=True, alpha=0.4)
    sns.ecdfplot(series, stat="proportion")
    if save_path:
        plt.savefig(save_path)
    plt.close()


def rtt_variability(df):
    latency_by_source = df.groupby(["source"])["latency"].mean()
    latency_by_target = df.groupby(["target"])["latency"].mean()
    return {
        "mean": df["latency"].mean(),
        "std_dev": df["latency"].std(),
        "min": df["latency"].min(),
        "max": df["latency"].max(),
        "latency_by_source_mean": latency_by_source.mean(),
        "latency_by_source_std_dev": latency_by_source.std(),
        "latency_by_source_min": latency_by_source.min(),
        "latency_by_source_max": latency_by_source.max(),
        "latency_by_target_mean": latency_by_target.mean(),
        "latency_by_target_std_dev": latency_by_target.std(),
        "latency_by_target_min": latency_by_target.min(),
        "latency_by_target_max": latency_by_target.max(),
    }


def plot_rtt_distribution(df, save_path: str = None) -> None:
    plt.figure(figsize=(12, 6))

    rtt_values = df.groupby(["time"])["latency"].mean().reset_index()

    sns.lineplot(data=rtt_values, x="time", y="latency")
    plt.title("RTT Distribution Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Round-Trip Time (seconds)")

    if save_path:
        plt.savefig(save_path)
    plt.close()


def route_churn(df):
    raise ("not implemented yet")


def network_performance(df):
    df_by_time = df.groupby(["time"]).mean()
    return {
        "avg_link_util": df["used_by"].mean(),
        "min_link_util": df["used_by"].min(),
        "max_link_util": df["used_by"].max(),
        "avg_link_util_by_time": df_by_time["used_by"].mean(),
        "min_link_util_by_time": df_by_time["used_by"].min(),
        "max_link_util_by_time": df_by_time["used_by"].max(),
    }


def plot_link_utilization_heatmap(df, save_path: str = None) -> None:
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")

    pivot_data = df.pivot_table(
        index="node_a", columns="node_b", values="used_by", aggfunc="mean"
    )

    sns.heatmap(pivot_data, cmap="YlOrRd", annot=True, fmt=".1f")
    plt.title("Average Link Utilization Between Satellites")
    plt.xlabel("Satellite ID")
    plt.ylabel("Satellite ID")

    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_hist(
    data_x: pd.Series,
    x_label: str,
    data_y: pd.Series,
    y_label: str,
    weight: pd.Series,
    weight_label: str,
    title: str,
    save_path: str = None,
):
    plt.figure(figsize=(12, 8))
    plt.hist2d(data_x, data_y, bins=[36, 11], weights=weight)
    plt.colorbar(label=weight_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_cdf(
    series: pd.Series,
    title: str,
    x_label: str,
    y_label: str = "Probability",
    label: str = "",
    stat: str = "proportion",
    save_path: str = None,
    series2: pd.Series = None,
    label2: str = "",
):
    plt.figure(figsize=(10, 6))
    sns.ecdfplot(data=series, stat=stat, label=label)
    sns.histplot(data=series, stat=stat, alpha=0.4, cumulative=True, label=label)

    if series2:
        sns.ecdfplot(data=series, stat=stat, label=label2)
        sns.histplot(
            data=series2,
            stat=stat,
            alpha=0.4,
            cumulative=True,
            label=label2,
        )
        plt.legend(fontsize=10)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def calc_path_length(path):
    return len(path + 2)


def extract_coords(source):
    return extract_latitude(source), extract_longitude(source)


def extract_latitude(source):
    return source.split("_")[1]


def extract_longitude(source):
    return source.split("_")[2]


def load_all_timesteps(data_type):
    return pd.read_pickle(f"{data_type}/all_timesteps.pckl")


def load_all_timesteps_again(data_type):
    dfs = np.empty(TOTAL_STEPS)
    for time in range(TOTAL_STEPS):
        dfs[time] = load_individual_timestep(data_type, time)
    return pd.concat(dfs, ignore_index=True)


def load_individual_timestep(data_type, time):
    return pd.read_pickle(f"{data_type}/{time}.pckl")


if __name__ == "__main__":
    main()
