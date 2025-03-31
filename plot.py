#
# Copyright (c) 2025 Noah Rauterberg. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

INTERVAL_LENGTH = 30
TOTAL_STEPS = 5_730  # one orbital period of shell 1
TOTAL_INTERVALS = TOTAL_STEPS // INTERVAL_LENGTH
SERVER_GS = ["gs_0_0", "gs_25_0", "gs_50_0"]
FIGURE_TYPE = "pdf"


def main():
    # Load data
    paths_df_1 = load_all_timesteps(1, "paths")
    paths_df_15 = load_all_timesteps(15, "paths")
    paths_df_30 = load_all_timesteps(30, "paths")

    # Add path length
    # paths_df_1["path_length"] = paths_df_1["path"].apply(calc_path_length)
    # paths_df_15["path_length"] = paths_df_15["path"].apply(calc_path_length)
    # paths_df_30["path_length"] = paths_df_30["path"].apply(calc_path_length)

    # Add Latitude and Longitude columns
    paths_df_1["latitude"] = paths_df_1["source"].apply(extract_latitude)
    paths_df_1["longitude"] = paths_df_1["source"].apply(extract_longitude)
    paths_df_15["latitude"] = paths_df_15["source"].apply(extract_latitude)
    paths_df_15["longitude"] = paths_df_15["source"].apply(extract_longitude)
    paths_df_30["latitude"] = paths_df_30["source"].apply(extract_latitude)
    paths_df_30["longitude"] = paths_df_30["source"].apply(extract_longitude)

    # Add interval column to reconfiguration interval dfs
    paths_df_15["interval"] = paths_df_15["time"] // 15
    paths_df_30["interval"] = paths_df_30["time"] // 30

    # gsl_switches_in_interval(paths_df_15, save_path="gsl-switches-15")
    # gsl_switches_in_interval(paths_df_30, save_path="gsl-switches-30")
    stability_per_gs = path_stability(
        paths_1=paths_df_1, paths_15=paths_df_15, paths_30=paths_df_30
    )
    latency_costs_by_connection = latency_costs(
        paths_1=paths_df_1, paths_15=paths_df_15, paths_30=paths_df_30
    )
    for gs in SERVER_GS:
        stability_per_gs[gs].to_csv(f"debug/stability-{gs}.csv")
    latency_costs_by_connection.to_csv("debug/latency-costs.csv")
    plot_stability_vs_latency_costs(stability_per_gs, latency_costs_by_connection)


def plot_stability_vs_latency_costs(stability_per_gs, latency_costs_by_connection):
    # Stability vs Latency-Costs
    for gs in SERVER_GS:
        plt.figure(figsize=(12, 8))
        stability = stability_per_gs[gs]
        costs = latency_costs_by_connection[latency_costs_by_connection["target"] == gs]
        sns.scatterplot(
            x=stability["path_stability_15_sec"],
            y=costs["latency_costs_15"],
            label="15-sec interval",
            color="darkorange",
        )
        # sns.scatterplot(
        #     x=stability["path_stability_30_sec"],
        #     y=costs["latency_costs_30"],
        #     label="30-sec interval",
        #     color="darkgreen",
        # )
        plt.title(f"Path Stability vs. Latency Costs for {gs}")
        plt.xlabel("Path Stability")
        plt.ylabel("Latency Costs")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"plots/stability-vs-latency-costs-{gs}-combined.{FIGURE_TYPE}")
        plt.close()


def gsl_duration(df_in):
    df = df_in.copy()
    df = df[df["target"] == "gs_0_0"]
    df["first_hop"] = df["path"].apply(lambda x: x[0] if len(x) > 0 else -1).astype(int)
    df = df[df["first_hop"] != -1]

    ret = []
    tmp = df.groupby(["interval", "source"])["first_hop"].unique().reset_index()
    for _, item in tmp.iterrows():
        interval_data = df[df["interval"] == item["interval"]]
        for sat in item["first_hop"]:
            duration = len(
                interval_data[
                    (interval_data["first_hop"] == sat)
                    & (interval_data["source"] == item["source"])
                ]
            )
            ret.append(
                {
                    "interval": item["interval"],
                    "source": item["source"],
                    "satellite": sat,
                    "duration": duration,
                }
            )
    return pd.DataFrame(ret)


def latency_costs(
    paths_1: pd.DataFrame,
    paths_15: pd.DataFrame = None,
    paths_30: pd.DataFrame = None,
    save_path: str = "latency-costs",
):
    merged = pd.merge(
        paths_15,
        paths_30,
        on=["source", "target", "time"],
        suffixes=("_15", "_30"),
    )
    merged = pd.merge(
        merged,
        paths_1,
        on=["source", "target", "time"],
    )

    merged["latency_costs_15"] = merged[f"latency_15"] - merged["latency"]
    merged["latency_costs_30"] = merged[f"latency_30"] - merged["latency"]
    latency_costs_by_connection = (
        merged.groupby(["source", "target"])["latency_costs_15", "latency_costs_30"]
        .agg(["max", "min", "mean"])
        .reset_index()
    )
    intervals_15 = (
        latency_costs_by_connection["latency_costs_15"]
        .sort_values("mean")
        .reset_index()
    )
    intervals_30 = (
        latency_costs_by_connection["latency_costs_30"]
        .sort_values("mean")
        .reset_index()
    )
    plot_cdf(
        [intervals_15["mean"], intervals_30["mean"]],
        title="CDF for the Average Latency Costs by Connection",
        x_label="Average Latency Costs",
        save_path=f"{save_path}-by-connection-combined",
        precision=4,
        marker_at=[0],
        colors=["darkorange", "green"],
        labels=["15-sec interval", "30-sec interval"],
        percentiles=[],
    )

    # Plot Latency Costs for 30-sec
    plt.figure(figsize=(12, 8))
    plt.plot(
        intervals_30["min"],
        label="Minimum Latency Costs",
        color="green",
    )
    plt.plot(
        intervals_30["max"],
        label="Maximum Latency Costs",
        color="green",
    )
    plt.plot(
        intervals_30["mean"],
        label="Average Latency Costs",
        color="lightgreen",
    )
    plt.fill_between(
        intervals_30.index,
        intervals_30["min"],
        intervals_30["max"],
        alpha=0.5,
        color="green",
    )
    plt.title("Latency Costs (30-sec interval)")
    plt.xlabel("Connection ID")
    plt.ylabel("Latency Costs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/{save_path}-min-max-mean-30.{FIGURE_TYPE}")

    plt.figure(figsize=(12, 8))
    plt.plot(
        intervals_15["min"],
        label="Minimum Latency Costs",
        color="darkorange",
    )
    plt.plot(
        intervals_15["max"],
        label="Maximum Latency Costs",
        color="darkorange",
    )
    plt.plot(
        intervals_15["mean"],
        label="Average Latency Costs",
        color="orange",
    )
    plt.fill_between(
        intervals_15.index,
        intervals_15["min"],
        intervals_15["max"],
        alpha=0.5,
        color="orange",
    )
    plt.title("Latency Costs (15-sec interval)")
    plt.xlabel("Connection ID")
    plt.ylabel("Latency Costs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/{save_path}-min-max-mean-15.{FIGURE_TYPE}")

    return latency_costs_by_connection


def path_stability(
    paths_1: pd.DataFrame,
    paths_15: pd.DataFrame = None,
    paths_30: pd.DataFrame = None,
    save_path: str = "path-stability",
) -> dict[str, pd.DataFrame]:
    ret = {}
    for gs in SERVER_GS:
        relevant_1 = paths_1.loc[(paths_1["target"] == gs) & (paths_1["source"] != gs)]
        relevant_15 = paths_15.loc[
            (paths_15["target"] == gs) & (paths_15["source"] != gs)
        ]
        relevant_30 = paths_30.loc[
            (paths_30["target"] == gs) & (paths_30["source"] != gs)
        ]

        relevant_1["path_tuple"] = relevant_1["path"].apply(tuple)
        relevant_15["path_tuple"] = relevant_15["path"].apply(tuple)
        relevant_30["path_tuple"] = relevant_30["path"].apply(tuple)
        unique_paths_1 = relevant_1.groupby("source")["path_tuple"].nunique()
        unique_paths_15 = relevant_15.groupby("source")["path_tuple"].nunique()
        unique_paths_30 = relevant_30.groupby("source")["path_tuple"].nunique()

        stability_metrics = pd.DataFrame(
            {
                "source": unique_paths_1.index.get_level_values("source"),
                "target": gs,
                "path_changes_1_sec": unique_paths_1.values,
                "path_changes_15_sec": unique_paths_15.values,
                "path_changes_30_sec": unique_paths_30.values,
                "path_stability_15_sec": unique_paths_15.values / unique_paths_1.values,
                "path_stability_30_sec": unique_paths_30.values / unique_paths_1.values,
                "avg_path_duration_1_sec": TOTAL_STEPS / unique_paths_1.values,
                "avg_path_duration_15_sec": TOTAL_STEPS / unique_paths_15.values,
                "avg_path_duration_30_sec": TOTAL_STEPS / unique_paths_30.values,
            }
        )
        coords = stability_metrics["source"].apply(extract_coords)
        stability_metrics[["latitude", "longitude"]] = coords.to_list()
        ret[gs] = stability_metrics

        plot_cdf(
            [unique_paths_1, unique_paths_15, unique_paths_30],
            title=f"CDF for the number of unique paths to {gs} by source",
            x_label="Number of unique paths",
            save_path=f"{save_path}-unique-paths-to-{gs}-combined",
            percentiles=[0.25, 0.5, 0.75, 0.95],
            precision=0,
        )
        plot_cdf(
            data=[
                stability_metrics["path_stability_15_sec"],
                stability_metrics["path_stability_30_sec"],
            ],
            title=f"CDF for the path stability to {gs}",
            x_label="Path Stability",
            save_path=f"{save_path}-ratio-{gs}-combined",
            percentiles=[0.25, 0.5, 0.75, 0.95],
            colors=["darkorange", "darkgreen"],
            labels=["15-sec interval", "30-sec interval"],
        )
        plot_cdf(
            data=[
                stability_metrics["avg_path_duration_1_sec"],
                stability_metrics["avg_path_duration_15_sec"],
                stability_metrics["avg_path_duration_30_sec"],
            ],
            title=f"CDF for the average path duration to {gs}",
            x_label="Average Path Duration in seconds",
            save_path=f"{save_path}-avg-path-duration-{gs}-combined",
            percentiles=[0.25, 0.5, 0.75],
        )
        # Filter stability metrics for histogram to not distort the plot with gs 25,0 or the server gs itself
        filtered = stability_metrics[
            (stability_metrics["source"] != "gs_25_0")
            & (stability_metrics["source"] != gs)
        ]
        plot_hist(
            data_x=filtered["longitude"],
            x_label="Longitude",
            data_y=filtered["latitude"],
            y_label="Latitude",
            weight=filtered["path_stability_15_sec"],
            weight_label="Path Stability",
            title=f"Path Stability Distribution for Server-GS {gs} and 15-sec intervals",
            save_path=f"{save_path}-stability-hist-{gs}-15",
        )
        plot_hist(
            data_x=filtered["longitude"],
            x_label="Longitude",
            data_y=filtered["latitude"],
            y_label="Latitude",
            weight=filtered["path_stability_30_sec"],
            weight_label="Path Stability",
            title=f"Path Stability Distribution for Server-GS {gs} and 30-sec intervals",
            save_path=f"{save_path}-stability-hist-{gs}-30",
        )
    return ret


def gsl_switches_in_interval(df, save_path: str = "gsl-switches"):
    df["first_hop"] = df["path"].apply(lambda x: x[0] if len(x) > 0 else -1).astype(int)
    df = df[df["first_hop"] != -1]
    gsl_switches = (
        df.groupby(["interval", "source"])["first_hop"]
        .nunique()
        .reset_index(name="gsl_switches")
    )

    # Filter sources where one gsl could be maintained over the whole interval
    gsl_switches = gsl_switches[gsl_switches["gsl_switches"] > 1]
    by_interval = gsl_switches.groupby(["interval"]).sum()
    by_source = gsl_switches.groupby(["source"]).sum()

    plot_cdf(
        [by_interval["gsl_switches"]],
        title="CDF for GSL switches by Interval",
        x_label="Number of GSL Switches",
        y_label="Proportion of Intervals",
        save_path=f"{save_path}-by-interval",
        bins=TOTAL_INTERVALS // 10,
    )

    # Bar plot by source
    plot_data = by_source.reset_index().sort_values("gsl_switches")
    unique_heights = plot_data["gsl_switches"].unique()
    plt.figure(figsize=(20, 8))
    bars = plt.bar(plot_data["source"], plot_data["gsl_switches"])
    # Bars with the same height get the same color
    color_map = dict(
        zip(unique_heights, sns.color_palette("Blues", len(unique_heights)))
    )
    for bar in bars:
        height = bar.get_height()
        bar.set_color(color_map[height])

    plt.title("Number of GSL Switches by Source Node", pad=20)
    plt.xlabel("Source Node ID")
    plt.ylabel("Total Number of GSL Switches")
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha="right", fontsize=6)
    # Add legend for colors
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=color_map[height])
        for height in unique_heights
    ]
    plt.legend(
        legend_elements,
        [f"{height:.0f} switches" for height in unique_heights],
        title="Number of Switches",
        loc="upper left",
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/{save_path}-by-source-{INTERVAL_LENGTH}.{FIGURE_TYPE}")
    plt.close()


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
    sns.histplot(series, stat="proportion", cumulative=True, alpha=0.25)
    sns.ecdfplot(series, stat="proportion")
    if save_path:
        plt.savefig(f"plots/{save_path}.{FIGURE_TYPE}")
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
        plt.savefig(f"plots/{save_path}.{FIGURE_TYPE}")
    plt.close()


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
        plt.savefig(f"plots/{save_path}.{FIGURE_TYPE}")
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
        plt.savefig(f"plots/{save_path}.{FIGURE_TYPE}")
    plt.close()


def plot_cdf(
    data: list[pd.Series],
    title: str,
    x_label: str,
    y_label: str = "Proportion",
    labels: list[str] = ["1-sec interval", "15-sec interval", "30-sec interval"],
    stat: str = "proportion",
    save_path: str = None,
    bins: int = "auto",
    percentiles: list[float] = [0.25, 0.5, 0.75],
    precision: int = 2,
    marker_at: list[float] = [],
    colors: list[str] = ["blue", "darkorange", "green"],
    bars: bool = False,
):
    plt.figure(figsize=(12, 8))
    for series, label, color, offset in zip(data, labels, colors, [-0.05, -0.05, -0.1]):
        ax = sns.ecdfplot(data=series, stat=stat, label=label, color=color)
        if bars:
            sns.histplot(
                data=series,
                stat=stat,
                alpha=0.4,
                cumulative=True,
                label=label,
                bins=bins,
                color=color,
            )

        quantiles = series.quantile(percentiles)
        for idx, q in enumerate(quantiles):
            ax.axvline(x=q, color=color, linestyle="--", alpha=0.4)
            ax.text(
                q,
                percentiles[idx] + offset,
                f"{int(percentiles[idx]*100)}\%\n({q:.{precision}f})",
                horizontalalignment="center",
                color=color,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5),
            )

        for marker in marker_at:
            c = color.strip("dark") if "dark" in color else f"light{color}"
            ax.axvline(x=marker, color=c, linestyle="--", alpha=0.4)

    plt.legend()

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(f"plots/{save_path}.{FIGURE_TYPE}")
    plt.close()


def calc_path_length(path):
    return len(path + 2)


def extract_coords(source):
    return extract_latitude(source), extract_longitude(source)


def extract_latitude(source):
    return int(source.split("_")[1])


def extract_longitude(source):
    return int(source.split("_")[2])


def load_all_timesteps(interval: int, data_type: str):
    return pd.read_pickle(f"{interval}-sec/{data_type}.pckl")


def load_all_timesteps_again(data_type):
    dfs = np.empty(TOTAL_STEPS)
    for time in range(TOTAL_STEPS):
        dfs[time] = load_individual_timestep(data_type, time)
    return pd.concat(dfs, ignore_index=True)


def load_individual_timestep(data_type, time):
    return pd.read_pickle(f"{data_type}/{time}.pckl")


if __name__ == "__main__":
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times"],
        }
    )
    plt.rcParams["font.size"] = 18
    # for gs in SERVER_GS:
    #     stability_per_gs[gs].to_csv(f"debug/stability-{gs}.csv")
    # latency_costs_by_connection.to_csv("debug/latency-costs.csv")
    stability_per_gs = {}
    for gs in SERVER_GS:
        stability_per_gs[gs] = pd.read_csv(f"debug/stability-{gs}.csv")
    latency_costs_by_connection = pd.read_csv("debug/latency-costs.csv")
    plot_stability_vs_latency_costs(stability_per_gs, latency_costs_by_connection)
    # main()
    # paths_df_15 = load_all_timesteps(15, "paths")
    # paths_df_15["interval"] = paths_df_15["time"] // 15
    # durations_15 = gsl_duration(paths_df_15)
    # durations_15.to_csv("debug/gsl-duration-15.csv")
    # durations_15 = pd.read_csv("debug/gsl-duration-15.csv")
    # durations_30 = pd.read_csv("debug/gsl-duration.csv")

    # plt.figure(figsize=(12, 8))
    # sns.ecdfplot(
    #     data=durations_15["duration"],
    #     label="15-sec interval",
    #     color="blue",
    # )
    # sns.ecdfplot(
    #     data=durations_30["duration"],
    #     label="30-sec interval",
    #     color="orange",
    # )
    # plt.title("CDF for GSL Duration within an interval")
    # plt.xlabel("GSL Duration")
    # plt.ylabel("Proportion")
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(f"plots/gsl-duration-cdf.{FIGURE_TYPE}")
    # plt.close()
