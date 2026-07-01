"""RSU deployment experiments using K-center approximation techniques.

This script packages a course-project experiment into a single reproducible
entry point. It includes:

1. A synthetic Euclidean-space comparison between Gonzalez greedy K-center
   and a random baseline.
2. A road-network experiment that uses OSMnx / OpenStreetMap data and compares
   Gonzalez greedy, greedy K-median, and a random baseline under network
   shortest-path distance.

The implementation is intentionally compact and honest about scope: it is an
algorithm experiment, not a production deployment system.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


DEFAULT_CENTER_LAT = 22.626
DEFAULT_CENTER_LON = 120.265


@dataclass
class SyntheticResult:
    algorithm: str
    k: int
    radius: float


@dataclass
class NetworkResult:
    algorithm: str
    k: int
    max_radius: float
    average_distance: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RSU deployment experiments based on K-center methods."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling and baseline selection.",
    )
    parser.add_argument(
        "--synthetic-points",
        type=int,
        default=100,
        help="Number of synthetic Euclidean points.",
    )
    parser.add_argument(
        "--synthetic-k",
        type=int,
        default=5,
        help="Number of centers for the synthetic experiment.",
    )
    parser.add_argument(
        "--network-k-values",
        type=int,
        nargs="+",
        default=[5, 10, 15],
        help="K values to evaluate in the road-network experiment.",
    )
    parser.add_argument(
        "--place-lat",
        type=float,
        default=DEFAULT_CENTER_LAT,
        help="Latitude of the road-network query center.",
    )
    parser.add_argument(
        "--place-lon",
        type=float,
        default=DEFAULT_CENTER_LON,
        help="Longitude of the road-network query center.",
    )
    parser.add_argument(
        "--network-distance-m",
        type=int,
        default=1500,
        help="Radius in meters for the OSMnx road-network query.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=120,
        help=(
            "Maximum number of connected graph nodes to keep for the network "
            "distance matrix. Smaller values keep all-pairs shortest paths "
            "tractable."
        ),
    )
    parser.add_argument(
        "--random-trials",
        type=int,
        default=5,
        help="Number of random-baseline repetitions for averaging.",
    )
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="Run only the synthetic experiment.",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Write CSV summaries under outputs/.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for optional CSV summaries and plots.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable matplotlib windows and plot file output.",
    )
    return parser.parse_args()


def generate_random_points(
    count: int, x_limit: float = 100.0, y_limit: float = 100.0
) -> np.ndarray:
    return np.random.rand(count, 2) * np.array([x_limit, y_limit])


def get_covering_radius(points: np.ndarray, centers: np.ndarray) -> float:
    if len(centers) == 0:
        return math.inf
    distances = np.linalg.norm(points[:, np.newaxis] - centers, axis=2)
    return float(np.max(np.min(distances, axis=1)))


def random_k_center(points: np.ndarray, k: int) -> np.ndarray:
    indices = np.random.choice(len(points), k, replace=False)
    return points[indices]


def gonzalez_k_center(points: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.empty((0, points.shape[1]))

    first_index = np.random.randint(0, len(points))
    centers = [points[first_index]]
    current_min_distances = np.linalg.norm(points - centers[0], axis=1)

    for _ in range(k - 1):
        farthest_index = int(np.argmax(current_min_distances))
        new_center = points[farthest_index]
        centers.append(new_center)
        distance_to_new_center = np.linalg.norm(points - new_center, axis=1)
        current_min_distances = np.minimum(
            current_min_distances, distance_to_new_center
        )

    return np.asarray(centers)


def run_synthetic_experiment(point_count: int, k: int) -> list[SyntheticResult]:
    points = generate_random_points(point_count)
    gonzalez_centers = gonzalez_k_center(points, k)
    random_centers = random_k_center(points, k)

    return [
        SyntheticResult(
            algorithm="gonzalez_greedy",
            k=k,
            radius=get_covering_radius(points, gonzalez_centers),
        ),
        SyntheticResult(
            algorithm="random_baseline",
            k=k,
            radius=get_covering_radius(points, random_centers),
        ),
    ]


def plot_synthetic_experiment(
    point_count: int, k: int, output_dir: Path | None = None
) -> None:
    points = generate_random_points(point_count)
    gonzalez_centers = gonzalez_k_center(points, k)
    random_centers = random_k_center(points, k)
    gonzalez_radius = get_covering_radius(points, gonzalez_centers)
    random_radius = get_covering_radius(points, random_centers)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.scatter(points[:, 0], points[:, 1], c="steelblue", alpha=0.6, label="Points")
    ax1.scatter(
        gonzalez_centers[:, 0],
        gonzalez_centers[:, 1],
        c="crimson",
        s=120,
        marker="*",
        label="Centers",
    )
    for center in gonzalez_centers:
        ax1.add_patch(
            plt.Circle(
                center,
                gonzalez_radius,
                color="crimson",
                fill=False,
                linestyle="--",
                alpha=0.3,
            )
        )
    ax1.set_title(f"Gonzalez greedy (r={gonzalez_radius:.2f})")
    ax1.set_aspect("equal")
    ax1.legend()

    ax2.scatter(points[:, 0], points[:, 1], c="steelblue", alpha=0.6, label="Points")
    ax2.scatter(
        random_centers[:, 0],
        random_centers[:, 1],
        c="darkgreen",
        s=90,
        marker="x",
        label="Centers",
    )
    for center in random_centers:
        ax2.add_patch(
            plt.Circle(
                center,
                random_radius,
                color="darkgreen",
                fill=False,
                linestyle="--",
                alpha=0.3,
            )
        )
    ax2.set_title(f"Random baseline (r={random_radius:.2f})")
    ax2.set_aspect("equal")
    ax2.legend()
    plt.tight_layout()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "synthetic_experiment.png", dpi=180, bbox_inches="tight")

    plt.show()


def build_projected_graph(center_lat: float, center_lon: float, distance_m: int):
    try:
        import osmnx as ox
    except ImportError as exc:
        raise SystemExit(
            "OSMnx is required for the road-network experiment. "
            "Install dependencies from requirements.txt first."
        ) from exc

    graph = ox.graph_from_point(
        (center_lat, center_lon), dist=distance_m, network_type="drive"
    )
    projected = ox.project_graph(graph)
    undirected = nx.Graph(projected)

    if not nx.is_connected(undirected):
        largest_component = max(nx.connected_components(undirected), key=len)
        undirected = undirected.subgraph(largest_component).copy()

    return ox, projected, undirected


def sample_graph_nodes(graph: nx.Graph, max_nodes: int) -> list[int]:
    nodes = list(graph.nodes)
    if len(nodes) <= max_nodes:
        return nodes
    return random.sample(nodes, max_nodes)


def compute_distance_matrix(graph: nx.Graph, node_ids: list[int]) -> np.ndarray:
    node_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    matrix = np.full((len(node_ids), len(node_ids)), np.inf)

    for source in node_ids:
        distances = nx.single_source_dijkstra_path_length(
            graph, source, weight="length"
        )
        source_index = node_to_index[source]
        for target, distance in distances.items():
            target_index = node_to_index.get(target)
            if target_index is not None:
                matrix[source_index, target_index] = distance

    return matrix


def get_covering_radius_matrix(
    distance_matrix: np.ndarray, center_indices: Iterable[int]
) -> float:
    center_indices = list(center_indices)
    if not center_indices:
        return math.inf
    distances_to_centers = distance_matrix[:, np.asarray(center_indices)]
    return float(np.max(np.min(distances_to_centers, axis=1)))


def gonzalez_k_center_matrix(distance_matrix: np.ndarray, k: int) -> list[int]:
    if k <= 0:
        return []

    first_center = np.random.randint(0, distance_matrix.shape[0])
    centers = [first_center]
    current_min_distances = distance_matrix[:, first_center].copy()

    for _ in range(k - 1):
        farthest_index = int(np.argmax(current_min_distances))
        centers.append(farthest_index)
        current_min_distances = np.minimum(
            current_min_distances, distance_matrix[:, farthest_index]
        )

    return centers


def greedy_k_median_matrix(distance_matrix: np.ndarray, k: int) -> list[int]:
    if k <= 0:
        return []

    sum_distances = np.sum(distance_matrix, axis=0)
    first_center = int(np.argmin(sum_distances))
    centers = [first_center]
    current_min_distances = distance_matrix[:, first_center].copy()

    for _ in range(k - 1):
        candidate_min_distances = np.minimum(
            current_min_distances[:, np.newaxis], distance_matrix
        )
        total_distances = np.sum(candidate_min_distances, axis=0)
        best_candidate = int(np.argmin(total_distances))
        centers.append(best_candidate)
        current_min_distances = np.minimum(
            current_min_distances, distance_matrix[:, best_candidate]
        )

    return centers


def evaluate_metrics(
    distance_matrix: np.ndarray, center_indices: Iterable[int]
) -> tuple[float, float]:
    center_indices = list(center_indices)
    if not center_indices:
        return math.inf, math.inf

    distances_to_centers = distance_matrix[:, np.asarray(center_indices)]
    min_distances = np.min(distances_to_centers, axis=1)
    return float(np.max(min_distances)), float(np.mean(min_distances))


def run_network_experiment(
    center_lat: float,
    center_lon: float,
    distance_m: int,
    max_nodes: int,
    k_values: list[int],
    random_trials: int,
) -> tuple[list[NetworkResult], dict[str, object]]:
    ox, projected_graph, connected_graph = build_projected_graph(
        center_lat=center_lat,
        center_lon=center_lon,
        distance_m=distance_m,
    )
    sampled_node_ids = sample_graph_nodes(connected_graph, max_nodes=max_nodes)
    distance_matrix = compute_distance_matrix(connected_graph, sampled_node_ids)
    gonzalez_centers_by_k: dict[int, list[int]] = {}

    results: list[NetworkResult] = []
    for k in k_values:
        gonzalez_centers = gonzalez_k_center_matrix(distance_matrix, k)
        gonzalez_centers_by_k[k] = gonzalez_centers
        gonzalez_max_radius, gonzalez_average = evaluate_metrics(
            distance_matrix, gonzalez_centers
        )
        results.append(
            NetworkResult(
                algorithm="gonzalez_greedy",
                k=k,
                max_radius=gonzalez_max_radius,
                average_distance=gonzalez_average,
            )
        )

        median_centers = greedy_k_median_matrix(distance_matrix, k)
        median_max_radius, median_average = evaluate_metrics(
            distance_matrix, median_centers
        )
        results.append(
            NetworkResult(
                algorithm="greedy_k_median",
                k=k,
                max_radius=median_max_radius,
                average_distance=median_average,
            )
        )

        random_radii = []
        random_averages = []
        for _ in range(random_trials):
            random_centers = np.random.choice(
                distance_matrix.shape[0], k, replace=False
            )
            max_radius, average_distance = evaluate_metrics(
                distance_matrix, random_centers
            )
            random_radii.append(max_radius)
            random_averages.append(average_distance)

        results.append(
            NetworkResult(
                algorithm="random_baseline",
                k=k,
                max_radius=float(np.mean(random_radii)),
                average_distance=float(np.mean(random_averages)),
            )
        )

    metadata = {
        "ox": ox,
        "projected_graph": projected_graph,
        "connected_graph": connected_graph,
        "sampled_node_ids": sampled_node_ids,
        "distance_matrix": distance_matrix,
        "gonzalez_centers_by_k": gonzalez_centers_by_k,
    }
    return results, metadata


def plot_network_map(
    ox_module,
    graph: nx.Graph,
    node_ids: list[int],
    center_indices: list[int],
    radius_m: float,
    output_dir: Path | None = None,
) -> None:
    center_node_ids = [node_ids[index] for index in center_indices]
    x_values = [graph.nodes[node]["x"] for node in center_node_ids]
    y_values = [graph.nodes[node]["y"] for node in center_node_ids]

    fig, ax = ox_module.plot_graph(
        graph,
        node_size=0,
        edge_color="#8a8a8a",
        edge_linewidth=0.5,
        show=False,
        close=False,
        figsize=(10, 10),
        bgcolor="white",
    )
    ax.scatter(
        x_values,
        y_values,
        c="crimson",
        s=180,
        marker="*",
        label="Selected RSU nodes",
        zorder=5,
    )
    plt.title(f"Network-distance Gonzalez solution (r={radius_m:.0f} m)")
    plt.legend()
    plt.tight_layout()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "network_map.png", dpi=180, bbox_inches="tight")

    plt.show()


def plot_network_comparison(
    results: list[NetworkResult], output_dir: Path | None = None
) -> None:
    grouped: dict[str, list[NetworkResult]] = {}
    for result in results:
        grouped.setdefault(result.algorithm, []).append(result)

    plt.figure(figsize=(10, 6))
    styles = {
        "gonzalez_greedy": ("crimson", "o", "Gonzalez greedy"),
        "greedy_k_median": ("royalblue", "s", "Greedy K-median"),
        "random_baseline": ("darkgreen", "x", "Random baseline"),
    }

    for algorithm, entries in grouped.items():
        entries = sorted(entries, key=lambda item: item.k)
        color, marker, label = styles[algorithm]
        plt.plot(
            [entry.k for entry in entries],
            [entry.max_radius for entry in entries],
            color=color,
            marker=marker,
            linewidth=2,
            label=label,
        )

    plt.title("Road-network comparison by RSU count")
    plt.xlabel("Number of RSUs (k)")
    plt.ylabel("Maximum service radius (meters)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            output_dir / "network_comparison.png", dpi=180, bbox_inches="tight"
        )

    plt.show()


def save_synthetic_results(results: list[SyntheticResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "synthetic_results.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["algorithm", "k", "radius"])
        for result in results:
            writer.writerow([result.algorithm, result.k, f"{result.radius:.6f}"])


def save_network_results(results: list[NetworkResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "network_results.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["algorithm", "k", "max_radius", "average_distance"])
        for result in results:
            writer.writerow(
                [
                    result.algorithm,
                    result.k,
                    f"{result.max_radius:.6f}",
                    f"{result.average_distance:.6f}",
                ]
            )


def print_synthetic_summary(results: list[SyntheticResult]) -> None:
    print("Synthetic Euclidean experiment")
    for result in results:
        print(
            f"  - {result.algorithm}: k={result.k}, "
            f"radius={result.radius:.2f}"
        )


def print_network_summary(results: list[NetworkResult], node_count: int) -> None:
    print("Road-network experiment")
    print(f"  - sampled nodes used for APSP: {node_count}")
    for result in sorted(results, key=lambda item: (item.k, item.algorithm)):
        print(
            f"  - {result.algorithm}: k={result.k}, "
            f"max_radius={result.max_radius:.2f} m, "
            f"avg_distance={result.average_distance:.2f} m"
        )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir) if args.save_results else None
    synthetic_results = run_synthetic_experiment(
        point_count=args.synthetic_points,
        k=args.synthetic_k,
    )
    print_synthetic_summary(synthetic_results)

    if output_dir is not None:
        save_synthetic_results(synthetic_results, output_dir)

    if not args.no_plots:
        plot_synthetic_experiment(
            point_count=args.synthetic_points,
            k=args.synthetic_k,
            output_dir=output_dir,
        )

    if args.skip_network:
        return

    network_results, metadata = run_network_experiment(
        center_lat=args.place_lat,
        center_lon=args.place_lon,
        distance_m=args.network_distance_m,
        max_nodes=args.max_nodes,
        k_values=args.network_k_values,
        random_trials=args.random_trials,
    )
    print_network_summary(network_results, node_count=len(metadata["sampled_node_ids"]))

    if output_dir is not None:
        save_network_results(network_results, output_dir)

    best_gonzalez = min(
        (result for result in network_results if result.algorithm == "gonzalez_greedy"),
        key=lambda item: item.k,
    )
    best_center_indices = metadata["gonzalez_centers_by_k"][best_gonzalez.k]

    if not args.no_plots:
        plot_network_map(
            ox_module=metadata["ox"],
            graph=metadata["connected_graph"],
            node_ids=metadata["sampled_node_ids"],
            center_indices=best_center_indices,
            radius_m=best_gonzalez.max_radius,
            output_dir=output_dir,
        )
        plot_network_comparison(network_results, output_dir=output_dir)


if __name__ == "__main__":
    main()

