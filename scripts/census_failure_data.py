#!/usr/bin/env python3
"""
census_failure_data.py — Census the RoboMIND failure dataset and perform
stratified random sampling for Experiment B.

Walks the failure_data directory, counts episodes per task and failure reason,
stratifies tasks into bins by (episode count × failure diversity), and samples
15 tasks with up to 10 trajectories each.

Usage
-----
  python scripts/census_failure_data.py \
      --data-dir /inspire/qb-ilm/project/robot-decision/public/datasets/robomind/RoboMIND/failure_data \
      --output-dir /inspire/qb-ilm/project/robot-decision/caijintian-p-caijintian/pyrmdp/experiment_output \
      --seed 42 \
      --num-tasks 15 \
      --max-episodes 10
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("census")


def census_failure_data(data_dir: Path) -> dict:
    """Walk failure_data and build a complete census."""
    tasks = {}

    for task_dir in sorted(data_dir.iterdir()):
        if not task_dir.is_dir():
            continue

        task_name = task_dir.name
        failure_reasons = {}
        total_episodes = 0

        for reason_dir in sorted(task_dir.iterdir()):
            if not reason_dir.is_dir():
                continue

            reason_name = reason_dir.name
            episodes = []

            for episode_dir in sorted(reason_dir.iterdir()):
                if not episode_dir.is_dir():
                    continue
                hdf5_path = episode_dir / "data" / "trajectory.hdf5"
                if hdf5_path.exists():
                    episodes.append({
                        "episode_id": episode_dir.name,
                        "hdf5_path": str(hdf5_path),
                    })

            if episodes:
                failure_reasons[reason_name] = {
                    "count": len(episodes),
                    "episodes": episodes,
                }
                total_episodes += len(episodes)

        if failure_reasons:
            tasks[task_name] = {
                "total_episodes": total_episodes,
                "num_failure_reasons": len(failure_reasons),
                "failure_reasons": failure_reasons,
            }

    return {
        "total_tasks": len(tasks),
        "total_episodes": sum(t["total_episodes"] for t in tasks.values()),
        "tasks": tasks,
    }


def _size_bin(n_episodes: int) -> str:
    if n_episodes <= 5:
        return "small"
    elif n_episodes <= 10:
        return "medium"
    else:
        return "large"


def _diversity_bin(n_reasons: int) -> str:
    if n_reasons <= 1:
        return "low"
    elif n_reasons <= 3:
        return "mid"
    else:
        return "high"


def stratified_sample(
    census: dict,
    num_tasks: int = 15,
    max_episodes: int = 10,
    seed: int = 42,
) -> dict:
    """Select tasks via stratified random sampling, then sample episodes."""
    rng = random.Random(seed)

    bins: dict[str, list[str]] = defaultdict(list)
    for task_name, info in census["tasks"].items():
        sb = _size_bin(info["total_episodes"])
        db = _diversity_bin(info["num_failure_reasons"])
        stratum = f"{sb}_{db}"
        bins[stratum].append(task_name)

    for v in bins.values():
        rng.shuffle(v)

    selected_tasks = {}
    occupied_bins = {k: list(v) for k, v in bins.items() if v}

    # Phase 1: one task from each occupied bin
    for stratum, pool in sorted(occupied_bins.items()):
        if pool and len(selected_tasks) < num_tasks:
            task_name = pool.pop(0)
            selected_tasks[task_name] = stratum

    # Phase 2: fill remaining slots proportionally
    remaining = num_tasks - len(selected_tasks)
    if remaining > 0:
        leftover_pool = []
        for stratum, pool in sorted(occupied_bins.items()):
            for t in pool:
                if t not in selected_tasks:
                    leftover_pool.append((t, stratum))
        rng.shuffle(leftover_pool)
        for t, stratum in leftover_pool[:remaining]:
            selected_tasks[t] = stratum

    # Phase 3: sample episodes per task
    result_tasks = {}
    total_sampled = 0
    for task_name, stratum in sorted(selected_tasks.items()):
        task_info = census["tasks"][task_name]
        all_episodes = []
        for reason_name, reason_info in task_info["failure_reasons"].items():
            for ep in reason_info["episodes"]:
                all_episodes.append({
                    "episode_id": ep["episode_id"],
                    "failure_reason": reason_name,
                    "hdf5_path": ep["hdf5_path"],
                })

        if len(all_episodes) <= max_episodes:
            sampled = all_episodes
        else:
            sampled = rng.sample(all_episodes, max_episodes)

        sampled.sort(key=lambda x: x["episode_id"])
        total_sampled += len(sampled)

        result_tasks[task_name] = {
            "stratum": stratum,
            "total_available": task_info["total_episodes"],
            "num_failure_reasons": task_info["num_failure_reasons"],
            "sampled_episodes": sampled,
        }

    return {
        "seed": seed,
        "sampling_protocol": (
            "stratified by episode count (small/medium/large) × "
            "failure diversity (low/mid/high); at least 1 task per "
            "occupied bin, remaining filled proportionally"
        ),
        "num_tasks": len(result_tasks),
        "max_episodes_per_task": max_episodes,
        "total_sampled_episodes": total_sampled,
        "stratum_distribution": {
            stratum: sum(1 for t in result_tasks.values() if t["stratum"] == stratum)
            for stratum in sorted(set(t["stratum"] for t in result_tasks.values()))
        },
        "tasks": result_tasks,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Census RoboMIND failure data and perform stratified sampling.",
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Root failure_data directory.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Where to save census.json and sampled_tasks.json.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-tasks", type=int, default=15)
    parser.add_argument("--max-episodes", type=int, default=10)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Censusing %s ...", data_dir)
    census = census_failure_data(data_dir)
    logger.info(
        "Census complete: %d tasks, %d total episodes",
        census["total_tasks"], census["total_episodes"],
    )

    # Print bin distribution
    bins: dict[str, int] = defaultdict(int)
    for task_info in census["tasks"].values():
        sb = _size_bin(task_info["total_episodes"])
        db = _diversity_bin(task_info["num_failure_reasons"])
        bins[f"{sb}_{db}"] += 1
    logger.info("Stratum distribution (all tasks):")
    for stratum in sorted(bins):
        logger.info("  %s: %d tasks", stratum, bins[stratum])

    census_path = output_dir / "census.json"
    census_path.write_text(
        json.dumps(census, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    logger.info("Saved census: %s", census_path)

    logger.info("Performing stratified sampling (seed=%d, %d tasks, max %d eps) ...",
                args.seed, args.num_tasks, args.max_episodes)
    sampled = stratified_sample(
        census,
        num_tasks=args.num_tasks,
        max_episodes=args.max_episodes,
        seed=args.seed,
    )
    logger.info(
        "Sampled: %d tasks, %d total episodes",
        sampled["num_tasks"], sampled["total_sampled_episodes"],
    )
    logger.info("Stratum distribution (sampled):")
    for stratum, count in sorted(sampled["stratum_distribution"].items()):
        logger.info("  %s: %d tasks", stratum, count)

    for task_name, info in sorted(sampled["tasks"].items()):
        logger.info(
            "  %s [%s]: %d/%d episodes",
            task_name, info["stratum"],
            len(info["sampled_episodes"]), info["total_available"],
        )

    sampled_path = output_dir / "sampled_tasks.json"
    sampled_path.write_text(
        json.dumps(sampled, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    logger.info("Saved sampled tasks: %s", sampled_path)


if __name__ == "__main__":
    main()
