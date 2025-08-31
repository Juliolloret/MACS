#!/usr/bin/env python3
"""Command line interface for running the MACS workflow."""

from __future__ import annotations

import argparse
import os
from typing import List

from multi_agent_llm_system import run_project_orchestration
from utils import load_app_config


def _collect_pdf_paths(directories: List[str]) -> List[str]:
    """Gather all PDF files from the provided directories."""
    pdf_paths: List[str] = []
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for root, _dirs, files in os.walk(directory):
            for name in files:
                if name.lower().endswith(".pdf"):
                    pdf_paths.append(os.path.join(root, name))
    return pdf_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MACS project orchestration")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to configuration file.",
    )
    parser.add_argument(
        "--pdf-dir",
        nargs="+",
        help="One or more directories containing PDF files.",
    )
    parser.add_argument(
        "--experimental-data",
        type=str,
        default="",
        help="Optional path to experimental data file.",
    )
    parser.add_argument(
        "--out-dir",
        help="Directory where outputs will be written.",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Run adaptive orchestration cycle instead of single pass.",
    )
    parser.add_argument(
        "--list-plugins",
        action="store_true",
        help="List available agents and plugins and exit.",
    )
    args = parser.parse_args()

    if args.list_plugins:
        from agents.registry import load_agents, load_plugins, list_plugins

        load_agents()
        load_plugins()
        for info in list_plugins():
            author = f" by {info['author']}" if info["author"] else ""
            print(f"{info['name']} ({info['version']}){author}")
        return

    if not args.pdf_dir or not args.out_dir:
        print("--pdf-dir and --out-dir are required unless --list-plugins is provided.")
        return

    pdf_file_paths = _collect_pdf_paths(args.pdf_dir)
    if not pdf_file_paths:
        print("No PDF files found in specified directories.")
        return

    if args.adaptive:
        from adaptive.adaptive_graph_runner import adaptive_cycle
        from adaptive.evaluation import evaluate_target_function

        inputs = {
            "initial_inputs": {
                "all_pdf_paths": pdf_file_paths,
                "experimental_data_file_path": args.experimental_data,
            },
            "project_base_output_dir": args.out_dir,
        }
        adaptive_cycle(
            config_path=args.config,
            inputs=inputs,
            evaluate_fn=evaluate_target_function,
            threshold=1.0,
            max_steps=5,
        )
        return

    app_config = load_app_config(config_path=args.config)
    if not app_config:
        print(f"Failed to load configuration from '{args.config}'.")
        return

    run_project_orchestration(
        pdf_file_paths=pdf_file_paths,
        experimental_data_path=args.experimental_data,
        project_base_output_dir=args.out_dir,
        status_update_callback=print,
        app_config=app_config,
    )


if __name__ == "__main__":
    main()
