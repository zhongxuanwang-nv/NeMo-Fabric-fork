# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan or run the Pi JSONL-RPC code-review example."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from nemo_fabric import Fabric

from examples.pi_jsonl_rpc_code_review.config import ROOT
from examples.pi_jsonl_rpc_code_review.config import code_review_config


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi-executable", required=True, type=Path)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument(
        "--input",
        default=(
            "/skill:code-review Review calculator.py for correctness bugs. "
            "Return only actionable findings."
        ),
    )
    args = parser.parse_args()

    config = code_review_config(args.pi_executable)
    fabric = Fabric()
    if args.plan:
        output = fabric.plan(config, base_dir=ROOT)
    else:
        async with await fabric.start_runtime(config, base_dir=ROOT) as runtime:
            output = await runtime.invoke(input=args.input)
    print(json.dumps(output.to_mapping(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
