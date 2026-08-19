# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch the wheel-vendored Pi npm distribution with packaged Node.js."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Run the pinned Pi CLI without resolving Node.js or Pi from ``PATH``."""

    from nodejs_wheel import node

    cli = (
        Path(__file__).parent
        / "_vendor"
        / "node_modules"
        / "@earendil-works"
        / "pi-coding-agent"
        / "dist"
        / "cli.js"
    )
    if not cli.is_file():
        raise SystemExit(
            "The packaged Pi runtime is unavailable; reinstall "
            "nemo-fabric-pi-cli-bin from a wheel"
        )
    completed = node(
        [str(cli), *sys.argv[1:]],
        return_completed_process=True,
    )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
