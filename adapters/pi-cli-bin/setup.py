# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the pinned npm-distributed Pi runtime into release wheels."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    """Vendor Pi's shrinkwrap-locked production tree into non-editable wheels."""

    def run(self) -> None:
        super().run()
        if getattr(self, "editable_mode", False):
            return

        project_root = Path(__file__).parent.resolve()
        with tempfile.TemporaryDirectory(prefix="nemo-fabric-pi-build-") as temporary:
            npm_root = Path(temporary)
            shutil.copy2(project_root / "npm/package.json", npm_root / "package.json")
            shutil.copy2(
                project_root / "npm/package-lock.json",
                npm_root / "package-lock.json",
            )
            environment = {
                **os.environ,
                "NPM_CONFIG_AUDIT": "false",
                "NPM_CONFIG_FUND": "false",
                "NPM_CONFIG_UPDATE_NOTIFIER": "false",
            }
            subprocess.run(
                [
                    "npm",
                    "ci",
                    "--omit=dev",
                    "--omit=optional",
                    "--ignore-scripts",
                    "--silent",
                ],
                cwd=npm_root,
                env=environment,
                check=True,
            )
            node_modules = npm_root / "node_modules"
            for path in node_modules.rglob("*"):
                if path.is_file() and path.name.endswith((".d.mts", ".d.ts", ".map")):
                    path.unlink()
            pi_package = node_modules / "@earendil-works" / "pi-coding-agent"
            shutil.rmtree(pi_package / "docs")
            shutil.rmtree(pi_package / "examples")
            destination = (
                Path(self.build_lib) / "nemo_fabric_pi_cli_bin/_vendor/node_modules"
            )
            shutil.copytree(node_modules, destination)


setup(cmdclass={"build_py": BuildPy})
