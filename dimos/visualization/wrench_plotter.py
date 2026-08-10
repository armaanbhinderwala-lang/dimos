# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Live force/torque graphing for :class:`WrenchStamped` streams.

Fans each wrench out into six scalar entities -- force x/y/z and torque x/y/z --
and logs them on a shared timeline, which is what Rerun needs to draw them as
live line charts. The wrench itself has no ``to_rerun()`` because there is no
single archetype for it: a force/torque reading is six independent series, not
one spatial primitive, so the split has to happen somewhere and it happens here.

Entities land under ``entity_prefix`` (default ``ft``), deliberately outside the
Rerun bridge's ``world`` prefix so a 3D view never tries to render them.
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import Field
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In
from dimos.msgs.geometry_msgs.WrenchStamped import WrenchStamped
from dimos.utils.logging_config import setup_logger
from dimos.visualization.rerun.constants import RERUN_GRPC_PORT
from dimos.visualization.rerun.init import rerun_init

if TYPE_CHECKING:
    from rerun.blueprint import Blueprint

logger = setup_logger()

# Conventional axis colouring, matching the RGB=XYZ convention Rerun uses for
# 3D axes so the charts read the same way as the spatial views.
AXIS_COLORS: dict[str, list[int]] = {
    "x": [220, 50, 47],
    "y": [50, 200, 90],
    "z": [60, 130, 246],
}

# The two quantities carried by a wrench, with the units they are plotted in.
QUANTITIES: dict[str, str] = {"force": "N", "torque": "Nm"}


def wrench_view_blueprint(
    entity_prefix: str = "ft",
    channels: tuple[str, ...] = ("ext_wrench", "raw_wrench"),
) -> Blueprint:
    """A grid of time-series views: one row per channel, force beside torque."""
    import rerun.blueprint as rrb

    rows = [
        rrb.Horizontal(
            *(
                rrb.TimeSeriesView(
                    origin=f"{entity_prefix}/{channel}/{quantity}",
                    name=f"{channel} {quantity} ({unit})",
                )
                for quantity, unit in QUANTITIES.items()
            )
        )
        for channel in channels
    ]
    return rrb.Blueprint(rrb.Vertical(*rows), collapse_panels=True)


class WrenchPlotterConfig(ModuleConfig):
    """Configuration for :class:`WrenchPlotter`."""

    # Root entity path for every series this module logs.
    entity_prefix: str = "ft"

    # Rerun timeline the samples are stamped on. Using the sensor's own ts
    # (rather than log time) is what makes the x-axis reflect acquisition time.
    timeline: str = "ft_time"

    # Where to serve/find the Rerun gRPC server. rerun_init connects to this URL
    # if something is already listening -- e.g. a RerunBridgeModule in the same
    # stack -- and otherwise serves it here, so the plotter works either way.
    connect_url: str = Field(
        default_factory=lambda: f"rerun+http://127.0.0.1:{RERUN_GRPC_PORT}/proxy"
    )
    server_memory_limit: str = "4GB"

    # Spawn a native viewer on start. Turn off for headless runs; the data is
    # still reachable by pointing a viewer at connect_url.
    open_viewer: bool = True

    # Send the time-series layout. Disable when something else owns the layout,
    # since Rerun keeps only the most recently sent blueprint.
    send_blueprint: bool = True


class WrenchPlotter(Module):
    """Graphs both xArm FT streams live, six series each.

    In ports are named for :class:`XArmFTSensor`'s outputs so ``autoconnect``
    wires them without remappings, exactly as the FT recorder does.
    """

    config: WrenchPlotterConfig

    ext_wrench: In[WrenchStamped]
    raw_wrench: In[WrenchStamped]

    @rpc
    def start(self) -> None:
        super().start()
        import rerun as rr

        server_uri = rerun_init(
            "dimos_ft_plot",
            start_grpc=True,
            grpc_config={
                "connect_url": self.config.connect_url,
                "server_memory_limit": self.config.server_memory_limit,
            },
        )

        channels = tuple(self.inputs)
        if self.config.send_blueprint:
            rr.send_blueprint(wrench_view_blueprint(self.config.entity_prefix, channels))

        self._name_series(channels)

        if self.config.open_viewer:
            self._spawn_viewer(server_uri)

        # Sync subscribe rather than async handle_* on purpose: the async path
        # drops intermediate messages under load, which would silently punch
        # holes in the plotted signal.
        for channel in channels:
            port: In[WrenchStamped] = getattr(self, channel)
            self.register_disposable(
                Disposable(port.subscribe(partial(self._plot_wrench, channel)))
            )

        logger.info(
            "WrenchPlotter graphing %s under '%s/' (viewer: %s)",
            ", ".join(channels),
            self.config.entity_prefix,
            server_uri or self.config.connect_url,
        )

    def _name_series(self, channels: tuple[str, ...]) -> None:
        """Label and colour each series once, statically."""
        import rerun as rr

        for channel in channels:
            for quantity in QUANTITIES:
                for axis, color in AXIS_COLORS.items():
                    rr.log(
                        f"{self.config.entity_prefix}/{channel}/{quantity}/{axis}",
                        rr.SeriesLines(names=axis, colors=color, widths=1.5),
                        static=True,
                    )

    @staticmethod
    def _viewer_executable() -> str | None:
        """Absolute path to a viewer binary sitting beside this interpreter.

        ``rerun_bindings.spawn(executable_name=...)`` resolves through PATH,
        and PATH carries no venv binaries unless the venv was activated --
        launching ``.venv/bin/dimos`` directly is enough to break it, which is
        why the viewer silently never opened. Looking next to sys.executable
        finds it either way.
        """
        bin_dir = Path(sys.executable).parent
        for name in ("dimos-viewer", "rerun"):
            candidate = bin_dir / name
            if candidate.exists():
                return str(candidate)
        return None

    def _spawn_viewer(self, server_uri: str | None) -> None:
        """Open a native viewer pointed at our server, preferring dimos-viewer."""
        uri = server_uri or self.config.connect_url

        try:
            import rerun_bindings
        except ImportError:
            logger.warning("rerun_bindings unavailable; connect a viewer to %s", uri)
            return

        executable = self._viewer_executable()
        # --connect so the viewer joins our gRPC server instead of starting its
        # own, which would conflict over the port.
        kwargs = {
            "memory_limit": self.config.server_memory_limit,
            "extra_args": ["--connect", uri],
        }

        try:
            if executable:
                rerun_bindings.spawn(executable_path=executable, **kwargs)
            else:
                # Nothing beside the interpreter -- fall back to a PATH lookup.
                rerun_bindings.spawn(executable_name="dimos-viewer", **kwargs)
        except Exception:
            logger.warning(
                "Could not open a viewer automatically (headless?). Plot data is "
                "still served -- run: dimos-viewer --connect %s",
                uri,
                exc_info=True,
            )

    def _plot_wrench(self, channel: str, msg: WrenchStamped) -> None:
        """Split one wrench into its six scalar series."""
        import rerun as rr

        # Stamp with the sensor's time so the chart's x-axis is acquisition
        # time, not the time this callback happened to run.
        rr.set_time(self.config.timeline, timestamp=msg.ts)

        prefix = f"{self.config.entity_prefix}/{channel}"
        for quantity in QUANTITIES:
            vector = getattr(msg, quantity)
            for axis in AXIS_COLORS:
                rr.log(f"{prefix}/{quantity}/{axis}", rr.Scalars(float(getattr(vector, axis))))
