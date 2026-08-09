#!/usr/bin/env python3
# Copyright 2025 Dimensional Inc.
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

"""
Force-Torque Module Test/Deployment Script

Deploys and connects the FT driver, visualizer and logger modules using Dimos.
Uses LCM transport for Vector3 force and torque messages.
Force/torque samples are recorded to a SQLite .db file.
"""

import time
import argparse
from pathlib import Path

from dimos.core import start, LCMTransport, pLCMTransport
from dimos.utils.logging_config import setup_logger
from dimos.msgs.geometry_msgs import Vector3
from dimos.hardware.ft_driver_module import FTDriverModule, RawSensorData
from dimos.hardware.ft_logger_module import FTLoggerModule
from dimos.hardware.ft_visualizer_module import FTVisualizerModule

logger = setup_logger(__name__)


def main():
    """Main deployment function for FT sensor modules."""
    parser = argparse.ArgumentParser(
        description="Deploy Force-Torque sensor driver and visualizer modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python ft_module_test.py

  # Run with calibration file
  python ft_module_test.py --calibration ft_calibration.json

  # Run with custom serial port and verbose output
  python ft_module_test.py --port /dev/ttyUSB0 --calibration ft_cal.json --verbose

  # Run with custom dashboard port
  python ft_module_test.py --dash-port 8080 --calibration ft_calibration.npz

  # Write the recording to a specific database file, including raw sensor values
  python ft_module_test.py --db ft_logs/run1.db --db-raw

  # Run without recording
  python ft_module_test.py --no-db
        """,
    )

    # Driver arguments
    parser.add_argument(
        "--port",
        default="/dev/ttyACM0",
        help="Serial port for sensor (default: /dev/ttyACM0)",
    )
    parser.add_argument(
        "--baud", type=int, default=115200, help="Serial baud rate (default: 115200)"
    )
    parser.add_argument(
        "--window", type=int, default=3, help="Moving average window size (default: 3)"
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default="dimos/hardware/ft_calibration.json",
        help="Path to calibration file (default: dimos/hardware/ft_calibration.json)",
    )

    # Visualizer arguments
    parser.add_argument(
        "--dash-port", type=int, default=8052, help="Port for Dash web server (default: 8052)"
    )
    parser.add_argument(
        "--dash-host", default="0.0.0.0", help="Host for Dash web server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--history", type=int, default=500, help="Max history points to keep (default: 500)"
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=100,
        help="Dashboard update interval in ms (default: 100)",
    )

    # Logger arguments
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="SQLite database file to record to (default: ft_logs/ft_log_<timestamp>.db)",
    )
    parser.add_argument("--no-db", action="store_true", help="Disable recording to a database file")
    parser.add_argument(
        "--db-raw", action="store_true", help="Also record raw sensor values to the database"
    )
    parser.add_argument(
        "--db-flush-interval",
        type=float,
        default=1.0,
        help="Seconds between database commits (default: 1.0)",
    )

    # LCM transport arguments
    parser.add_argument(
        "--lcm-force-channel",
        default="/ft/force",
        help="LCM channel for force Vector3 data (default: /ft/force)",
    )
    parser.add_argument(
        "--lcm-torque-channel",
        default="/ft/torque",
        help="LCM channel for torque Vector3 data (default: /ft/torque)",
    )
    parser.add_argument(
        "--lcm-raw-channel",
        default="/ft/raw_sensors",
        help="LCM channel for raw sensor data (default: /ft/raw_sensors)",
    )

    # General arguments
    parser.add_argument(
        "--processes", type=int, default=3, help="Number of Dimos processes (default: 3)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--no-visualizer", action="store_true", help="Run driver only, without visualizer"
    )
    parser.add_argument("--no-raw", action="store_true", help="Don't publish raw sensor data")

    args = parser.parse_args()

    # Resolve the database path (absolute, so worker processes agree on it)
    db_path = None
    if not args.no_db:
        db_path = Path(
            args.db or f"ft_logs/ft_log_{time.strftime('%Y%m%d_%H%M%S')}.db"
        ).expanduser()
        db_path = db_path if db_path.is_absolute() else Path.cwd() / db_path

    # Check if calibration file exists if specified
    if args.calibration:
        cal_path = Path(args.calibration)
        if not cal_path.exists():
            logger.warning(f"Calibration file {cal_path} not found")
            logger.warning("Will run without calibration (raw sensor values only)")
            args.calibration = None

    # Start Dimos
    logger.info("=" * 60)
    logger.info("Force-Torque Sensor Module Deployment")
    logger.info("=" * 60)
    logger.info(f"Starting Dimos with {args.processes} processes...")
    dimos = start(args.processes)

    # Deploy FT driver module
    logger.info("Deploying FT driver module...")
    logger.info(f"  Serial port: {args.port}")
    logger.info(f"  Baud rate: {args.baud}")
    logger.info(f"  Moving average window: {args.window}")
    logger.info(f"  Calibration file: {args.calibration or 'None (raw data only)'}")

    driver = dimos.deploy(
        FTDriverModule,
        serial_port=args.port,
        baud_rate=args.baud,
        window_size=args.window,
        calibration_file=args.calibration,
        verbose=args.verbose,
    )
    logger.info("Driver deployment complete")

    # Set up LCM transport for driver outputs
    logger.info("Setting up LCM transports...")

    # Force and torque use proper LCMTransport with Vector3 type
    driver.force.transport = LCMTransport(args.lcm_force_channel, Vector3)
    logger.info(f"  Force Vector3 channel: {args.lcm_force_channel}")

    driver.torque.transport = LCMTransport(args.lcm_torque_channel, Vector3)
    logger.info(f"  Torque Vector3 channel: {args.lcm_torque_channel}")

    # Raw sensor data (optional) uses pLCMTransport since it's a custom dataclass
    if not args.no_raw:
        driver.raw_sensor_data.transport = pLCMTransport(args.lcm_raw_channel)
        logger.info(f"  Raw sensor data channel: {args.lcm_raw_channel}")

    # Deploy logger module (records to a SQLite .db file)
    ft_logger = None
    if db_path:
        log_raw = args.db_raw and not args.no_raw
        if args.db_raw and args.no_raw:
            logger.warning("--db-raw ignored because --no-raw disables raw sensor publishing")

        logger.info("Deploying FT logger module...")
        logger.info(f"  Database file: {db_path}")
        logger.info(f"  Flush interval: {args.db_flush_interval}s")
        logger.info(f"  Logging raw sensor values: {log_raw}")

        ft_logger = dimos.deploy(
            FTLoggerModule,
            db_path=str(db_path),
            flush_interval=args.db_flush_interval,
            log_raw=log_raw,
            metadata={
                "serial_port": args.port,
                "baud_rate": args.baud,
                "window_size": args.window,
                "calibration_file": args.calibration or "none",
            },
            verbose=args.verbose,
        )

        # Connect logger inputs to driver outputs
        ft_logger.force.connect(driver.force)
        ft_logger.torque.connect(driver.torque)
        if log_raw:
            ft_logger.raw_sensor_data.connect(driver.raw_sensor_data)
        logger.info("  Connected to driver output streams")

        if not args.calibration:
            logger.warning("Without calibration the driver publishes no force/torque data")
            logger.warning("  Use --db-raw to record raw sensor values instead")

    # Deploy visualizer if requested
    visualizer = None
    if not args.no_visualizer and args.calibration:
        logger.info("Deploying FT visualizer module...")
        logger.info(f"  Dashboard port: {args.dash_port}")
        logger.info(f"  Dashboard host: {args.dash_host}")
        logger.info(f"  History points: {args.history}")
        logger.info(f"  Update interval: {args.update_interval}ms")
        logger.warning("Note: Visualizer may have issues in multiprocess environment")
        logger.warning("  Consider using --no-visualizer flag to disable if not needed")

        visualizer = dimos.deploy(
            FTVisualizerModule,
            max_history=args.history,
            update_interval_ms=args.update_interval,
            dash_port=args.dash_port,
            dash_host=args.dash_host,
            verbose=args.verbose,
        )

        # Connect visualizer inputs to driver outputs
        visualizer.force.connect(driver.force)
        visualizer.torque.connect(driver.torque)
        logger.info(f"  Connected to force and torque streams")

    elif not args.no_visualizer and not args.calibration:
        logger.warning("Visualizer requires calibration file to run")
        logger.warning("  Please provide a calibration file with --calibration flag")

    # Start modules
    logger.info("=" * 60)
    logger.info("Starting modules...")
    logger.info("=" * 60)

    # Start logger before the driver so no samples are missed
    if ft_logger:
        if ft_logger.start():
            logger.info(f"Recording to {db_path}")
        else:
            logger.error(f"Logger failed to start - no data will be recorded to {db_path}")
            ft_logger = None

    # Start driver
    if not driver.start():
        logger.error("CRITICAL: FT driver failed to start - no data will be published!")
        logger.error("Check that:")
        logger.error(f"  1. Serial port {args.port} exists and is accessible")
        logger.error("  2. No other process is using the serial port")
        logger.error("  3. You have permission to access the serial port")
        logger.error("  4. The sensor is connected and powered on")
        logger.info(f"\nTry running: ls -la {args.port}")
        logger.info("If it is owned by the 'dialout' group, either start a new login shell")
        logger.info(f"  after 'sudo usermod -aG dialout $USER', or: sudo chmod 666 {args.port}")

        if ft_logger:
            ft_logger.stop()
        dimos.shutdown()
        return

    # Start visualizer
    if visualizer:
        visualizer.start()
        logger.info(
            f"Dashboard running at http://{'127.0.0.1' if args.dash_host == '0.0.0.0' else args.dash_host}:{args.dash_port}"
        )

    logger.info("All modules started successfully!")
    logger.info("Press Ctrl+C to stop...")

    # Main loop - print statistics periodically
    try:
        last_print_time = time.time()
        while True:
            time.sleep(1)

            # Print stats every 10 seconds
            if time.time() - last_print_time > 10:
                driver_stats = driver.get_stats()
                logger.info(
                    f"Driver Stats: Messages={driver_stats['message_count']}, "
                    f"Errors={driver_stats['error_count']}, "
                    f"Calibrated={driver_stats['calibrated_count']}, "
                    f"Calibration={'Yes' if driver_stats['calibration_loaded'] else 'No'}"
                )

                if driver_stats["calibration_loaded"]:
                    logger.info(
                        f"  Latest |F|={driver_stats['latest_force_magnitude']:.2f} N, "
                        f"|T|={driver_stats['latest_torque_magnitude']:.4f} N⋅m"
                    )

                if visualizer:
                    viz_stats = visualizer.get_stats()
                    logger.info(
                        f"Visualizer Stats: Force msgs={viz_stats['force_count']}, "
                        f"Torque msgs={viz_stats['torque_count']}, "
                        f"Data points={viz_stats['data_points']}"
                    )

                if ft_logger:
                    log_stats = ft_logger.get_stats()
                    logger.info(
                        f"Logger Stats: Rows written={log_stats['written_count']}, "
                        f"Pending={log_stats['pending_rows']}, "
                        f"Force={log_stats['force_count']}, "
                        f"Torque={log_stats['torque_count']}, "
                        f"Raw={log_stats['raw_count']}"
                    )

                last_print_time = time.time()

    except KeyboardInterrupt:
        logger.info("=" * 60)
        logger.info("Shutting down...")
        logger.info("=" * 60)

        # Stop modules - driver first so the logger can drain the last samples
        driver.stop()
        if visualizer:
            visualizer.stop()
        if ft_logger:
            log_stats = ft_logger.get_stats()
            ft_logger.stop()
            logger.info(
                f"Recording saved to {db_path} "
                f"(force={log_stats['force_count']}, torque={log_stats['torque_count']}, "
                f"raw={log_stats['raw_count']})"
            )

        # Shutdown Dimos
        time.sleep(0.5)  # Give modules time to clean up
        dimos.shutdown()

        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
