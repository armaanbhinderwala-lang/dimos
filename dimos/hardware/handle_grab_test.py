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
Handle Grab Module Test/Deployment Script

Deploys and connects the RealSense camera module and handle grab skill module using Dimos.
"""

import time
import argparse
from dimos.core import start, LCMTransport
from dimos.utils.logging_config import setup_logger
from dimos.msgs.sensor_msgs import Image, CameraInfo
from dimos.hardware.realsense_module import RealsenseModule
from dimos.hardware.handle_grab_skill import HandleGrabModule
from dimos.agents2.agent import Agent
from dimos.agents2.cli.human import HumanInput

logger = setup_logger(__name__)


def main():
    """Main deployment function for handle grab system."""
    parser = argparse.ArgumentParser(
        description="Deploy RealSense camera and handle grab skill modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (simulation only)
  python handle_grab_test.py

  # Run with xARM robot
  python handle_grab_test.py --xarm 192.168.1.100

  # Run in test mode (get positions but don't move)
  python handle_grab_test.py --xarm 192.168.1.100 --test

  # Run with Qwen automatic detection and grab
  python handle_grab_test.py --xarm 192.168.1.100 --qwen --grab

  # Run with multiple iterations
  python handle_grab_test.py --xarm 192.168.1.100 --loop 3 --grab

  # Run without visualization (headless)
  python handle_grab_test.py --xarm 192.168.1.100 --no-visualization
        """,
    )

    # Module arguments
    parser.add_argument(
        "--fastsam-model",
        type=str,
        default="./weights/FastSAM-x.pt",
        help="Path to FastSAM model weights (default: ./weights/FastSAM-x.pt)",
    )
    parser.add_argument(
        "--xarm", type=str, default="192.168.1.210", help="xARM IP address (e.g., 192.168.1.100)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: get xARM positions but do not execute movements",
    )

    # Skill arguments (for default execution)
    parser.add_argument(
        "--qwen",
        action="store_true",
        help="Use Qwen vision model to automatically detect handle point",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=1,
        help="Number of times to repeat the detection and movement cycle (default: 1)",
    )
    parser.add_argument(
        "--grab", action="store_true", help="Execute grab sequence after positioning"
    )

    # RealSense camera arguments
    parser.add_argument("--width", type=int, default=1280, help="Camera width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Camera height (default: 720)")
    parser.add_argument("--fps", type=int, default=30, help="Camera frame rate (default: 30)")

    # LCM transport arguments
    parser.add_argument(
        "--lcm-color-channel",
        default="/camera/color_image",
        help="LCM channel for color image data (default: /camera/color_image)",
    )
    parser.add_argument(
        "--lcm-depth-channel",
        default="/camera/depth_image",
        help="LCM channel for depth image data (default: /camera/depth_image)",
    )
    parser.add_argument(
        "--lcm-info-channel",
        default="/camera/camera_info",
        help="LCM channel for camera info (default: /camera/camera_info)",
    )

    # Execution mode
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode with agent and human input",
    )
    parser.add_argument(
        "--auto-run", action="store_true", help="Automatically run the grab_handle skill on startup"
    )

    # System arguments
    parser.add_argument(
        "--processes", type=int, default=5, help="Number of Dimos processes (default: 3)"
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Run without visualization (useful for headless systems)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--xarm7", action="store_true", help="Use the 7-DOF xArm7 URDF/joint set instead of xArm6"
    )

    args = parser.parse_args()

    # Start Dimos
    logger.info("=" * 60)
    logger.info("Handle Grab System Deployment")
    logger.info("=" * 60)
    logger.info(f"Starting Dimos with {args.processes} processes...")
    dimos = start(args.processes)

    # Deploy RealSense module
    logger.info("Deploying RealSense camera module...")
    logger.info(f"  Resolution: {args.width}x{args.height}")
    logger.info(f"  FPS: {args.fps}")

    camera = dimos.deploy(
        RealsenseModule,
        width=args.width,
        height=args.height,
        fps=args.fps,
        verbose=args.verbose,
    )

    # Set up LCM transports for camera outputs
    camera.color_image.transport = LCMTransport(args.lcm_color_channel, Image)
    camera.depth_image.transport = LCMTransport(args.lcm_depth_channel, Image)
    camera.camera_info.transport = LCMTransport(args.lcm_info_channel, CameraInfo)

    logger.info("Camera LCM channels configured:")
    logger.info(f"  Color: {args.lcm_color_channel}")
    logger.info(f"  Depth: {args.lcm_depth_channel}")
    logger.info(f"  Info: {args.lcm_info_channel}")

    # Deploy handle grab module
    logger.info("Deploying handle grab module...")
    logger.info(f"  FastSAM model: {args.fastsam_model}")
    logger.info(f"  xARM IP: {args.xarm or 'None (simulation only)'}")
    logger.info(f"  Test mode: {'ON' if args.test else 'OFF'}")

    handle_grab = dimos.deploy(
        HandleGrabModule,
        fastsam_model_path=args.fastsam_model,
        xarm_ip=args.xarm,
        test_mode=args.test,
        num_arm_joints=7 if args.xarm7 else 6,
        urdf_filename="xarm7_openft_gripper.urdf" if args.xarm7 else "xarm6_openft_gripper.urdf",
    )

    # Connect handle grab inputs to camera outputs
    handle_grab.color_image.connect(camera.color_image)
    handle_grab.depth_image.connect(camera.depth_image)
    handle_grab.camera_info.connect(camera.camera_info)
    logger.info("Connected handle grab module to camera data streams (color, depth, camera_info)")

    # Start modules
    logger.info("=" * 60)
    logger.info("Starting modules...")
    logger.info("=" * 60)

    # Start RealSense camera
    camera.start()
    logger.info("RealSense camera started")

    # Start handle grab module
    handle_grab.start()
    logger.info("Handle grab module started")

    # Setup interactive mode if requested
    if args.interactive:
        logger.info("Setting up interactive agent mode...")

        # Deploy human input module
        human_input = dimos.deploy(HumanInput)

        # Deploy agent
        agent = dimos.deploy(
            Agent,
            system_prompt="""You are a helpful robotic assistant that can control a handle grabbing system.
            You have access to a skill called 'grab_handle' that can detect and grab handles on objects like microwaves.

            The skill accepts these parameters:
            - use_qwen: Use AI vision to detect handles automatically (boolean)
            - loop_count: Number of detection attempts (integer)
            - execute_grab: Actually close gripper after positioning (boolean)

            Be helpful and explain what you're doing when executing skills.""",
        )

        # Register skills
        agent.register_skills(handle_grab)
        agent.register_skills(human_input)

        # Start agent
        agent.run_implicit_skill("human")
        agent.start()

        logger.info("Interactive agent ready!")
        logger.info("You can now interact with the system through the agent.")
        logger.info("Example commands:")
        logger.info('  "Grab the handle using AI detection"')
        logger.info('  "Try to grab the handle 3 times"')
        logger.info('  "Position at the handle but dont grab"')

        # Keep agent running
        agent.loop_thread()

        while True:
            time.sleep(1)

    # Auto-run mode if requested
    elif args.auto_run:
        logger.info("Auto-running grab_handle skill...")

        # Wait a moment for data to start flowing
        time.sleep(2)

        # Execute the skill
        result = handle_grab.grab_handle(
            use_qwen=args.qwen, loop_count=args.loop, execute_grab=args.grab
        )

        logger.info(f"Skill result: {result}")

        # Keep running for a bit to allow cleanup
        time.sleep(5)

    # Default mode - if grab is requested, auto-run it
    else:
        # If --grab or --qwen was specified, automatically run the skill
        if args.grab or args.qwen:
            logger.info("Auto-running grab_handle skill (use --interactive for manual control)...")

            # Wait a moment for data to start flowing
            time.sleep(2)

            # Execute the skill
            result = handle_grab.grab_handle(
                use_qwen=args.qwen, loop_count=args.loop, execute_grab=args.grab
            )

            logger.info(f"Skill result: {result}")

            # Keep running for a bit to allow cleanup
            time.sleep(5)
        else:
            logger.info("Modules running. The grab_handle skill is available for RPC calls.")
            logger.info("Press Ctrl+C to stop...")

            try:
                # Main loop - print statistics periodically
                last_print_time = time.time()
                while True:
                    time.sleep(1)

                    # Print stats every 10 seconds
                    if time.time() - last_print_time > 10:
                        stats = camera.get_stats()
                        logger.info(f"Camera stats: {stats}")
                        last_print_time = time.time()

            except KeyboardInterrupt:
                logger.info("\n" + "=" * 60)
                logger.info("Shutting down...")
                logger.info("=" * 60)

    # Cleanup
    if not args.interactive:
        # Stop modules
        camera.stop()
        handle_grab.cleanup()

        # Shutdown Dimos
        time.sleep(0.5)
        dimos.shutdown()

        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
