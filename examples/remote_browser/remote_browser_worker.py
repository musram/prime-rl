"""
Example: Remote Browser Worker for Forward Deployment.

This script demonstrates how to run a browser environment locally
while training happens on a remote PRIME-RL cluster.
"""

import asyncio
import argparse
from pathlib import Path

try:
    import gymnasium as gym
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False

from prime_rl.integrations import BrowserGymAdapter
from prime_rl.client import RemoteEnvironmentWorker
from loguru import logger


async def main():
    """Main function to run remote browser worker."""
    parser = argparse.ArgumentParser(description="Remote Browser Worker")
    parser.add_argument(
        "--server-url",
        type=str,
        required=True,
        help="Base URL of PRIME-RL Orchestrator API (e.g., http://localhost:8000)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication",
    )
    parser.add_argument(
        "--worker-id",
        type=str,
        default=None,
        help="Worker identifier (generated if not provided)",
    )
    parser.add_argument(
        "--use-remote-inference",
        action="store_true",
        help="Use server-side inference (Inference-as-a-Service mode)",
    )
    parser.add_argument(
        "--trace-buffer-size",
        type=int,
        default=10,
        help="Number of traces to buffer before sending",
    )
    parser.add_argument(
        "--policy-poll-interval",
        type=float,
        default=60.0,
        help="Interval in seconds to poll for policy updates",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to run",
    )
    
    args = parser.parse_args()
    
    # Create a dummy Gym environment (in production, would use actual browser env)
    if GYMNASIUM_AVAILABLE:
        try:
            env = gym.make("CartPole-v1")  # Using CartPole as example
        except Exception:
            logger.warning("CartPole not available, using mock environment")
            env = None
    else:
        env = None
    
    if env is None:
        # Create a mock environment for demonstration
        from unittest.mock import Mock
        mock_env = Mock()
        mock_env.observation_space = Mock()
        mock_env.action_space = Mock()
        mock_env.reset.return_value = ("obs", {})
        mock_env.step.return_value = ("obs", 1.0, False, False, {})
        mock_env.close = Mock()
        env = mock_env
    
    # Wrap environment with BrowserGymAdapter
    adapter = BrowserGymAdapter(env)
    
    # Create remote worker
    worker = RemoteEnvironmentWorker(
        environment=adapter,
        server_url=args.server_url,
        api_key=args.api_key,
        worker_id=args.worker_id,
        use_remote_inference=args.use_remote_inference,
        policy_poll_interval=args.policy_poll_interval,
        trace_buffer_size=args.trace_buffer_size,
    )
    
    try:
        # Start worker
        logger.info("Starting remote worker...")
        await worker.start()
        logger.info(f"Worker started: session_id={worker.session_id}")
        
        # Run episodes
        logger.info(f"Running {args.episodes} episodes...")
        for i in range(args.episodes):
            logger.info(f"Running episode {i+1}/{args.episodes}")
            rollout = await worker.run_episode(max_steps=10)
            logger.info(f"Episode {i+1} completed: reward={rollout.rewards[0]}")
        
        # Flush remaining traces
        logger.info("Flushing trace buffer...")
        await worker.flush_traces()
        
        logger.info("All episodes completed")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    finally:
        # Stop worker
        await worker.stop()
        logger.info("Worker stopped")


if __name__ == "__main__":
    asyncio.run(main())

