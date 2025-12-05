"""
Redis-based task queue for distributed orchestration.

This module implements a high-throughput task queue using Redis to decouple
the Orchestrator loop from worker execution, enabling massive parallelism.
"""

import json
import uuid
from typing import Any, Dict, List, Optional
import asyncio

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    try:
        import redis
        REDIS_AVAILABLE = False
        aioredis = None
    except ImportError:
        REDIS_AVAILABLE = False
        redis = None  # type: ignore
        aioredis = None  # type: ignore

from loguru import logger


class RedisQueue:
    """
    Redis-based task queue for distributed environment stepping.
    
    Provides a high-throughput task queue for pushing environment steps
    and popping observations, decoupling the Orchestrator loop from worker
    execution. This enables scaling to 10,000+ concurrent environment instances.
    
    Queue Structure:
    - Task queue: `prime_rl:queue:tasks` (list)
    - Result queue: `prime_rl:queue:results` (list)
    - Task metadata: `prime_rl:queue:task:{task_id}` (hash)
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        queue_prefix: str = "prime_rl:queue",
        max_queue_size: int = 100000,
    ):
        """
        Initialize Redis queue.
        
        Args:
            redis_url: Redis connection URL
            queue_prefix: Redis key prefix for queues
            max_queue_size: Maximum queue size (for backpressure)
        """
        if not REDIS_AVAILABLE and redis is None:
            raise ImportError(
                "Redis required for RedisQueue. Install with: pip install redis"
            )
        
        self.redis_url = redis_url
        self.queue_prefix = queue_prefix
        self.max_queue_size = max_queue_size
        
        # Redis client (created lazily)
        self._redis_client: Optional[Any] = None
        self._use_async = REDIS_AVAILABLE
    
    async def _get_redis_client(self):
        """Get or create Redis client."""
        if self._redis_client is None:
            if self._use_async:
                self._redis_client = aioredis.from_url(self.redis_url)
            else:
                self._redis_client = redis.from_url(self.redis_url)
        return self._redis_client
    
    async def push_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 0,
    ) -> str:
        """
        Push a task to the queue.
        
        Args:
            task_type: Task type (e.g., "env_step", "env_reset")
            payload: Task payload dictionary
            priority: Task priority (higher = more important)
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        redis_client = await self._get_redis_client()
        
        task_data = {
            "task_id": task_id,
            "task_type": task_type,
            "payload": payload,
            "priority": priority,
            "created_at": str(asyncio.get_event_loop().time()),
        }
        
        task_key = f"{self.queue_prefix}:task:{task_id}"
        queue_key = f"{self.queue_prefix}:tasks"
        
        # Store task metadata
        if self._use_async:
            await redis_client.set(task_key, json.dumps(task_data), ex=3600)
            
            # Check queue size for backpressure
            queue_size = await redis_client.llen(queue_key)
            if queue_size >= self.max_queue_size:
                raise RuntimeError(f"Queue full: {queue_size} >= {self.max_queue_size}")
            
            # Push to queue (using priority score for sorted set, or simple list)
            await redis_client.lpush(queue_key, task_id)
        else:
            redis_client.set(task_key, json.dumps(task_data), ex=3600)
            
            queue_size = redis_client.llen(queue_key)
            if queue_size >= self.max_queue_size:
                raise RuntimeError(f"Queue full: {queue_size} >= {self.max_queue_size}")
            
            redis_client.lpush(queue_key, task_id)
        
        logger.debug(f"Task pushed: task_id={task_id}, type={task_type}")
        return task_id
    
    async def pop_task(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        Pop a task from the queue.
        
        Args:
            timeout: Blocking timeout in seconds
            
        Returns:
            Task dictionary or None if timeout
        """
        redis_client = await self._get_redis_client()
        queue_key = f"{self.queue_prefix}:tasks"
        
        if self._use_async:
            # Blocking pop with timeout
            result = await redis_client.brpop(queue_key, timeout=int(timeout))
            if result is None:
                return None
            
            _, task_id_bytes = result
            task_id = task_id_bytes.decode("utf-8") if isinstance(task_id_bytes, bytes) else task_id_bytes
        else:
            # Non-blocking pop (would need threading for blocking in sync mode)
            task_id_bytes = redis_client.rpop(queue_key)
            if task_id_bytes is None:
                return None
            
            task_id = task_id_bytes.decode("utf-8") if isinstance(task_id_bytes, bytes) else task_id_bytes
        
        # Get task data
        task_key = f"{self.queue_prefix}:task:{task_id}"
        
        if self._use_async:
            task_data_bytes = await redis_client.get(task_key)
        else:
            task_data_bytes = redis_client.get(task_key)
        
        if task_data_bytes is None:
            logger.warning(f"Task data not found for task_id={task_id}")
            return None
        
        if isinstance(task_data_bytes, bytes):
            task_data_bytes = task_data_bytes.decode("utf-8")
        
        task_data = json.loads(task_data_bytes)
        return task_data
    
    async def push_result(self, task_id: str, result: Dict[str, Any]) -> None:
        """
        Push a result for a completed task.
        
        Args:
            task_id: Task identifier
            result: Result dictionary
        """
        redis_client = await self._get_redis_client()
        
        result_data = {
            "task_id": task_id,
            "result": result,
            "completed_at": str(asyncio.get_event_loop().time()),
        }
        
        result_key = f"{self.queue_prefix}:result:{task_id}"
        results_queue_key = f"{self.queue_prefix}:results"
        
        if self._use_async:
            await redis_client.set(result_key, json.dumps(result_data), ex=3600)
            await redis_client.lpush(results_queue_key, task_id)
        else:
            redis_client.set(result_key, json.dumps(result_data), ex=3600)
            redis_client.lpush(results_queue_key, task_id)
        
        logger.debug(f"Result pushed: task_id={task_id}")
    
    async def pop_result(self, task_id: str, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """
        Pop a result for a specific task.
        
        Args:
            task_id: Task identifier
            timeout: Timeout in seconds
            
        Returns:
            Result dictionary or None if timeout
        """
        redis_client = await self._get_redis_client()
        result_key = f"{self.queue_prefix}:result:{task_id}"
        
        # Poll for result
        start_time = asyncio.get_event_loop().time()
        while True:
            if self._use_async:
                result_data_bytes = await redis_client.get(result_key)
            else:
                result_data_bytes = redis_client.get(result_key)
            
            if result_data_bytes is not None:
                if isinstance(result_data_bytes, bytes):
                    result_data_bytes = result_data_bytes.decode("utf-8")
                result_data = json.loads(result_data_bytes)
                return result_data.get("result")
            
            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                return None
            
            # Small delay before retry
            await asyncio.sleep(0.1)
    
    async def get_queue_size(self) -> int:
        """
        Get current queue size.
        
        Returns:
            Number of tasks in queue
        """
        redis_client = await self._get_redis_client()
        queue_key = f"{self.queue_prefix}:tasks"
        
        if self._use_async:
            size = await redis_client.llen(queue_key)
        else:
            size = redis_client.llen(queue_key)
        
        return int(size)
    
    async def clear_queue(self) -> None:
        """Clear all tasks from the queue."""
        redis_client = await self._get_redis_client()
        
        queue_key = f"{self.queue_prefix}:tasks"
        results_queue_key = f"{self.queue_prefix}:results"
        
        if self._use_async:
            await redis_client.delete(queue_key)
            await redis_client.delete(results_queue_key)
        else:
            redis_client.delete(queue_key)
            redis_client.delete(results_queue_key)
        
        logger.info("Queue cleared")
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis_client:
            if self._use_async:
                await self._redis_client.close()
            else:
                self._redis_client.close()
            self._redis_client = None

