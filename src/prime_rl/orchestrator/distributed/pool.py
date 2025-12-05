"""
Worker pool interface and implementations for distributed orchestration.

This module provides abstract and concrete implementations of worker pools
for managing remote workers in distributed RL training scenarios.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import asyncio

from loguru import logger


class WorkerPool(ABC):
    """
    Abstract base class for managing remote workers.
    
    Worker pools abstract away the details of worker management (Kubernetes,
    Ray, etc.) and provide a unified interface for scaling environment
    execution across multiple nodes.
    
    This interface enables support for 10,000+ concurrent environment instances
    by sharding EnvGroup across multiple nodes.
    """
    
    @abstractmethod
    async def register_worker(self, worker_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Register a new worker with the pool.
        
        Args:
            worker_id: Unique worker identifier
            metadata: Optional worker metadata
            
        Returns:
            Session ID for the worker
        """
        pass
    
    @abstractmethod
    async def get_worker_count(self) -> int:
        """
        Get the current number of active workers.
        
        Returns:
            Number of active workers
        """
        pass
    
    @abstractmethod
    async def scale_workers(self, target_count: int) -> None:
        """
        Scale the worker pool to a target count.
        
        Args:
            target_count: Target number of workers
        """
        pass
    
    @abstractmethod
    async def get_worker_status(self, worker_id: str) -> Dict[str, Any]:
        """
        Get status of a specific worker.
        
        Args:
            worker_id: Worker identifier
            
        Returns:
            Worker status dictionary
        """
        pass
    
    @abstractmethod
    async def shutdown_worker(self, worker_id: str) -> None:
        """
        Shutdown a specific worker.
        
        Args:
            worker_id: Worker identifier
        """
        pass
    
    @abstractmethod
    async def shutdown_all(self) -> None:
        """
        Shutdown all workers in the pool.
        """
        pass


class RedisWorkerPool(WorkerPool):
    """
    Redis-based worker pool implementation.
    
    Uses Redis for worker registration and coordination. Workers register
    themselves in Redis, and the pool tracks active workers via Redis keys.
    
    This implementation is designed for high-throughput scenarios where
    workers can be dynamically scaled across multiple nodes.
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        worker_prefix: str = "prime_rl:worker",
        session_prefix: str = "prime_rl:session",
    ):
        """
        Initialize Redis worker pool.
        
        Args:
            redis_url: Redis connection URL
            worker_prefix: Redis key prefix for worker data
            session_prefix: Redis key prefix for session data
        """
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
        
        if not REDIS_AVAILABLE and redis is None:
            raise ImportError(
                "Redis required for RedisWorkerPool. Install with: pip install redis"
            )
        
        self.redis_url = redis_url
        self.worker_prefix = worker_prefix
        self.session_prefix = session_prefix
        
        # Redis client (created lazily)
        self._redis_client: Optional[Any] = None
        self._use_async = REDIS_AVAILABLE
    
    async def _get_redis_client(self):
        """Get or create Redis client."""
        if self._redis_client is None:
            if self._use_async:
                import redis.asyncio as aioredis
                self._redis_client = aioredis.from_url(self.redis_url)
            else:
                import redis
                self._redis_client = redis.from_url(self.redis_url)
        return self._redis_client
    
    async def register_worker(self, worker_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Register a new worker with the pool.
        
        Args:
            worker_id: Unique worker identifier
            metadata: Optional worker metadata
            
        Returns:
            Session ID for the worker
        """
        import uuid
        import json
        
        session_id = str(uuid.uuid4())
        redis_client = await self._get_redis_client()
        
        worker_key = f"{self.worker_prefix}:{worker_id}"
        session_key = f"{self.session_prefix}:{session_id}"
        
        worker_data = {
            "worker_id": worker_id,
            "session_id": session_id,
            "metadata": metadata or {},
            "registered_at": str(asyncio.get_event_loop().time()),
        }
        
        if self._use_async:
            await redis_client.set(worker_key, json.dumps(worker_data), ex=3600)  # 1 hour TTL
            await redis_client.set(session_key, json.dumps({"worker_id": worker_id}), ex=3600)
            await redis_client.sadd(f"{self.worker_prefix}:active", worker_id)
        else:
            redis_client.set(worker_key, json.dumps(worker_data), ex=3600)
            redis_client.set(session_key, json.dumps({"worker_id": worker_id}), ex=3600)
            redis_client.sadd(f"{self.worker_prefix}:active", worker_id)
        
        logger.info(f"Worker registered: worker_id={worker_id}, session_id={session_id}")
        return session_id
    
    async def get_worker_count(self) -> int:
        """
        Get the current number of active workers.
        
        Returns:
            Number of active workers
        """
        redis_client = await self._get_redis_client()
        
        if self._use_async:
            count = await redis_client.scard(f"{self.worker_prefix}:active")
        else:
            count = redis_client.scard(f"{self.worker_prefix}:active")
        
        return int(count)
    
    async def scale_workers(self, target_count: int) -> None:
        """
        Scale the worker pool to a target count.
        
        Note: This is a stub implementation. In production, would integrate
        with Kubernetes/Ray/etc. to actually scale worker nodes.
        
        Args:
            target_count: Target number of workers
        """
        current_count = await self.get_worker_count()
        logger.info(f"Scaling workers: current={current_count}, target={target_count}")
        
        # Stub: In production, would call Kubernetes API or Ray API to scale
        # For now, just log the intent
        if target_count > current_count:
            logger.info(f"Would scale up {target_count - current_count} workers")
        elif target_count < current_count:
            logger.info(f"Would scale down {current_count - target_count} workers")
    
    async def get_worker_status(self, worker_id: str) -> Dict[str, Any]:
        """
        Get status of a specific worker.
        
        Args:
            worker_id: Worker identifier
            
        Returns:
            Worker status dictionary
        """
        import json
        
        redis_client = await self._get_redis_client()
        worker_key = f"{self.worker_prefix}:{worker_id}"
        
        if self._use_async:
            data = await redis_client.get(worker_key)
        else:
            data = redis_client.get(worker_key)
        
        if data is None:
            raise ValueError(f"Worker {worker_id} not found")
        
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        
        return json.loads(data)
    
    async def shutdown_worker(self, worker_id: str) -> None:
        """
        Shutdown a specific worker.
        
        Args:
            worker_id: Worker identifier
        """
        redis_client = await self._get_redis_client()
        
        worker_key = f"{self.worker_prefix}:{worker_id}"
        
        if self._use_async:
            await redis_client.delete(worker_key)
            await redis_client.srem(f"{self.worker_prefix}:active", worker_id)
        else:
            redis_client.delete(worker_key)
            redis_client.srem(f"{self.worker_prefix}:active", worker_id)
        
        logger.info(f"Worker shutdown: worker_id={worker_id}")
    
    async def shutdown_all(self) -> None:
        """
        Shutdown all workers in the pool.
        """
        redis_client = await self._get_redis_client()
        
        if self._use_async:
            workers = await redis_client.smembers(f"{self.worker_prefix}:active")
            for worker_id_bytes in workers:
                worker_id = worker_id_bytes.decode("utf-8") if isinstance(worker_id_bytes, bytes) else worker_id_bytes
                await self.shutdown_worker(worker_id)
        else:
            workers = redis_client.smembers(f"{self.worker_prefix}:active")
            for worker_id_bytes in workers:
                worker_id = worker_id_bytes.decode("utf-8") if isinstance(worker_id_bytes, bytes) else worker_id_bytes
                await self.shutdown_worker(worker_id)
        
        logger.info("All workers shutdown")

