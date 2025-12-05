"""Unit tests for WorkerPool implementations."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import json

from prime_rl.orchestrator.distributed.pool import WorkerPool, RedisWorkerPool


class TestWorkerPool:
    """Tests for WorkerPool abstract interface."""
    
    def test_worker_pool_is_abstract(self):
        """Test that WorkerPool cannot be instantiated."""
        with pytest.raises(TypeError):
            WorkerPool()


class TestRedisWorkerPool:
    """Tests for RedisWorkerPool implementation."""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        with patch("prime_rl.orchestrator.distributed.pool.aioredis") as mock:
            mock_client = AsyncMock()
            mock.from_url.return_value = mock_client
            yield mock_client
    
    @pytest.mark.asyncio
    async def test_register_worker(self, mock_redis):
        """Test worker registration."""
        pool = RedisWorkerPool(redis_url="redis://localhost:6379")
        pool._redis_client = mock_redis
        
        mock_redis.set = AsyncMock()
        mock_redis.sadd = AsyncMock()
        
        session_id = await pool.register_worker("worker-1", {"env": "test"})
        
        assert session_id is not None
        assert mock_redis.set.call_count == 2
        assert mock_redis.sadd.call_count == 1
    
    @pytest.mark.asyncio
    async def test_get_worker_count(self, mock_redis):
        """Test getting worker count."""
        pool = RedisWorkerPool(redis_url="redis://localhost:6379")
        pool._redis_client = mock_redis
        
        mock_redis.scard = AsyncMock(return_value=5)
        
        count = await pool.get_worker_count()
        
        assert count == 5
        mock_redis.scard.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_worker_status(self, mock_redis):
        """Test getting worker status."""
        pool = RedisWorkerPool(redis_url="redis://localhost:6379")
        pool._redis_client = mock_redis
        
        worker_data = {
            "worker_id": "worker-1",
            "session_id": "session-1",
            "metadata": {},
        }
        mock_redis.get = AsyncMock(return_value=json.dumps(worker_data).encode())
        
        status = await pool.get_worker_status("worker-1")
        
        assert status["worker_id"] == "worker-1"
        assert status["session_id"] == "session-1"
    
    @pytest.mark.asyncio
    async def test_shutdown_worker(self, mock_redis):
        """Test shutting down a worker."""
        pool = RedisWorkerPool(redis_url="redis://localhost:6379")
        pool._redis_client = mock_redis
        
        mock_redis.delete = AsyncMock()
        mock_redis.srem = AsyncMock()
        
        await pool.shutdown_worker("worker-1")
        
        mock_redis.delete.assert_called_once()
        mock_redis.srem.assert_called_once()
    
    def test_redis_not_available(self):
        """Test error when Redis is not installed."""
        with patch("prime_rl.orchestrator.distributed.pool.aioredis", None):
            with patch("prime_rl.orchestrator.distributed.pool.redis", None):
                with pytest.raises(ImportError):
                    RedisWorkerPool()

