"""Unit tests for RedisQueue."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import json

from prime_rl.orchestrator.distributed.queue import RedisQueue


class TestRedisQueue:
    """Tests for RedisQueue implementation."""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        with patch("prime_rl.orchestrator.distributed.queue.aioredis") as mock:
            mock_client = AsyncMock()
            mock.from_url.return_value = mock_client
            yield mock_client
    
    @pytest.mark.asyncio
    async def test_push_task(self, mock_redis):
        """Test pushing a task to the queue."""
        queue = RedisQueue(redis_url="redis://localhost:6379")
        queue._redis_client = mock_redis
        
        mock_redis.set = AsyncMock()
        mock_redis.llen = AsyncMock(return_value=0)
        mock_redis.lpush = AsyncMock()
        
        task_id = await queue.push_task("env_step", {"env_id": "test"})
        
        assert task_id is not None
        mock_redis.set.assert_called_once()
        mock_redis.lpush.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_push_task_queue_full(self, mock_redis):
        """Test pushing when queue is full."""
        queue = RedisQueue(redis_url="redis://localhost:6379", max_queue_size=10)
        queue._redis_client = mock_redis
        
        mock_redis.llen = AsyncMock(return_value=10)
        
        with pytest.raises(RuntimeError, match="Queue full"):
            await queue.push_task("env_step", {"env_id": "test"})
    
    @pytest.mark.asyncio
    async def test_pop_task(self, mock_redis):
        """Test popping a task from the queue."""
        queue = RedisQueue(redis_url="redis://localhost:6379")
        queue._redis_client = mock_redis
        
        task_data = {
            "task_id": "task-1",
            "task_type": "env_step",
            "payload": {"env_id": "test"},
        }
        
        mock_redis.brpop = AsyncMock(return_value=(b"queue", b"task-1"))
        mock_redis.get = AsyncMock(return_value=json.dumps(task_data).encode())
        
        task = await queue.pop_task(timeout=1.0)
        
        assert task is not None
        assert task["task_id"] == "task-1"
        assert task["task_type"] == "env_step"
    
    @pytest.mark.asyncio
    async def test_pop_task_timeout(self, mock_redis):
        """Test popping with timeout."""
        queue = RedisQueue(redis_url="redis://localhost:6379")
        queue._redis_client = mock_redis
        
        mock_redis.brpop = AsyncMock(return_value=None)
        
        task = await queue.pop_task(timeout=0.1)
        
        assert task is None
    
    @pytest.mark.asyncio
    async def test_push_result(self, mock_redis):
        """Test pushing a result."""
        queue = RedisQueue(redis_url="redis://localhost:6379")
        queue._redis_client = mock_redis
        
        mock_redis.set = AsyncMock()
        mock_redis.lpush = AsyncMock()
        
        await queue.push_result("task-1", {"result": "success"})
        
        assert mock_redis.set.call_count == 1
        assert mock_redis.lpush.call_count == 1
    
    @pytest.mark.asyncio
    async def test_get_queue_size(self, mock_redis):
        """Test getting queue size."""
        queue = RedisQueue(redis_url="redis://localhost:6379")
        queue._redis_client = mock_redis
        
        mock_redis.llen = AsyncMock(return_value=42)
        
        size = await queue.get_queue_size()
        
        assert size == 42
    
    @pytest.mark.asyncio
    async def test_clear_queue(self, mock_redis):
        """Test clearing the queue."""
        queue = RedisQueue(redis_url="redis://localhost:6379")
        queue._redis_client = mock_redis
        
        mock_redis.delete = AsyncMock()
        
        await queue.clear_queue()
        
        assert mock_redis.delete.call_count == 2
    
    def test_redis_not_available(self):
        """Test error when Redis is not installed."""
        with patch("prime_rl.orchestrator.distributed.queue.aioredis", None):
            with patch("prime_rl.orchestrator.distributed.queue.redis", None):
                with pytest.raises(ImportError):
                    RedisQueue()

