"""
Tests for HttpVerifierClient.

Tests HTTP verification with mocked requests to ensure proper error handling,
retry logic, and response parsing.
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

try:
    import requests
    import aiohttp
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False

if HTTP_AVAILABLE:
    from prime_rl.integrations.remote.http_verifier import HttpVerifierClient
    from prime_rl.core.verifier import (
        VerificationResult,
        VerifierError,
        VerifierTimeoutError,
        VerifierNetworkError,
    )


@pytest.mark.skipif(not HTTP_AVAILABLE, reason="HTTP libraries not available")
class TestHttpVerifierClient:
    """Tests for HttpVerifierClient."""
    
    def test_initialization(self):
        """Test HttpVerifierClient initialization."""
        client = HttpVerifierClient(
            endpoint_url="https://api.example.com/verify",
            api_key="test-key",
        )
        assert client.endpoint_url == "https://api.example.com/verify"
        assert client.api_key == "test-key"
        assert "Authorization" in client.headers
        assert client.headers["Authorization"] == "Bearer test-key"
    
    def test_initialization_without_api_key(self):
        """Test initialization without API key."""
        client = HttpVerifierClient(endpoint_url="https://api.example.com/verify")
        assert client.api_key is None
        assert "Authorization" not in client.headers
    
    def test_build_request_payload(self):
        """Test request payload building."""
        client = HttpVerifierClient(endpoint_url="https://api.example.com/verify")
        
        payload = client._build_request_payload(
            observation="test_obs",
            action="test_action",
            trace_id="trace-123",
            metadata={"key": "value"},
        )
        
        assert payload["observation"] == "test_obs"
        assert payload["action"] == "test_action"
        assert payload["trace_id"] == "trace-123"
        assert payload["metadata"] == {"key": "value"}
    
    def test_parse_response(self):
        """Test response parsing."""
        client = HttpVerifierClient(endpoint_url="https://api.example.com/verify")
        
        response_data = {
            "reward": 1.0,
            "success": True,
            "metadata": {"confidence": 0.9},
            "trace_id": "trace-123",
        }
        
        result = client._parse_response(response_data)
        assert isinstance(result, VerificationResult)
        assert result.reward == 1.0
        assert result.success is True
        assert result.metadata == {"confidence": 0.9}
        assert result.trace_id == "trace-123"
    
    def test_parse_response_minimal(self):
        """Test parsing minimal response."""
        client = HttpVerifierClient(endpoint_url="https://api.example.com/verify")
        
        response_data = {}
        result = client._parse_response(response_data)
        assert result.reward == 0.0
        assert result.success is False
    
    @patch("requests.post")
    def test_verify_success(self, mock_post):
        """Test successful verification."""
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {
            "reward": 1.0,
            "success": True,
            "metadata": {},
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        client = HttpVerifierClient(endpoint_url="https://api.example.com/verify")
        result = client.verify(
            observation="test_obs",
            action="test_action",
        )
        
        assert result.reward == 1.0
        assert result.success is True
        mock_post.assert_called_once()
    
    @patch("requests.post")
    def test_verify_timeout(self, mock_post):
        """Test timeout handling."""
        import requests
        
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
        
        client = HttpVerifierClient(endpoint_url="https://api.example.com/verify")
        
        with pytest.raises(VerifierTimeoutError):
            client.verify(
                observation="test_obs",
                action="test_action",
                timeout=1.0,
            )
    
    @patch("requests.post")
    def test_verify_network_error(self, mock_post):
        """Test network error handling."""
        import requests
        
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        client = HttpVerifierClient(
            endpoint_url="https://api.example.com/verify",
            default_retry_config={"max_retries": 1},
        )
        
        with pytest.raises(VerifierNetworkError):
            client.verify(
                observation="test_obs",
                action="test_action",
            )
    
    @patch("requests.post")
    def test_verify_invalid_json(self, mock_post):
        """Test invalid JSON response handling."""
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        client = HttpVerifierClient(endpoint_url="https://api.example.com/verify")
        
        with pytest.raises(VerifierError):
            client.verify(
                observation="test_obs",
                action="test_action",
            )
    
    @pytest.mark.asyncio
    async def test_verify_async_success(self):
        """Test successful async verification."""
        client = HttpVerifierClient(endpoint_url="https://api.example.com/verify")
        
        # Mock async session
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={
            "reward": 1.0,
            "success": True,
            "metadata": {},
        })
        mock_response.raise_for_status = Mock()
        
        mock_session = AsyncMock()
        mock_session.post = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        
        client._async_session = mock_session
        
        result = await client.verify_async(
            observation="test_obs",
            action="test_action",
        )
        
        assert result.reward == 1.0
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_verify_batch(self):
        """Test batch verification."""
        client = HttpVerifierClient(endpoint_url="https://api.example.com/verify")
        
        # Mock async verify_async
        async def mock_verify_async(*args, **kwargs):
            return VerificationResult(
                reward=1.0,
                success=True,
                metadata={},
            )
        
        client.verify_async = mock_verify_async
        
        results = await client.verify_batch(
            observations=["obs1", "obs2"],
            actions=["act1", "act2"],
            trace_ids=["trace1", "trace2"],
        )
        
        assert len(results) == 2
        assert all(isinstance(r, VerificationResult) for r in results)
    
    @pytest.mark.asyncio
    async def test_verify_batch_mismatched_lengths(self):
        """Test batch verification with mismatched lengths."""
        client = HttpVerifierClient(endpoint_url="https://api.example.com/verify")
        
        with pytest.raises(VerifierError):
            await client.verify_batch(
                observations=["obs1"],
                actions=["act1", "act2"],
            )
    
    def test_close(self):
        """Test client cleanup."""
        client = HttpVerifierClient(endpoint_url="https://api.example.com/verify")
        
        # Should not raise
        client.close()
        
        # With async session
        mock_session = AsyncMock()
        client._async_session = mock_session
        
        client.close()
        # Session should be closed (mocked)

