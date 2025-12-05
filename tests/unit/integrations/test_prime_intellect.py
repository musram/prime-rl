"""
Tests for Prime Intellect integrations.

Tests PrimeIntellectEnvAdapter and PrimeIntellectVerifierClient with mocked HTTP calls.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    try:
        import requests
        HTTPX_AVAILABLE = False
    except ImportError:
        HTTPX_AVAILABLE = False
        httpx = None
        requests = None

if HTTPX_AVAILABLE or requests is not None:
    from prime_rl.integrations.prime_intellect.adapter import PrimeIntellectEnvAdapter
    from prime_rl.integrations.prime_intellect.client import PrimeIntellectVerifierClient
    from prime_rl.core.verifier import (
        VerificationResult,
        VerifierError,
        VerifierTimeoutError,
        VerifierNetworkError,
    )


@pytest.mark.skipif(not (HTTPX_AVAILABLE or requests is not None), reason="HTTP client not available")
class TestPrimeIntellectEnvAdapter:
    """Tests for PrimeIntellectEnvAdapter."""
    
    def test_initialization(self):
        """Test adapter initialization."""
        adapter = PrimeIntellectEnvAdapter(
            environment_id="browser-gym-v1",
            endpoint="https://api.primeintellect.ai/environments",
            api_key="test-key",
        )
        assert adapter.environment_id == "browser-gym-v1"
        assert adapter.endpoint == "https://api.primeintellect.ai/environments"
        assert adapter.api_key == "test-key"
    
    def test_initialization_with_env_var(self, monkeypatch):
        """Test initialization with environment variable."""
        monkeypatch.setenv("TEST_API_KEY", "env-key-value")
        
        adapter = PrimeIntellectEnvAdapter(
            environment_id="test-env",
            endpoint="https://api.example.com",
            api_key="env:TEST_API_KEY",
        )
        assert adapter.api_key == "env-key-value"
    
    @patch("httpx.Client.post" if HTTPX_AVAILABLE else "requests.Session.post")
    def test_reset(self, mock_post):
        """Test environment reset."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "observation": "initial_obs",
            "info": {"episode_id": "ep-123"},
            "episode_id": "ep-123",
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        adapter = PrimeIntellectEnvAdapter(
            environment_id="test-env",
            endpoint="https://api.example.com",
            api_key="test-key",
        )
        
        obs, info = adapter.reset(seed=42)
        
        assert obs == "initial_obs"
        assert info == {"episode_id": "ep-123"}
        assert adapter._episode_id == "ep-123"
        mock_post.assert_called_once()
    
    @patch("httpx.Client.post" if HTTPX_AVAILABLE else "requests.Session.post")
    def test_step(self, mock_post):
        """Test environment step."""
        # Mock reset response
        mock_reset_response = Mock()
        mock_reset_response.json.return_value = {
            "observation": "initial_obs",
            "episode_id": "ep-123",
        }
        mock_reset_response.raise_for_status = Mock()
        
        # Mock step response
        mock_step_response = Mock()
        mock_step_response.json.return_value = {
            "observation": "next_obs",
            "reward": 1.0,
            "terminated": False,
            "truncated": False,
            "info": {},
        }
        mock_step_response.raise_for_status = Mock()
        
        if HTTPX_AVAILABLE:
            mock_post.side_effect = [mock_reset_response, mock_step_response]
        else:
            mock_post.side_effect = [mock_reset_response, mock_step_response]
        
        adapter = PrimeIntellectEnvAdapter(
            environment_id="test-env",
            endpoint="https://api.example.com",
            api_key="test-key",
        )
        
        adapter.reset()
        obs, reward, terminated, truncated, info = adapter.step("action1")
        
        assert obs == "next_obs"
        assert reward == 1.0
        assert terminated is False
        assert truncated is False
    
    @patch("httpx.Client.post" if HTTPX_AVAILABLE else "requests.Session.post")
    def test_step_before_reset(self, mock_post):
        """Test that stepping before reset raises error."""
        adapter = PrimeIntellectEnvAdapter(
            environment_id="test-env",
            endpoint="https://api.example.com",
        )
        
        with pytest.raises(RuntimeError, match="must be reset"):
            adapter.step("action")
    
    def test_close(self):
        """Test adapter cleanup."""
        adapter = PrimeIntellectEnvAdapter(
            environment_id="test-env",
            endpoint="https://api.example.com",
        )
        
        adapter.close()
        assert adapter._current_observation is None
        assert adapter._episode_id is None


@pytest.mark.skipif(not (HTTPX_AVAILABLE or requests is not None), reason="HTTP client not available")
class TestPrimeIntellectVerifierClient:
    """Tests for PrimeIntellectVerifierClient."""
    
    def test_initialization(self):
        """Test client initialization."""
        client = PrimeIntellectVerifierClient(
            endpoint="https://api.primeintellect.ai/verifier/rubric",
            api_key="test-key",
        )
        assert client.endpoint == "https://api.primeintellect.ai/verifier/rubric"
        assert client.api_key == "test-key"
    
    def test_initialization_with_env_var(self, monkeypatch):
        """Test initialization with environment variable."""
        monkeypatch.setenv("PI_API_KEY", "env-key-value")
        
        client = PrimeIntellectVerifierClient(
            endpoint="https://api.example.com",
            api_key="env:PI_API_KEY",
        )
        assert client.api_key == "env-key-value"
    
    @patch("httpx.Client.post" if HTTPX_AVAILABLE else "requests.Session.post")
    def test_verify_success(self, mock_post):
        """Test successful verification."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "reward": 0.87,
            "success": True,
            "metadata": {"rubric_scores": {"criteria1": 0.9, "criteria2": 0.85}},
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        client = PrimeIntellectVerifierClient(
            endpoint="https://api.example.com",
            api_key="test-key",
        )
        
        result = client.verify(
            observation="prompt",
            action="completion",
            trace_id="trace-123",
        )
        
        assert isinstance(result, VerificationResult)
        assert result.reward == 0.87
        assert result.success is True
        assert "rubric_scores" in result.metadata
        mock_post.assert_called_once()
    
    @patch("httpx.Client.post" if HTTPX_AVAILABLE else "requests.Session.post")
    def test_verify_with_rubric_metadata(self, mock_post):
        """Test verification with rubric metadata."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "reward": 0.9,
            "success": True,
            "rubric_scores": {"criteria1": 1.0, "criteria2": 0.8},
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        client = PrimeIntellectVerifierClient(
            endpoint="https://api.example.com",
        )
        
        # Build payload with rubric metadata (would be passed via kwargs in real usage)
        result = client.verify(
            observation="prompt",
            action="completion",
        )
        
        assert result.reward == 0.9
        assert "rubric_scores" in result.metadata
    
    @patch("httpx.Client.post" if HTTPX_AVAILABLE else "requests.Session.post")
    def test_verify_timeout(self, mock_post):
        """Test timeout handling."""
        if HTTPX_AVAILABLE:
            mock_post.side_effect = httpx.TimeoutException("Request timed out")
        else:
            import requests
            mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
        
        client = PrimeIntellectVerifierClient(
            endpoint="https://api.example.com",
            default_retry_config={"max_retries": 1},
        )
        
        with pytest.raises(VerifierTimeoutError):
            client.verify(
                observation="prompt",
                action="completion",
                timeout=1.0,
            )
    
    @patch("httpx.Client.post" if HTTPX_AVAILABLE else "requests.Session.post")
    def test_verify_network_error(self, mock_post):
        """Test network error handling."""
        if HTTPX_AVAILABLE:
            mock_post.side_effect = httpx.NetworkError("Connection failed")
        else:
            import requests
            mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        client = PrimeIntellectVerifierClient(
            endpoint="https://api.example.com",
            default_retry_config={"max_retries": 1},
        )
        
        with pytest.raises(VerifierNetworkError):
            client.verify(
                observation="prompt",
                action="completion",
            )
    
    @pytest.mark.asyncio
    async def test_verify_async_success(self):
        """Test successful async verification."""
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={
            "reward": 0.87,
            "success": True,
        })
        mock_response.raise_for_status = Mock()
        
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        
        client = PrimeIntellectVerifierClient(
            endpoint="https://api.example.com",
        )
        client._async_client = mock_client
        
        result = await client.verify_async(
            observation="prompt",
            action="completion",
        )
        
        assert result.reward == 0.87
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_verify_batch(self):
        """Test batch verification."""
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={
            "reward": 0.5,
            "success": True,
        })
        mock_response.raise_for_status = Mock()
        
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        
        client = PrimeIntellectVerifierClient(
            endpoint="https://api.example.com",
        )
        client._async_client = mock_client
        
        results = await client.verify_batch(
            observations=["obs1", "obs2"],
            actions=["act1", "act2"],
            trace_ids=["trace1", "trace2"],
        )
        
        assert len(results) == 2
        assert all(isinstance(r, VerificationResult) for r in results)
    
    def test_close(self):
        """Test client cleanup."""
        client = PrimeIntellectVerifierClient(
            endpoint="https://api.example.com",
        )
        
        # Should not raise
        client.close()


@pytest.mark.skipif(not (HTTPX_AVAILABLE or requests is not None), reason="HTTP client not available")
class TestRegistry:
    """Tests for registry integration."""
    
    def test_registry_imports(self):
        """Test that Prime Intellect adapters are registered."""
        from prime_rl.core.registry import (
            get_registered_adapters,
            get_registered_verifiers,
            create_environment_adapter,
            create_verifier_client,
        )
        
        adapters = get_registered_adapters()
        verifiers = get_registered_verifiers()
        
        # Check that Prime Intellect is registered
        assert "prime_intellect" in adapters
        assert "prime_intellect" in verifiers or "prime_intellect_rar" in verifiers
    
    def test_create_adapter_from_registry(self):
        """Test creating adapter from registry."""
        from prime_rl.core.registry import create_environment_adapter
        
        with patch("httpx.Client.post" if HTTPX_AVAILABLE else "requests.Session.post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "observation": "obs",
                "episode_id": "ep-123",
            }
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response
            
            adapter = create_environment_adapter(
                "prime_intellect",
                environment_id="test-env",
                endpoint="https://api.example.com",
                api_key="test-key",
            )
            
            assert isinstance(adapter, PrimeIntellectEnvAdapter)
            obs, info = adapter.reset()
            assert obs == "obs"
    
    def test_create_verifier_from_registry(self):
        """Test creating verifier from registry."""
        from prime_rl.core.registry import create_verifier_client
        
        with patch("httpx.Client.post" if HTTPX_AVAILABLE else "requests.Session.post") as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "reward": 0.9,
                "success": True,
            }
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response
            
            client = create_verifier_client(
                "prime_intellect_rar",
                endpoint="https://api.example.com",
                api_key="test-key",
            )
            
            assert isinstance(client, PrimeIntellectVerifierClient)
            result = client.verify("prompt", "completion")
            assert result.reward == 0.9

