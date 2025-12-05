"""Tests for VerifierClient interface."""

import pytest

from prime_rl.core.verifier import (
    VerifierClient,
    VerificationResult,
    VerificationRequest,
    VerifierError,
    VerifierTimeoutError,
    VerifierNetworkError,
)


def test_verification_result():
    """Test VerificationResult creation."""
    result = VerificationResult(
        reward=1.0,
        success=True,
        metadata={"confidence": 0.9},
        trace_id="test-trace",
    )
    assert result.reward == 1.0
    assert result.success is True
    assert result.metadata == {"confidence": 0.9}
    assert result.trace_id == "test-trace"
    
    data = result.to_dict()
    assert data["reward"] == 1.0
    assert data["success"] is True


def test_verification_request():
    """Test VerificationRequest creation."""
    request = VerificationRequest(
        observation="test_obs",
        action="test_action",
        trace_id="test-trace",
        metadata={"key": "value"},
    )
    assert request.observation == "test_obs"
    assert request.action == "test_action"
    assert request.trace_id == "test-trace"
    assert request.metadata == {"key": "value"}
    
    data = request.to_dict()
    assert data["observation"] == "test_obs"
    assert data["action"] == "test_action"


def test_verifier_error_hierarchy():
    """Test VerifierError exception hierarchy."""
    assert issubclass(VerifierTimeoutError, VerifierError)
    assert issubclass(VerifierNetworkError, VerifierError)

