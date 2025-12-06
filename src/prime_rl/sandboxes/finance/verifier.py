"""
Finance Reconciliation Sandbox Verifier.

Provides rubric-based verification for finance reconciliation workflows.
"""

from typing import Any, Dict, List, Optional
from prime_rl.core.verifier import VerifierClient, VerificationRequest, VerificationResult


class FinanceReconciliationRubricVerifier(VerifierClient):
    """
    Rubric-based verifier for finance reconciliation sandbox.
    
    Scores episodes based on:
    - Correctness: Are transactions matched correctly?
    - Constraint satisfaction: Are accounting constraints satisfied?
    - Completeness: Are all transactions reconciled?
    - Accuracy: Are amounts and dates matched accurately?
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize finance rubric verifier.
        
        Args:
            weights: Optional weights for rubric criteria (default: equal weights)
        """
        self.weights = weights or {
            "correctness": 0.5,
            "constraint_satisfaction": 0.2,
            "completeness": 0.2,
            "accuracy": 0.1,
        }
    
    def verify(
        self,
        observation: Any,
        action: Any,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        """
        Verify a single action in finance context.
        
        Args:
            observation: Environment observation
            action: Agent action
            trace_id: Optional trace identifier
            metadata: Optional metadata (can include rubric scores)
            
        Returns:
            VerificationResult with reward and metadata
        """
        # Extract rubric scores from metadata if provided
        rubric_scores = metadata.get("rubric_scores", {}) if metadata else {}
        
        # Compute weighted score
        score = sum(
            self.weights.get(criterion, 0.0) * rubric_scores.get(criterion, 0.0)
            for criterion in self.weights.keys()
        )
        
        # Normalize to [0, 1]
        reward = max(0.0, min(1.0, score))
        
        success = reward >= 0.8  # Higher threshold for finance
        
        return VerificationResult(
            reward=reward,
            success=success,
            metadata={
                "rubric_scores": rubric_scores,
                "weights": self.weights,
            },
            trace_id=trace_id,
        )
    
    async def verify_async(
        self,
        observation: Any,
        action: Any,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        """Async version of verify."""
        return self.verify(observation, action, trace_id, metadata)
    
    async def verify_batch(
        self,
        requests: List[VerificationRequest],
    ) -> List[VerificationResult]:
        """
        Verify a batch of requests.
        
        Args:
            requests: List of VerificationRequest objects
            
        Returns:
            List of VerificationResult objects
        """
        results = []
        for req in requests:
            result = self.verify(
                req.observation,
                req.action,
                req.trace_id,
                req.metadata,
            )
            results.append(result)
        return results

