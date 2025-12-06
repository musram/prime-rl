"""
Pytest-based verifier for replication training.

This verifier runs pytest (or configurable test command) in an isolated
environment and converts test results into scalar rewards.
"""

import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from prime_rl.core.verifier import VerifierClient, VerificationRequest, VerificationResult
from loguru import logger


class PytestVerifierClient(VerifierClient):
    """
    Pytest-based verifier for replication training.
    
    Checks out/loads a target repository, runs pytest (or configurable command)
    in an isolated environment, and converts test results into scalar rewards.
    
    Example:
        ```python
        verifier = PytestVerifierClient(
            repo_path=Path("benchmarks/replication/project_a"),
            command="pytest -q",
        )
        
        result = verifier.verify(
            observation={"code": "def func(): return True"},
            action={"file": "src/module.py", "content": "..."},
        )
        ```
    """
    
    def __init__(
        self,
        repo_path: Path,
        command: str = "pytest -q",
        timeout: int = 60,
        work_dir: Optional[Path] = None,
    ):
        """
        Initialize Pytest verifier.
        
        Args:
            repo_path: Path to repository or code directory
            command: Test command to run (default: "pytest -q")
            timeout: Timeout in seconds for test execution
            work_dir: Optional working directory (uses temp dir if not provided)
        """
        self.repo_path = Path(repo_path)
        self.command = command
        self.timeout = timeout
        self.work_dir = work_dir
        
        if not self.repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")
    
    def verify(
        self,
        observation: Any,
        action: Any,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        """
        Verify code by running tests.
        
        Args:
            observation: Environment observation (can include code context)
            action: Agent action (should include file path and content)
            trace_id: Optional trace identifier
            metadata: Optional metadata
            
        Returns:
            VerificationResult with reward based on test results
        """
        # Extract code changes from action
        if isinstance(action, dict):
            file_path = action.get("file")
            content = action.get("content", action.get("code", ""))
        else:
            file_path = None
            content = str(action)
        
        if not file_path or not content:
            return VerificationResult(
                reward=0.0,
                success=False,
                metadata={"error": "No file path or content provided"},
                trace_id=trace_id,
            )
        
        # Create temporary working directory
        with tempfile.TemporaryDirectory() as temp_dir:
            work_path = Path(temp_dir)
            
            # Copy repository to temp directory
            if self.repo_path.is_dir():
                shutil.copytree(self.repo_path, work_path / "repo", dirs_exist_ok=True)
                repo_work_path = work_path / "repo"
            else:
                # Single file
                shutil.copy(self.repo_path, work_path / self.repo_path.name)
                repo_work_path = work_path
            
            # Write modified file
            target_file = repo_work_path / file_path
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_text(content)
            
            # Run tests
            try:
                result = self._run_tests(repo_work_path)
                reward = self._compute_reward(result)
                success = result.get("all_passed", False)
                
                return VerificationResult(
                    reward=reward,
                    success=success,
                    metadata={
                        "test_result": result,
                        "file": file_path,
                    },
                    trace_id=trace_id,
                )
            except Exception as e:
                logger.error(f"Test execution failed: {e}")
                return VerificationResult(
                    reward=0.0,
                    success=False,
                    metadata={"error": str(e)},
                    trace_id=trace_id,
                )
    
    def _run_tests(self, work_path: Path) -> Dict[str, Any]:
        """
        Run tests in the working directory.
        
        Args:
            work_path: Path to working directory
            
        Returns:
            Dictionary with test results
        """
        # Parse command
        cmd_parts = self.command.split()
        if not cmd_parts:
            cmd_parts = ["pytest", "-q"]
        
        # Run command
        try:
            result = subprocess.run(
                cmd_parts,
                cwd=work_path,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            
            stdout = result.stdout
            stderr = result.stderr
            
            # Parse pytest output (simplified)
            passed = result.returncode == 0
            num_passed = stdout.count("PASSED") if passed else 0
            num_failed = stdout.count("FAILED")
            num_total = num_passed + num_failed
            
            # Try to extract more detailed info
            if "passed" in stdout.lower():
                # Simple parsing
                lines = stdout.split("\n")
                for line in lines:
                    if "passed" in line.lower() and "failed" in line.lower():
                        # Format: "X passed, Y failed"
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.isdigit() and i + 1 < len(parts):
                                if "passed" in parts[i + 1].lower():
                                    num_passed = int(part)
                                elif "failed" in parts[i + 1].lower():
                                    num_failed = int(part)
                        num_total = num_passed + num_failed
                        break
            
            return {
                "all_passed": passed and num_failed == 0,
                "num_passed": num_passed,
                "num_failed": num_failed,
                "num_total": num_total,
                "returncode": result.returncode,
                "stdout": stdout,
                "stderr": stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "all_passed": False,
                "num_passed": 0,
                "num_failed": 0,
                "num_total": 0,
                "returncode": -1,
                "stdout": "",
                "stderr": "Test execution timed out",
                "timeout": True,
            }
    
    def _compute_reward(self, result: Dict[str, Any]) -> float:
        """
        Compute reward from test results.
        
        Args:
            result: Test result dictionary
            
        Returns:
            Reward value in [0, 1]
        """
        if result.get("timeout"):
            return 0.0
        
        if result.get("all_passed"):
            return 1.0
        
        num_passed = result.get("num_passed", 0)
        num_total = result.get("num_total", 1)
        
        if num_total == 0:
            return 0.0
        
        # Reward proportional to pass rate
        reward = num_passed / num_total
        
        return reward
    
    async def verify_async(
        self,
        observation: Any,
        action: Any,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        """Async version of verify (runs synchronously)."""
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

