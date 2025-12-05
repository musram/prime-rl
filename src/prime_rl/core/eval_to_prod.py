"""
Eval-to-Prod dashboard integration for correlation tracking.

This module provides pre-configured dashboards (e.g., via W&B) for tracking
eval-to-prod correlation as specified in the PRD (§4.3).
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import wandb

from prime_rl.core.training_run import TrainingRun


class EvalToProdTracker:
    """
    Tracks eval-to-prod correlation metrics.
    
    Provides pre-configured W&B dashboards for tracking how evaluation metrics
    correlate with production performance.
    """
    
    def __init__(
        self,
        run: TrainingRun,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
    ):
        """
        Initialize eval-to-prod tracker.
        
        Args:
            run: TrainingRun instance
            wandb_project: Optional W&B project name
            wandb_run_name: Optional W&B run name
        """
        self.run = run
        self.wandb_project = wandb_project or f"prime-rl-{run.project_id}"
        self.wandb_run_name = wandb_run_name or run.run_id
        
        # Initialize W&B run if not already initialized
        self.wandb_run = None
        self._init_wandb()
    
    def _init_wandb(self) -> None:
        """Initialize W&B run for tracking."""
        try:
            self.wandb_run = wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                id=self.run.run_id,
                resume="allow",
                config={
                    "run_id": self.run.run_id,
                    "project_id": self.run.project_id,
                    "code_revision": self.run.code_revision,
                    **self.run.config,
                },
            )
        except Exception:
            # W&B not available or not configured
            self.wandb_run = None
    
    def log_eval_metrics(
        self,
        step: int,
        eval_metrics: Dict[str, float],
        environment: Optional[str] = None,
    ) -> None:
        """
        Log evaluation metrics.
        
        Args:
            step: Training step
            eval_metrics: Dictionary of evaluation metrics
            environment: Optional environment identifier
        """
        if not self.wandb_run:
            return
        
        metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
        if environment:
            metrics["eval/environment"] = environment
        metrics["step"] = step
        
        wandb.log(metrics, step=step)
    
    def log_prod_metrics(
        self,
        step: int,
        prod_metrics: Dict[str, float],
        environment: Optional[str] = None,
    ) -> None:
        """
        Log production metrics.
        
        Args:
            step: Training step
            prod_metrics: Dictionary of production metrics
            environment: Optional environment identifier
        """
        if not self.wandb_run:
            return
        
        metrics = {f"prod/{k}": v for k, v in prod_metrics.items()}
        if environment:
            metrics["prod/environment"] = environment
        metrics["step"] = step
        
        wandb.log(metrics, step=step)
    
    def log_correlation(
        self,
        step: int,
        eval_metric: str,
        prod_metric: str,
        correlation: float,
    ) -> None:
        """
        Log eval-to-prod correlation.
        
        Args:
            step: Training step
            eval_metric: Evaluation metric name
            prod_metric: Production metric name
            correlation: Correlation coefficient
        """
        if not self.wandb_run:
            return
        
        wandb.log(
            {
                f"correlation/{eval_metric}_to_{prod_metric}": correlation,
                "step": step,
            },
            step=step,
        )
    
    def log_comparison(
        self,
        step: int,
        comparisons: List[Dict[str, Any]],
    ) -> None:
        """
        Log comparison between eval and prod.
        
        Args:
            step: Training step
            comparisons: List of comparison dictionaries with keys:
                - eval_value: Evaluation metric value
                - prod_value: Production metric value
                - metric_name: Name of the metric
                - environment: Optional environment identifier
        """
        if not self.wandb_run:
            return
        
        for comp in comparisons:
            metrics = {
                f"comparison/{comp['metric_name']}/eval": comp["eval_value"],
                f"comparison/{comp['metric_name']}/prod": comp["prod_value"],
                f"comparison/{comp['metric_name']}/diff": comp["prod_value"] - comp["eval_value"],
                "step": step,
            }
            if "environment" in comp:
                metrics[f"comparison/{comp['metric_name']}/environment"] = comp["environment"]
            
            wandb.log(metrics, step=step)
    
    def finish(self) -> None:
        """Finish tracking and close W&B run."""
        if self.wandb_run:
            wandb.finish()

