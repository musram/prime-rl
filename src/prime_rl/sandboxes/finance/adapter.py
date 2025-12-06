"""
Finance Reconciliation Sandbox Environment.

A reference implementation of a finance reconciliation sandbox that simulates
financial workflows with transactions, ledgers, and reconciliation tasks.
"""

import json
import uuid
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal

from prime_rl.core.environment import EnvironmentAdapter
from prime_rl.core.algorithms import UniversalRollout
from loguru import logger


@dataclass
class Transaction:
    """Financial transaction entity."""
    transaction_id: str
    account_id: str
    amount: Decimal
    description: str
    date: str
    category: str = "unknown"
    status: str = "pending"  # pending, matched, reconciled, exception
    matched_transaction_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LedgerEntry:
    """Ledger entry entity."""
    entry_id: str
    account_id: str
    amount: Decimal
    description: str
    date: str
    source: str = "unknown"  # bank, internal, external
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReconciliationTask:
    """Reconciliation task entity."""
    task_id: str
    account_id: str
    period_start: str
    period_end: str
    transactions: List[str] = field(default_factory=list)  # transaction IDs
    ledger_entries: List[str] = field(default_factory=list)  # entry IDs
    status: str = "open"  # open, in_progress, completed, exception
    exceptions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class FinanceReconciliationSandbox(EnvironmentAdapter):
    """
    Finance Reconciliation Sandbox Environment.
    
    Simulates a financial reconciliation system with:
    - Transactions (from various sources)
    - Ledger entries (internal accounting records)
    - Reconciliation tasks (matching transactions to ledger entries)
    
    Actions:
    - view_transaction: View transaction details
    - view_ledger_entry: View ledger entry details
    - match_transaction: Match transaction to ledger entry
    - create_reconciliation_task: Create a new reconciliation task
    - reconcile_account: Reconcile an account for a period
    - flag_exception: Flag a reconciliation exception
    
    Example:
        ```python
        sandbox = FinanceReconciliationSandbox()
        obs, info = sandbox.reset()
        obs, reward, done, truncated, info = sandbox.step({
            "action_type": "view_transaction",
            "transaction_id": "txn-1"
        })
        ```
    """
    
    def __init__(
        self,
        num_accounts: int = 5,
        num_transactions: int = 20,
        num_ledger_entries: int = 18,
        seed: Optional[int] = None,
    ):
        """
        Initialize Finance Reconciliation Sandbox.
        
        Args:
            num_accounts: Number of synthetic accounts
            num_transactions: Number of transactions
            num_ledger_entries: Number of ledger entries
            seed: Random seed for reproducibility
        """
        self.num_accounts = num_accounts
        self.num_transactions = num_transactions
        self.num_ledger_entries = num_ledger_entries
        self.seed = seed
        
        # State
        self.accounts: List[str] = []
        self.transactions: Dict[str, Transaction] = {}
        self.ledger_entries: Dict[str, LedgerEntry] = {}
        self.reconciliation_tasks: Dict[str, ReconciliationTask] = {}
        
        # Episode state
        self._current_task_id: Optional[str] = None
        self._episode_started = False
        self._step_count = 0
        self._max_steps = 30
        
        # Trajectory tracking
        self._trajectory: Dict[str, list] = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }
        
        # Initialize synthetic data
        self._initialize_data()
    
    def _initialize_data(self) -> None:
        """Initialize synthetic accounts, transactions, and ledger entries."""
        import random
        if self.seed is not None:
            random.seed(self.seed)
        
        # Create accounts
        self.accounts = [f"account-{i+1}" for i in range(self.num_accounts)]
        
        # Create transactions
        categories = ["payment", "refund", "fee", "transfer", "deposit"]
        base_date = datetime.utcnow() - timedelta(days=30)
        
        for i in range(self.num_transactions):
            transaction_id = f"txn-{i+1}"
            account_id = random.choice(self.accounts)
            amount = Decimal(str(round(random.uniform(-1000, 1000), 2)))
            date = (base_date + timedelta(days=random.randint(0, 29))).isoformat()
            
            self.transactions[transaction_id] = Transaction(
                transaction_id=transaction_id,
                account_id=account_id,
                amount=amount,
                description=f"Transaction {i+1}",
                date=date,
                category=random.choice(categories),
                status="pending",
            )
        
        # Create ledger entries (some matching, some not)
        for i in range(self.num_ledger_entries):
            entry_id = f"ledger-{i+1}"
            account_id = random.choice(self.accounts)
            # Try to match some entries to transactions
            if i < self.num_transactions - 2:
                txn = list(self.transactions.values())[i]
                amount = txn.amount
                date = txn.date
            else:
                amount = Decimal(str(round(random.uniform(-1000, 1000), 2)))
                date = (base_date + timedelta(days=random.randint(0, 29))).isoformat()
            
            self.ledger_entries[entry_id] = LedgerEntry(
                entry_id=entry_id,
                account_id=account_id,
                amount=amount,
                description=f"Ledger Entry {i+1}",
                date=date,
                source=random.choice(["bank", "internal", "external"]),
            )
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the finance sandbox to initial state.
        
        Args:
            seed: Optional random seed
            options: Optional reset options (e.g., account_id, period)
            
        Returns:
            Tuple of (observation, info_dict)
        """
        # Reset episode state
        self._step_count = 0
        self._trajectory = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }
        
        # Create or select reconciliation task
        account_id = options.get("account_id") if options else None
        if not account_id:
            account_id = self.accounts[0]
        
        period_start = options.get("period_start") if options else None
        period_end = options.get("period_end") if options else None
        
        if not period_start:
            period_start = (datetime.utcnow() - timedelta(days=30)).isoformat()
        if not period_end:
            period_end = datetime.utcnow().isoformat()
        
        # Create reconciliation task
        task_id = f"recon-task-{uuid.uuid4().hex[:8]}"
        account_transactions = [
            txn_id for txn_id, txn in self.transactions.items()
            if txn.account_id == account_id and period_start <= txn.date <= period_end
        ]
        account_ledger_entries = [
            entry_id for entry_id, entry in self.ledger_entries.items()
            if entry.account_id == account_id and period_start <= entry.date <= period_end
        ]
        
        task = ReconciliationTask(
            task_id=task_id,
            account_id=account_id,
            period_start=period_start,
            period_end=period_end,
            transactions=account_transactions,
            ledger_entries=account_ledger_entries,
            status="open",
        )
        
        self.reconciliation_tasks[task_id] = task
        self._current_task_id = task_id
        self._episode_started = True
        
        # Initial observation
        observation = {
            "task_id": task_id,
            "account_id": account_id,
            "period": {
                "start": period_start,
                "end": period_end,
            },
            "summary": {
                "num_transactions": len(account_transactions),
                "num_ledger_entries": len(account_ledger_entries),
                "unmatched_transactions": len(account_transactions),
                "unmatched_ledger_entries": len(account_ledger_entries),
            },
            "available_actions": [
                "view_transaction",
                "view_ledger_entry",
                "match_transaction",
                "reconcile_account",
                "flag_exception",
            ],
        }
        
        self._trajectory["observations"].append(observation)
        
        info = {
            "task_id": task_id,
            "account_id": account_id,
            "environment": "finance_reconciliation_sandbox",
        }
        
        return observation, info
    
    def step(
        self,
        action: Any,
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the finance sandbox.
        
        Args:
            action: Action dictionary with "action_type" and parameters
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info_dict)
        """
        if not self._episode_started:
            raise RuntimeError("Environment must be reset before stepping")
        
        if self._current_task_id is None:
            raise RuntimeError("No reconciliation task selected")
        
        # Parse action
        if isinstance(action, str):
            try:
                action = json.loads(action)
            except json.JSONDecodeError:
                action = {"action_type": action}
        
        if not isinstance(action, dict):
            action = {"action_type": str(action)}
        
        action_type = action.get("action_type", "unknown")
        self._step_count += 1
        
        # Execute action
        reward = 0.0
        observation = {}
        terminated = False
        truncated = False
        
        task = self.reconciliation_tasks[self._current_task_id]
        
        if action_type == "view_transaction":
            transaction_id = action.get("transaction_id")
            if transaction_id and transaction_id in self.transactions:
                txn = self.transactions[transaction_id]
                reward = 0.1
                observation = {
                    "transaction": {
                        "transaction_id": transaction_id,
                        "account_id": txn.account_id,
                        "amount": str(txn.amount),
                        "description": txn.description,
                        "date": txn.date,
                        "category": txn.category,
                        "status": txn.status,
                    },
                    "status": "viewed",
                }
            else:
                reward = -0.1
                observation = {"status": "transaction_not_found"}
        
        elif action_type == "view_ledger_entry":
            entry_id = action.get("entry_id")
            if entry_id and entry_id in self.ledger_entries:
                entry = self.ledger_entries[entry_id]
                reward = 0.1
                observation = {
                    "ledger_entry": {
                        "entry_id": entry_id,
                        "account_id": entry.account_id,
                        "amount": str(entry.amount),
                        "description": entry.description,
                        "date": entry.date,
                        "source": entry.source,
                    },
                    "status": "viewed",
                }
            else:
                reward = -0.1
                observation = {"status": "ledger_entry_not_found"}
        
        elif action_type == "match_transaction":
            transaction_id = action.get("transaction_id")
            entry_id = action.get("entry_id")
            
            if transaction_id in self.transactions and entry_id in self.ledger_entries:
                txn = self.transactions[transaction_id]
                entry = self.ledger_entries[entry_id]
                
                # Check if match is valid (same account, similar amount, similar date)
                if (txn.account_id == entry.account_id and
                    abs(txn.amount - entry.amount) < Decimal("0.01") and
                    abs((datetime.fromisoformat(txn.date) - datetime.fromisoformat(entry.date)).days) < 2):
                    txn.status = "matched"
                    txn.matched_transaction_id = entry_id
                    entry.metadata["matched_transaction_id"] = transaction_id
                    reward = 0.8  # High reward for correct match
                    observation = {
                        "status": "matched",
                        "transaction_id": transaction_id,
                        "entry_id": entry_id,
                    }
                else:
                    reward = -0.3  # Penalty for incorrect match
                    observation = {
                        "status": "match_invalid",
                        "reason": "account/amount/date mismatch",
                    }
            else:
                reward = -0.2
                observation = {"status": "invalid_ids"}
        
        elif action_type == "reconcile_account":
            # Check if all transactions are matched
            unmatched_txns = [
                txn_id for txn_id in task.transactions
                if self.transactions[txn_id].status == "pending"
            ]
            unmatched_entries = [
                entry_id for entry_id in task.ledger_entries
                if "matched_transaction_id" not in self.ledger_entries[entry_id].metadata
            ]
            
            if len(unmatched_txns) == 0 and len(unmatched_entries) == 0:
                task.status = "completed"
                reward = 2.0  # High reward for successful reconciliation
                observation = {
                    "status": "reconciled",
                    "message": "All transactions matched successfully",
                }
                terminated = True
            else:
                # Partial reconciliation
                matched_count = len(task.transactions) - len(unmatched_txns)
                reward = 0.5 * (matched_count / len(task.transactions))
                observation = {
                    "status": "partial",
                    "unmatched_transactions": len(unmatched_txns),
                    "unmatched_ledger_entries": len(unmatched_entries),
                }
        
        elif action_type == "flag_exception":
            exception_type = action.get("exception_type", "unknown")
            description = action.get("description", "")
            
            if len(description) > 10:
                task.exceptions.append({
                    "exception_id": str(uuid.uuid4()),
                    "type": exception_type,
                    "description": description,
                    "timestamp": datetime.utcnow().isoformat(),
                })
                task.status = "exception"
                reward = 0.3
                observation = {
                    "status": "exception_flagged",
                    "exception_type": exception_type,
                }
            else:
                reward = -0.1
                observation = {"status": "description_too_short"}
        
        else:
            reward = -0.2
            observation = {"status": "unknown_action", "action_type": action_type}
        
        # Check truncation
        if self._step_count >= self._max_steps:
            truncated = True
            reward -= 0.5
        
        # Track trajectory
        self._trajectory["observations"].append(observation)
        self._trajectory["actions"].append(action)
        self._trajectory["rewards"].append(reward)
        self._trajectory["dones"].append(terminated or truncated)
        
        info = {
            "step": self._step_count,
            "action_type": action_type,
            "task_id": self._current_task_id,
            "task_status": task.status,
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self) -> Optional[Any]:
        """Render current state."""
        if self._current_task_id:
            task = self.reconciliation_tasks[self._current_task_id]
            return {
                "task": task,
                "unmatched_transactions": [
                    txn_id for txn_id in task.transactions
                    if self.transactions[txn_id].status == "pending"
                ],
                "step": self._step_count,
            }
        return None
    
    def close(self) -> None:
        """Clean up resources."""
        self._current_task_id = None
        self._episode_started = False
        self._step_count = 0
        self._trajectory = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }
    
    @property
    def observation_space(self) -> Any:
        """Observation space (not formally defined)."""
        return None
    
    @property
    def action_space(self) -> Any:
        """Action space."""
        return [
            "view_transaction",
            "view_ledger_entry",
            "match_transaction",
            "reconcile_account",
            "flag_exception",
        ]
    
    def get_rollout(self) -> UniversalRollout:
        """Convert trajectory to UniversalRollout."""
        if not self._episode_started or len(self._trajectory["observations"]) == 0:
            raise RuntimeError("No trajectory data available. Reset and step the environment first.")
        
        prompt = json.dumps(self._trajectory["observations"][0])
        actions_str = [json.dumps(act) for act in self._trajectory["actions"]]
        completion = " ".join(actions_str) if len(actions_str) > 1 else (actions_str[0] if actions_str else "")
        total_reward = sum(self._trajectory["rewards"])
        
        metadata = {
            "num_steps": len(self._trajectory["actions"]),
            "task_id": self._current_task_id,
        }
        
        observations = None
        actions = None
        dones = None
        
        if len(self._trajectory["observations"]) > 1:
            observations = [json.dumps(obs) for obs in self._trajectory["observations"]]
            actions = [json.dumps(act) for act in self._trajectory["actions"]]
            dones = self._trajectory["dones"]
        
        return UniversalRollout(
            prompts=[prompt],
            completions=[completion],
            rewards=[total_reward],
            observations=observations,
            actions=actions,
            dones=dones,
            metadata=metadata,
        )

