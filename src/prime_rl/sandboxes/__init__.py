"""
Enterprise sandboxes for PRIME-RL.

This module provides reference implementations of application-level sandboxes
(CRM, Finance, etc.) that mirror real-world enterprise workflows.
"""

from prime_rl.sandboxes.crm.adapter import CRMSupportSandbox
from prime_rl.sandboxes.finance.adapter import FinanceReconciliationSandbox

__all__ = [
    "CRMSupportSandbox",
    "FinanceReconciliationSandbox",
]

