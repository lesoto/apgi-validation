#!/usr/bin/env python3
"""
Secret Management Policy Enforcer
=================================

Enforces strict secret management rules at startup:
- Mandatory check for APGI_MASTER_KEY in environment
- Ensures keys exist and are valid
- Prevents plaintext secret fallbacks
"""

import logging
import os
import sys
from pathlib import Path

# Add parent directory to path for standalone execution
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.secure_key_manager import get_secure_key_manager

logger = logging.getLogger("secret_policy")


def enforce_secret_policy():
    """
    Check environment and key status to enforce strict secret policy.
    Aborts application startup if policy is violated.
    """
    logger.info("Enforcing APGI Secret Management Policy...")

    # 1. Mandatory env checks
    master_key = os.environ.get("APGI_MASTER_KEY")
    if not master_key:
        logger.error("CRITICAL: APGI_MASTER_KEY is not set in environment.")
        logger.error("No plaintext fallbacks are allowed per policy. Aborting startup.")
        sys.exit(1)

    # 2. Verify Key Manager can load keys (validates encryption)
    try:
        manager = get_secure_key_manager()
        manager.get_pickle_secret_key()
        manager.get_backup_hmac_key()
    except Exception as e:
        logger.error(f"CRITICAL: Failed to load secure keys: {e}")
        logger.error("Key integrity check failed. Aborting startup.")
        sys.exit(1)

    # 3. Check for plaintext secrets in env
    forbidden_plaintexts = [
        "APGI_JWT_SECRET",
        "PICKLE_SECRET_KEY",
        "APGI_BACKUP_HMAC_KEY",
    ]

    for secret in forbidden_plaintexts:
        val = os.environ.get(secret)
        if val and (
            len(val) < 16
            or val in ["dev-fallback-secret-do-not-use-in-prod", "secret", "password"]
        ):
            logger.error(f"CRITICAL: Weak or plaintext fallback detected in {secret}.")
            logger.error("Plaintext secret fallbacks are forbidden. Aborting startup.")
            sys.exit(1)

    logger.info("Secret Management Policy checks passed successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    enforce_secret_policy()
