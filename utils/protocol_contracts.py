"""
LEVEL DESIGNATION: Level 2 (information-theoretic)

Bridge to Level 1
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping

from pydantic import BaseModel

# Add parent directory to path for standalone execution
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.error_handler import ConfigurationError, ProtocolError


class ProtocolContract(BaseModel):
    """Static contract for protocol registration."""

    protocol_id: str
    file: str
    entrypoint: str
    schema_version: str = "1.0"
    folder: str = ""
    """Optional sub-folder relative to project_root (e.g. 'Validation',
    'Falsification', 'Theory', 'protocols').  When empty, ``file`` is
    treated as a path relative to project_root directly."""


class ProtocolContractRegistry:
    """Registry enforcing protocol metadata contracts."""

    def __init__(self, contracts: Iterable[ProtocolContract]):
        self._contracts = {contract.protocol_id: contract for contract in contracts}

    @property
    def contracts(self) -> Mapping[str, ProtocolContract]:
        return self._contracts

    def validate(self, project_root: Path) -> Dict[str, str]:
        """Validate contract completeness and protocol artifact presence."""
        diagnostics: Dict[str, str] = {}
        for protocol_id, contract in self._contracts.items():
            if not contract.schema_version:
                raise ConfigurationError(
                    f"Missing schema_version for {protocol_id}",
                    config_file="protocol_contracts",
                )
            if not contract.entrypoint:
                raise ConfigurationError(
                    f"Missing entrypoint for {protocol_id}",
                    config_file="protocol_contracts",
                )

            if contract.folder:
                protocol_path = project_root / contract.folder / contract.file
            else:
                # file already carries a full relative path (e.g. "Validation/VP_01.py")
                protocol_path = project_root / contract.file

            if not protocol_path.exists():
                raise ProtocolError(
                    f"Protocol file not found: {protocol_path}",
                    protocol_name=protocol_id,
                )

            diagnostics[protocol_id] = f"OK schema={contract.schema_version} entrypoint={contract.entrypoint}"
        return diagnostics
