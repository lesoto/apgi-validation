"""Data Transfer Objects (DTOs) for APGI services.

This module defines standardized Pydantic models for service inputs/outputs,
ensuring stable contracts between layers and removing ad-hoc dict drift.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Add parent directory to path for standalone execution
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.protocol_schema import ProtocolResult


class PerformanceMetricDTO(BaseModel):
    """DTO for performance metrics."""

    p95_latency_ms: float
    throughput_ops_per_sec: float
    cpu_usage_percent: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ValidationTierSummaryDTO(BaseModel):
    """Summary statistics for a validation tier."""

    passed: int = 0
    total: int = 0
    pending: int = 0
    success_rate: float = 0.0


class MasterValidationReportDTO(BaseModel):
    """DTO for the comprehensive master validation report."""

    overall_decision: str
    total_protocols: int
    completed_protocols: int
    passed_protocols: int
    pending_protocols: int
    success_rate: float
    weighted_score: float
    tier_summary: Dict[str, ValidationTierSummaryDTO]
    protocol_results: Dict[str, ProtocolResult]
    falsification_status: Dict[str, List[Any]] = Field(default_factory=dict)
    summary: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ServiceResponseDTO(BaseModel):
    """Generic wrapper for service responses."""

    success: bool
    data: Optional[Any] = None
    error_code: Optional[str] = None
    message: Optional[str] = None
    trace_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AuditLogDTO(BaseModel):
    """DTO for audit logging entries."""

    event_type: str
    user_id: str
    action: str
    resource: str
    status: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    details: Dict[str, Any] = Field(default_factory=dict)
    correlation_id: Optional[str] = None


class ConfigExplainDTO(BaseModel):
    """DTO for the result of 'config explain' command."""

    resolved_config: Dict[str, Any]
    sources: Dict[str, str]  # Param name -> source (env, file, default)
    precedence_order: List[str]


class ErrorResponseDTO(BaseModel):
    """DTO for standardized error responses across the framework."""

    success: bool = False
    code: str
    message: str
    category: str
    severity: str
    remediation: Optional[str] = None
    correlation_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    context: Dict[str, Any] = Field(default_factory=dict)
    traceback: Optional[str] = None
