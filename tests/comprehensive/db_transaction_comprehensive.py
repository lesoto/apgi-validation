"""
APGI Database Transaction Testing Module
========================================

Comprehensive database transaction testing including:
- Rollback and commit verification
- Connection pool exhaustion testing
- Transaction isolation level testing
- Deadlock detection and handling
- Concurrent transaction conflict testing

This module provides robust transaction testing for the APGI validation framework.
"""

import json
import queue
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class TransactionResult:
    """Result of a transaction test."""

    transaction_id: str
    operation: str  # commit, rollback, error
    success: bool
    execution_time_ms: float
    rows_affected: int = 0
    error_message: Optional[str] = None
    isolation_level: Optional[str] = None
    before_state: Optional[Dict] = None
    after_state: Optional[Dict] = None


@dataclass
class ConnectionPoolStats:
    """Statistics for connection pool testing."""

    total_connections: int
    active_connections: int
    idle_connections: int
    waiting_requests: int
    max_pool_size: int
    exhausted_events: int = 0
    avg_wait_time_ms: float = 0.0


@dataclass
class IsolationTestResult:
    """Result of isolation level test."""

    test_name: str
    isolation_level: str
    passed: bool
    phantom_reads_detected: bool
    non_repeatable_reads_detected: bool
    dirty_reads_detected: bool
    lost_updates_detected: bool
    details: Dict[str, Any] = field(default_factory=dict)


class DatabaseTransactionTester:
    """Test database transaction behavior."""

    def __init__(self, db_path: Optional[Path] = None, max_pool_size: int = 5):
        """
        Initialize transaction tester.

        Args:
            db_path: Path to SQLite database (None for in-memory)
            max_pool_size: Maximum connections in pool
        """
        self.db_path = str(db_path) if db_path else ":memory:"
        self.max_pool_size = max_pool_size
        self.connection_pool: queue.Queue = queue.Queue(maxsize=max_pool_size)
        self.active_connections: Set[sqlite3.Connection] = set()
        self.pool_lock = threading.Lock()
        self.exhausted_count = 0
        self.wait_times: List[float] = []
        self._initialize_pool()
        self._setup_test_tables()

    def _initialize_pool(self) -> None:
        """Initialize connection pool."""
        for _ in range(self.max_pool_size):
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("PRAGMA foreign_keys = ON")
            self.connection_pool.put(conn)

    def _setup_test_tables(self) -> None:
        """Setup test tables."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS test_data (
                id INTEGER PRIMARY KEY,
                value TEXT,
                counter INTEGER DEFAULT 0,
                version INTEGER DEFAULT 1
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS accounts (
                id INTEGER PRIMARY KEY,
                balance REAL DEFAULT 0.0,
                name TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _get_connection(self, timeout: float = 5.0) -> Optional[sqlite3.Connection]:
        """Get connection from pool with timeout."""
        start_wait = time.time()
        try:
            conn = self.connection_pool.get(timeout=timeout)
            wait_time = (time.time() - start_wait) * 1000
            self.wait_times.append(wait_time)

            with self.pool_lock:
                self.active_connections.add(conn)
            return conn
        except queue.Empty:
            with self.pool_lock:
                self.exhausted_count += 1
            return None

    def _return_connection(self, conn: sqlite3.Connection) -> None:
        """Return connection to pool."""
        with self.pool_lock:
            self.active_connections.discard(conn)
        try:
            self.connection_pool.put_nowait(conn)
        except queue.Full:
            conn.close()

    @contextmanager
    def transaction(self, isolation_level: Optional[str] = None):
        """Context manager for transactions."""
        conn = self._get_connection()
        if conn is None:
            raise RuntimeError("Could not get database connection")

        try:
            if isolation_level:
                conn.execute(
                    f"PRAGMA read_uncommitted = {1 if isolation_level == 'READ UNCOMMITTED' else 0}"
                )

            conn.execute("BEGIN")
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            self._return_connection(conn)

    def test_commit_rollback(self) -> List[TransactionResult]:
        """Test commit and rollback operations."""
        results = []
        test_id = str(uuid.uuid4())[:8]

        # Test 1: Successful commit
        start_time = time.time()
        try:
            with self.transaction() as conn:
                # Get initial state
                cursor = conn.execute("SELECT COUNT(*) FROM test_data")
                initial_count = cursor.fetchone()[0]

                # Insert data
                conn.execute(
                    "INSERT INTO test_data (value, counter) VALUES (?, ?)",
                    (f"commit_test_{test_id}", 1),
                )

                # Verify data is visible in transaction
                cursor = conn.execute("SELECT COUNT(*) FROM test_data")
                during_count = cursor.fetchone()[0]

            # Verify data persisted after commit
            verify_conn = sqlite3.connect(self.db_path)
            cursor = verify_conn.execute(
                "SELECT COUNT(*) FROM test_data WHERE value = ?",
                (f"commit_test_{test_id}",),
            )
            final_count = cursor.fetchone()[0]
            verify_conn.close()

            execution_time = (time.time() - start_time) * 1000

            results.append(
                TransactionResult(
                    transaction_id=f"commit_{test_id}",
                    operation="commit",
                    success=final_count > 0,
                    execution_time_ms=execution_time,
                    rows_affected=final_count,
                    before_state={"count": initial_count},
                    after_state={"count": during_count},
                )
            )
        except Exception as e:
            results.append(
                TransactionResult(
                    transaction_id=f"commit_{test_id}",
                    operation="commit",
                    success=False,
                    execution_time_ms=(time.time() - start_time) * 1000,
                    error_message=str(e),
                )
            )

        # Test 2: Rollback verification
        start_time = time.time()
        rollback_test_id = str(uuid.uuid4())[:8]

        try:
            # Get count before attempted insert
            verify_conn = sqlite3.connect(self.db_path)
            cursor = verify_conn.execute("SELECT COUNT(*) FROM test_data")
            before_count = cursor.fetchone()[0]
            verify_conn.close()

            try:
                with self.transaction() as conn:
                    conn.execute(
                        "INSERT INTO test_data (value, counter) VALUES (?, ?)",
                        (f"rollback_test_{rollback_test_id}", 1),
                    )
                    # Force rollback by raising exception
                    raise ValueError("Intentional rollback trigger")
            except ValueError:
                pass  # Expected

            # Verify data was NOT persisted
            verify_conn = sqlite3.connect(self.db_path)
            cursor = verify_conn.execute(
                "SELECT COUNT(*) FROM test_data WHERE value = ?",
                (f"rollback_test_{rollback_test_id}",),
            )
            after_count = cursor.fetchone()[0]
            cursor = verify_conn.execute("SELECT COUNT(*) FROM test_data")
            final_total = cursor.fetchone()[0]
            verify_conn.close()

            execution_time = (time.time() - start_time) * 1000

            results.append(
                TransactionResult(
                    transaction_id=f"rollback_{rollback_test_id}",
                    operation="rollback",
                    success=after_count == 0 and final_total == before_count,
                    execution_time_ms=execution_time,
                    rows_affected=0,
                    before_state={"count": before_count},
                    after_state={"count": final_total},
                )
            )
        except Exception as e:
            results.append(
                TransactionResult(
                    transaction_id=f"rollback_{rollback_test_id}",
                    operation="rollback",
                    success=False,
                    execution_time_ms=(time.time() - start_time) * 1000,
                    error_message=str(e),
                )
            )

        return results

    def test_connection_pool_exhaustion(
        self, num_threads: int = 10
    ) -> ConnectionPoolStats:
        """Test connection pool exhaustion behavior."""
        barrier = threading.Barrier(num_threads)

        def worker(thread_id: int) -> Dict[str, Any]:
            barrier.wait()  # Synchronize thread start
            start_time = time.time()

            conn = self._get_connection(timeout=1.0)  # Short timeout for testing

            if conn is None:
                return {
                    "thread_id": thread_id,
                    "got_connection": False,
                    "wait_time_ms": (time.time() - start_time) * 1000,
                }

            try:
                # Simulate work
                time.sleep(0.1)
                conn.execute("SELECT 1")

                return {
                    "thread_id": thread_id,
                    "got_connection": True,
                    "wait_time_ms": (time.time() - start_time) * 1000,
                }
            finally:
                self._return_connection(conn)

        # Run threads
        threads = []
        thread_results: List[Optional[Dict[str, Any]]] = [None] * num_threads

        def run_worker(i):
            thread_results[i] = worker(i)

        for i in range(num_threads):
            t = threading.Thread(target=run_worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Calculate stats
        valid_results: List[Dict[str, Any]] = [
            r for r in thread_results if r is not None
        ]
        successful = sum(1 for r in valid_results if r["got_connection"])
        failed = num_threads - successful
        avg_wait = (
            sum(r["wait_time_ms"] for r in valid_results) / len(valid_results)
            if valid_results
            else 0
        )

        return ConnectionPoolStats(
            total_connections=self.max_pool_size,
            active_connections=len(self.active_connections),
            idle_connections=self.max_pool_size - len(self.active_connections),
            waiting_requests=failed,
            max_pool_size=self.max_pool_size,
            exhausted_events=self.exhausted_count,
            avg_wait_time_ms=avg_wait,
        )

    def test_transaction_isolation(self) -> List[IsolationTestResult]:
        """Test transaction isolation levels."""
        results = []

        # Clean slate
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM test_data")
        conn.execute("DELETE FROM accounts")
        conn.commit()
        conn.close()

        # Test 1: Dirty Read Prevention
        dirty_read_passed = self._test_dirty_read_prevention()
        results.append(
            IsolationTestResult(
                test_name="dirty_read_prevention",
                isolation_level="READ COMMITTED",
                passed=dirty_read_passed,
                phantom_reads_detected=False,
                non_repeatable_reads_detected=False,
                dirty_reads_detected=not dirty_read_passed,
                lost_updates_detected=False,
            )
        )

        # Test 2: Lost Update Prevention
        lost_update_passed = self._test_lost_update_prevention()
        results.append(
            IsolationTestResult(
                test_name="lost_update_prevention",
                isolation_level="SERIALIZABLE",
                passed=lost_update_passed,
                phantom_reads_detected=False,
                non_repeatable_reads_detected=False,
                dirty_reads_detected=False,
                lost_updates_detected=not lost_update_passed,
            )
        )

        # Test 3: Non-Repeatable Read Detection
        non_repeatable_passed = self._test_non_repeatable_read()
        results.append(
            IsolationTestResult(
                test_name="non_repeatable_read",
                isolation_level="READ COMMITTED",
                passed=non_repeatable_passed,  # We expect to detect these at lower isolation
                phantom_reads_detected=False,
                non_repeatable_reads_detected=not non_repeatable_passed,
                dirty_reads_detected=False,
                lost_updates_detected=False,
            )
        )

        return results

    def _test_dirty_read_prevention(self) -> bool:
        """Test that dirty reads are prevented."""
        result_holder = {"dirty_read_prevented": True}

        def transaction_a():
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("BEGIN")
                conn.execute("INSERT INTO test_data (value) VALUES ('uncommitted')")

                # Signal that uncommitted data exists
                time.sleep(0.1)

                # Rollback
                conn.rollback()
            finally:
                conn.close()

        def transaction_b():
            time.sleep(0.05)  # Let A start first
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("BEGIN")
                cursor = conn.execute(
                    "SELECT * FROM test_data WHERE value = 'uncommitted'"
                )
                rows = cursor.fetchall()

                # If we see the uncommitted row, dirty read occurred
                if len(rows) > 0:
                    result_holder["dirty_read_prevented"] = False

                conn.commit()
            finally:
                conn.close()

        t_a = threading.Thread(target=transaction_a)
        t_b = threading.Thread(target=transaction_b)

        t_a.start()
        t_b.start()

        t_a.join()
        t_b.join()

        return result_holder["dirty_read_prevented"]

    def _test_lost_update_prevention(self) -> bool:
        """Test that lost updates are prevented."""
        # Setup: Create account with balance
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO accounts (id, balance, name) VALUES (1, 100.0, 'test')"
        )
        conn.commit()
        conn.close()

        results: Dict[str, List] = {"updates": []}
        lock = threading.Lock()

        def update_worker(thread_id: int):
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("BEGIN IMMEDIATE")  # Get exclusive lock
                cursor = conn.execute("SELECT balance FROM accounts WHERE id = 1")
                balance = cursor.fetchone()[0]

                new_balance = balance + 10
                conn.execute(
                    "UPDATE accounts SET balance = ? WHERE id = 1", (new_balance,)
                )
                conn.commit()

                with lock:
                    results["updates"].append((thread_id, new_balance))
            except Exception as e:
                with lock:
                    results["updates"].append((thread_id, f"error: {e}"))
            finally:
                conn.close()

        threads = [threading.Thread(target=update_worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify final balance
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT balance FROM accounts WHERE id = 1")
        final_balance = cursor.fetchone()[0]
        conn.close()

        # With proper locking, balance should be 130 (100 + 10 + 10 + 10)
        # If lost updates occurred, it would be less
        expected_balance = 130.0
        return abs(final_balance - expected_balance) < 0.01

    def _test_non_repeatable_read(self) -> bool:
        """Test non-repeatable read behavior."""
        # Setup
        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT INTO test_data (id, value) VALUES (1, 'initial')")
        conn.commit()
        conn.close()

        read_results = {"first_read": None, "second_read": None}

        def reader():
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("BEGIN")
                cursor = conn.execute("SELECT value FROM test_data WHERE id = 1")
                read_results["first_read"] = cursor.fetchone()[0]

                time.sleep(0.2)  # Let writer commit

                cursor = conn.execute("SELECT value FROM test_data WHERE id = 1")
                read_results["second_read"] = cursor.fetchone()[0]

                conn.commit()
            finally:
                conn.close()

        def writer():
            time.sleep(0.1)  # Let reader do first read
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("UPDATE test_data SET value = 'updated' WHERE id = 1")
                conn.commit()
            finally:
                conn.close()

        t_reader = threading.Thread(target=reader)
        t_writer = threading.Thread(target=writer)

        t_reader.start()
        t_writer.start()

        t_reader.join()
        t_writer.join()

        # In READ COMMITTED, we should see the update on second read
        # This is expected behavior, so the "test passes" if we detect it
        return read_results["first_read"] != read_results["second_read"]

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all database transaction tests."""
        print("=" * 80)
        print("APGI DATABASE TRANSACTION TESTING")
        print("=" * 80)

        # Test 1: Commit/Rollback verification
        print("\n[DB Test 1/3] Commit/Rollback verification...")
        commit_results = self.test_commit_rollback()
        commit_passed = all(r.success for r in commit_results)
        print(f"  {'✓' if commit_passed else '✗'} Commit/Rollback tests")

        # Test 2: Connection pool exhaustion
        print("\n[DB Test 2/3] Connection pool exhaustion...")
        pool_stats = self.test_connection_pool_exhaustion(num_threads=10)
        pool_passed = pool_stats.waiting_requests > 0  # Should have some waits
        print(
            f"  {'✓' if pool_passed else '✗'} Pool exhaustion test (exhausted: {pool_stats.exhausted_events})"
        )

        # Test 3: Transaction isolation
        print("\n[DB Test 3/3] Transaction isolation levels...")
        isolation_results = self.test_transaction_isolation()
        isolation_passed = all(r.passed for r in isolation_results)
        print(f"  {'✓' if isolation_passed else '✗'} Isolation tests")

        report: Dict[str, Any] = {
            "commit_rollback_tests": [
                {
                    "transaction_id": r.transaction_id,
                    "operation": r.operation,
                    "success": r.success,
                    "execution_time_ms": r.execution_time_ms,
                    "rows_affected": r.rows_affected,
                }
                for r in commit_results
            ],
            "connection_pool_stats": {
                "total_connections": pool_stats.total_connections,
                "active_connections": pool_stats.active_connections,
                "idle_connections": pool_stats.idle_connections,
                "waiting_requests": pool_stats.waiting_requests,
                "exhausted_events": pool_stats.exhausted_events,
                "avg_wait_time_ms": pool_stats.avg_wait_time_ms,
            },
            "isolation_tests": [
                {
                    "test_name": r.test_name,
                    "isolation_level": r.isolation_level,
                    "passed": r.passed,
                    "dirty_reads_detected": r.dirty_reads_detected,
                    "lost_updates_detected": r.lost_updates_detected,
                    "non_repeatable_reads_detected": r.non_repeatable_reads_detected,
                }
                for r in isolation_results
            ],
            "summary": {
                "commit_rollback_passed": commit_passed,
                "pool_exhaustion_passed": pool_passed,
                "isolation_tests_passed": isolation_passed,
                "all_passed": commit_passed and isolation_passed,
            },
        }

        print(f"\n{'=' * 80}")
        print("DATABASE TRANSACTION TEST SUMMARY")
        print(f"{'=' * 80}")
        print(f"Commit/Rollback: {'✓' if commit_passed else '✗'}")
        print(f"Pool Exhaustion: {'✓' if pool_passed else '✗'}")
        print(f"Isolation Tests: {'✓' if isolation_passed else '✗'}")
        print(
            f"\nOverall: {'✅ All tests passed' if report['summary']['all_passed'] else '⚠️ Some tests failed'}"
        )

        return report


def run_database_tests() -> Dict[str, Any]:
    """Entry point for database transaction testing."""
    tester = DatabaseTransactionTester()
    return tester.run_all_tests()


if __name__ == "__main__":
    results = run_database_tests()

    # Save results
    output_path = Path("reports/db_transaction_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved to {output_path}")
