import pytest

from utils.error_handler import (
    APGIError,
    APGIImportWarning,
    ConfigurationError,
    DataError,
    ErrorHandler,
    ErrorInfo,
    ProtocolError,
    ValidationError,
    config_error,
    critical_error,
    data_error,
    error_boundary,
    format_user_message,
    get_error_summary,
    get_recovery_suggestions,
    handle_error,
    handle_errors,
    handle_import_error,
    import_error,
    io_error,
    safe_execute,
    simulation_error,
    validation_error,
)
from utils.errors import ErrorCategory, ErrorCode, ErrorSeverity


def test_error_info():
    info = ErrorInfo(
        category=ErrorCategory.RUNTIME,
        severity=ErrorSeverity.MEDIUM,
        code=ErrorCode.GEN_UNKNOWN,
        message="test",
    )
    assert info.message == "test"
    assert info.details == {}


def test_apgi_error_init():
    err1 = APGIError("msg")
    assert err1.message == "msg"
    assert err1.category == ErrorCategory.RUNTIME

    info = ErrorInfo(
        category=ErrorCategory.DATA,
        severity=ErrorSeverity.HIGH,
        code=ErrorCode.GEN_UNKNOWN,
        message="test_info",
    )
    err2 = APGIError(info)
    assert err2.message == "test_info"
    assert err2.category == ErrorCategory.DATA
    assert err2.error_info is info


def test_apgi_error_to_dict():
    err = APGIError("msg")
    d = err.to_dict()
    assert d["message"] == "msg"
    assert "severity" in d


def test_apgi_error_str():
    err = APGIError("msg", severity=ErrorSeverity.CRITICAL, category=ErrorCategory.DATA)
    assert "[CRITICAL]" in str(err)
    assert "DATA" in str(err)
    assert "msg" in str(err)


def test_subclasses():
    assert isinstance(ValidationError("val fail", "field"), APGIError)
    assert isinstance(ConfigurationError("conf fail", "file.py"), APGIError)
    assert isinstance(ProtocolError("prot fail", "proto_name"), APGIError)
    assert isinstance(DataError("data fail", "db"), APGIError)
    assert isinstance(APGIImportWarning("import warn", "pkg"), APGIError)


def test_error_handler_create_error():
    handler = ErrorHandler()
    info = handler.create_error(
        ErrorCategory.CONFIGURATION,
        ErrorSeverity.CRITICAL,
        "INVALID_CONFIG",
        details="bad json",
    )
    assert "corrupted or invalid" in info.message
    assert "bad json" in info.message


def test_error_handler_handle_error():
    handler = ErrorHandler()
    err = handler.handle_error(
        ErrorCategory.CONFIGURATION,
        ErrorSeverity.CRITICAL,
        "INVALID_CONFIG",
        details="bad json",
    )
    assert isinstance(err, APGIError)
    assert err.category == ErrorCategory.CONFIGURATION


def test_error_handler_register_handler():
    handler = ErrorHandler()
    calls = []
    handler.register_handler(ErrorCategory.DATA, lambda e: calls.append(e))
    handler.handle_error(
        ErrorCategory.DATA, ErrorSeverity.MEDIUM, ErrorCode.GEN_UNKNOWN
    )
    assert len(calls) == 1


def test_handle_errors_decorator():
    @handle_errors(ErrorCategory.RUNTIME, reraise=False)
    def fail():
        raise ValueError("inner")

    assert fail() is None

    @handle_errors(ErrorCategory.RUNTIME, reraise=True)
    def fail2():
        raise ValueError("inner")

    with pytest.raises(APGIError):
        fail2()


def test_convenience_functions():
    assert isinstance(config_error("INVALID_CONFIG", details="x"), APGIError)
    assert isinstance(validation_error("VALIDATION_FAILED", protocol="p"), APGIError)
    assert isinstance(simulation_error("CONVERGENCE_FAILED", details="d"), APGIError)
    assert isinstance(data_error("DATA_CORRUPTION", details="d"), APGIError)
    assert isinstance(io_error("FILE_NOT_FOUND", file_path="f"), APGIError)
    assert isinstance(import_error("MODULE_NOT_FOUND", module="m"), APGIError)
    assert isinstance(critical_error(ErrorCode.GEN_UNKNOWN), APGIError)


def test_handle_import_error():
    # Test just executes to ensure no crash, relies on logger side effects
    handle_import_error("pandas", ImportError("No module named 'pandas'"))
    handle_import_error("pkg", ImportError("DLL load failed"))
    handle_import_error("pkg", ImportError("Permission denied"))
    handle_import_error("pkg", Exception("Other"))


def test_format_user_message():
    err = APGIError("my message", suggestion="do this", context={"a": 1})
    msg = format_user_message(err)
    assert "my message" in msg
    assert "do this" in msg
    assert "{'a': 1}" in msg


def test_get_recovery_suggestions():
    assert len(get_recovery_suggestions("FILE_NOT_FOUND")) > 0
    assert len(get_recovery_suggestions("UNKNOWN_CODE")) > 0


def test_get_error_summary():
    # Just ensure it returns dict without crashing
    s = get_error_summary()
    assert "total_errors" in s


def test_handle_error_function():
    err = handle_error(ValueError("test err"))
    assert isinstance(err, APGIError)


def test_safe_execute():
    def good():
        return 5

    def bad():
        raise ValueError("bad")

    assert safe_execute(good) == 5
    assert safe_execute(bad, default_return=10) == 10

    with pytest.raises(ValueError):
        safe_execute(bad, error_type=ValueError)
    with pytest.raises(APGIError):
        safe_execute(bad)


def test_error_boundary():
    @error_boundary(default_return=42)
    def f():
        raise Exception("x")

    assert f() == 42
