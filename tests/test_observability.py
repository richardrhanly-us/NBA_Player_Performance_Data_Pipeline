import json

from src.services import observability


def test_get_logger_returns_same_logger_for_same_service():
    a = observability.get_logger("billing")
    b = observability.get_logger("billing")
    assert a is b


def test_get_logger_does_not_attach_duplicate_handlers():
    logger = observability.get_logger("dedup-test")
    handler_count = len(logger.handlers)
    observability.get_logger("dedup-test")
    assert len(logger.handlers) == handler_count


def test_log_event_emits_one_json_line(caplog):
    logger = observability.get_logger("test-service")
    with caplog.at_level("INFO", logger=logger.name):
        observability.log_event(logger, "test.event", status="ok", count=3)

    assert len(caplog.records) == 1
    payload = json.loads(caplog.records[0].message)
    assert payload["event"] == "test.event"
    assert payload["severity"] == "info"
    assert payload["status"] == "ok"
    assert payload["count"] == 3
    assert "timestamp" in payload


def test_log_event_redacts_sensitive_field_names(caplog):
    logger = observability.get_logger("test-service-2")
    with caplog.at_level("INFO", logger=logger.name):
        observability.log_event(
            logger,
            "test.event",
            access_token="tok_should_not_appear",
            refresh_token="refresh_should_not_appear",
            stripe_secret_key="sk_should_not_appear",
            webhook_secret="whsec_should_not_appear",
            password="hunter2",
            authorization_header="Bearer abc",
            safe_field="visible",
        )

    payload = json.loads(caplog.records[0].message)
    assert payload["access_token"] == "[REDACTED]"
    assert payload["refresh_token"] == "[REDACTED]"
    assert payload["stripe_secret_key"] == "[REDACTED]"
    assert payload["webhook_secret"] == "[REDACTED]"
    assert payload["password"] == "[REDACTED]"
    assert payload["authorization_header"] == "[REDACTED]"
    assert payload["safe_field"] == "visible"

    raw_line = caplog.records[0].message
    assert "tok_should_not_appear" not in raw_line
    assert "refresh_should_not_appear" not in raw_line
    assert "sk_should_not_appear" not in raw_line
    assert "whsec_should_not_appear" not in raw_line
    assert "hunter2" not in raw_line


def test_log_event_severity_maps_to_log_level(caplog):
    logger = observability.get_logger("test-service-3")
    with caplog.at_level("WARNING", logger=logger.name):
        observability.log_event(logger, "test.warn", severity="warning")
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
