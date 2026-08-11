from datetime import datetime, timezone
from src.search.temporal import TemporalParser

# Injected fixed reference date: Wednesday 2026-08-12 10:00:00 UTC
REF = datetime(2026, 8, 12, 10, 0, 0, tzinfo=timezone.utc)


def test_today():
    parser = TemporalParser()
    tr = parser.parse("today", reference_date=REF)
    assert tr.resolved is True
    assert tr.precision == "exact_day"
    assert tr.start_ts == "2026-08-12T00:00:00+00:00"
    assert tr.end_ts == "2026-08-12T23:59:59.999999+00:00"


def test_yesterday():
    parser = TemporalParser()
    tr = parser.parse("yesterday", reference_date=REF)
    assert tr.resolved is True
    assert tr.precision == "exact_day"
    assert tr.start_ts == "2026-08-11T00:00:00+00:00"


def test_last_tuesday():
    # Today is Wednesday Aug 12 2026. Last Tuesday was Aug 11 2026.
    parser = TemporalParser()
    tr = parser.parse("last Tuesday", reference_date=REF)
    assert tr.resolved is True
    assert tr.precision == "exact_day"
    assert tr.start_ts == "2026-08-11T00:00:00+00:00"


def test_last_tuesday_when_today_is_tuesday():
    # If today is Tuesday Aug 11 2026, "last Tuesday" must resolve to Aug 4 2026.
    ref_tues = datetime(2026, 8, 11, 10, 0, 0, tzinfo=timezone.utc)
    parser = TemporalParser()
    tr = parser.parse("last Tuesday", reference_date=ref_tues)
    assert tr.resolved is True
    assert tr.start_ts == "2026-08-04T00:00:00+00:00"


def test_last_week():
    # Aug 12 2026 is Wed of week Aug 10 - Aug 16.
    # Last week was Mon Aug 3 - Sun Aug 9.
    parser = TemporalParser()
    tr = parser.parse("last week", reference_date=REF)
    assert tr.resolved is True
    assert tr.precision == "week"
    assert tr.start_ts == "2026-08-03T00:00:00+00:00"
    assert tr.end_ts == "2026-08-09T23:59:59.999999+00:00"


def test_around_may():
    parser = TemporalParser()
    tr = parser.parse("around May", reference_date=REF)
    assert tr.resolved is True
    assert tr.precision == "approximate"
    assert "2026-04" in tr.start_ts
    assert "2026-06" in tr.end_ts


def test_exact_date():
    parser = TemporalParser()
    tr = parser.parse("2026-05-14", reference_date=REF)
    assert tr.resolved is True
    assert tr.precision == "exact_day"
    assert tr.start_ts == "2026-05-14T00:00:00+00:00"


def test_between_hours_on_date():
    parser = TemporalParser()
    tr = parser.parse("between 10pm and 11pm on 2026-08-05", reference_date=REF)
    assert tr.resolved is True
    assert tr.start_ts == "2026-08-05T22:00:00+00:00"
    assert tr.end_ts == "2026-08-05T23:00:00+00:00"
