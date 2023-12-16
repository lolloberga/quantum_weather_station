from typing import List
from sqlalchemy import inspect
from datetime import timezone, datetime
from database.model.measure import Measure


def object_as_dict(obj, cls=Measure):
    return {c.key: getattr(obj, c.key) for c in Measure.__table__.columns}


def object_status(obj):
    insp = inspect(obj)
    status = {
        "transient": insp.transient,
        "pending": insp.pending,
        "persistent": insp.persistent,
        "deleted": insp.deleted,
        "detached": insp.detached
    }
    return status


def unix_to_datetime(timestamp: int):
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def datetime_to_unix(dt: datetime):
    return datetime.timestamp(dt)


def unix_to_datetime_list(timestamps: List):
    return [datetime.fromtimestamp(timestamp, tz=timezone.utc) for timestamp in timestamps]
