"""
MongoDB helper functions for persisting dashboard alerts and annotations.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

DEFAULT_DB = "inventory_dashboard"
DEFAULT_COLLECTION = "alerts"


def _build_client() -> Optional[MongoClient]:
    uri = os.getenv("MONGO_URI")
    if not uri:
        return None
    try:
        return MongoClient(uri, serverSelectionTimeoutMS=5_000)
    except PyMongoError:
        return None


def _get_collection() -> Optional[Collection]:
    client = _build_client()
    if client is None:
        return None
    db_name = os.getenv("MONGO_DB", DEFAULT_DB)
    coll_name = os.getenv("MONGO_COLLECTION", DEFAULT_COLLECTION)
    return client[db_name][coll_name]


def save_alert(alert: Dict[str, Any]) -> bool:
    """
    Persist an alert document. Returns True on success, False otherwise.
    """
    collection = _get_collection()
    if collection is None:
        return False

    payload = alert.copy()
    payload.setdefault("created_at", datetime.now(timezone.utc))
    try:
        collection.insert_one(payload)
        return True
    except PyMongoError:
        return False


def fetch_recent_alerts(limit: int = 20) -> List[Dict[str, Any]]:
    """
    Retrieve the most recent alerts for display in the dashboard.
    """
    collection = _get_collection()
    if collection is None:
        return []
    try:
        cursor = (
            collection.find().sort("created_at", -1).limit(max(limit, 1))
        )
        return list(cursor)
    except PyMongoError:
        return []


