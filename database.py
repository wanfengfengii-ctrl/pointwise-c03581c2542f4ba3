"""Shared SQLite ownership and atomic transaction facility."""
from contextlib import contextmanager
from itertools import count
import json
import sqlite3

_serial = count()


def connect(path):
    db = sqlite3.connect(path, isolation_level=None)
    db.execute("PRAGMA foreign_keys=ON")
    return db


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


@contextmanager
def atomic(db, before_commit=None):
    nested = db.in_transaction
    savepoint = "atomic_" + str(next(_serial))
    db.execute("SAVEPOINT " + savepoint if nested else "BEGIN IMMEDIATE")
    try:
        yield
        if before_commit is not None:
            before_commit()
        db.execute("RELEASE " + savepoint if nested else "COMMIT")
    except BaseException:
        if db.in_transaction:
            if nested:
                db.execute("ROLLBACK TO " + savepoint)
                db.execute("RELEASE " + savepoint)
            else:
                db.execute("ROLLBACK")
        raise
