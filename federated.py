"""Federated center: an independent archive for site summaries.

The center never sees observations.  It stores one common prior per case,
receives fixed-size site packets, and publishes exact rational moments of the
joint conditional distribution.  All arithmetic is exact; a contradictory
support is reported as `inconsistent` without jitter or tolerance.
"""
import json
from fractions import Fraction

from archive import identifier
from database import atomic, dumps
from exact import (affine_solve, canonical_constraints, inverse, matvec,
                   multiply, pseudo_inverse, rational, transpose, zeros)
from gauss2 import Affine, Potential, decode as decode_model
from specs import Result, linear_rows
from site_summary import decode_packet

_FORMAT = 1


def install(db):
    with atomic(db):
        db.execute("""CREATE TABLE IF NOT EXISTS fed_cases (
            case_id TEXT PRIMARY KEY, payload TEXT NOT NULL)""")
        db.execute("""CREATE TABLE IF NOT EXISTS fed_sites (
            case_id TEXT NOT NULL REFERENCES fed_cases(case_id),
            site_id TEXT NOT NULL, packet BLOB NOT NULL,
            PRIMARY KEY (case_id, site_id))""")
        db.execute("""CREATE TABLE IF NOT EXISTS fed_results (
            case_id TEXT PRIMARY KEY REFERENCES fed_cases(case_id),
            payload TEXT NOT NULL)""")


def _site_names(values):
    values = tuple(values)
    if not 2 <= len(values) <= 3:
        raise ValueError("two or three sites required")
    if any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values):
        raise ValueError("site identifiers must be unique nonempty strings")
    return values


def _rows_payload(rows):
    return {label: {v: str(rational(x)) for v, x in terms.items()}
            for label, terms in rows.items()}


def _prior_system(prior):
    """Return (precision, information, constraints, values) for a prior model."""
    n = len(prior.variables)
    if isinstance(prior, Affine):
        mean, covariance = prior._moments()
        precision = pseudo_inverse(covariance)
        information = matvec(precision, mean)
        null_basis = affine_solve(covariance, [0] * n, n)[1]
        rank = len(null_basis[0]) if null_basis else 0
        constraints = [list(row) for row in transpose(null_basis, rank)] if rank else []
        values = [sum(row[i] * mean[i] for i in range(n)) for row in constraints]
    elif isinstance(prior, Potential):
        precision = [list(row) for row in prior.precision]
        information = list(prior.information)
        constraints = [list(row) for row in prior.constraints]
        values = list(prior.values)
    else:
        raise ValueError("prior must be Affine or Potential")
    canonical = canonical_constraints(constraints, values, n)
    if canonical is None:  # Constructors already reject this; kept as a guard.
        raise ValueError("inconsistent prior")
    return precision, information, canonical[0], canonical[1]


def _solve(precision, information, constraints, values, n):
    """Return (mean, covariance) of the quadratic on the affine support, or None."""
    canonical = canonical_constraints(constraints, values, n)
    if canonical is None:
        return None
    constraints, values = canonical
    origin, basis = affine_solve(constraints, values, n)
    r = len(basis[0]) if basis else 0
    if r == 0:
        return origin, zeros(n, n)
    bt = transpose(basis, r)
    restricted = multiply(multiply(bt, precision), basis, r)
    inverse_restricted = inverse(restricted)  # Prior properness makes this valid.
    covariance = multiply(multiply(basis, inverse_restricted, r), bt, n)
    adjusted = [info - q for info, q in zip(information, matvec(precision, origin))]
    shift = matvec(covariance, adjusted)
    mean = [x + y for x, y in zip(origin, shift)]
    return mean, covariance


def _combine(prior_payload, variable_order, labels, output_rows, packets):
    prior = decode_model(prior_payload)
    if tuple(prior.variables) != tuple(variable_order):
        raise ValueError("prior variables changed")
    n = len(variable_order)
    precision, information, constraints, values = _prior_system(prior)
    precision = [row[:] for row in precision]
    information = list(information)
    constraints = [row[:] for row in constraints]
    values = list(values)
    for packet in packets:
        if not packet.feasible:
            return Result("inconsistent", labels)
        constraints.extend([list(row) for row in packet.constraints])
        values.extend(packet.values)
        precision = [[precision[i][j] + packet.precision[i][j] for j in range(n)]
                     for i in range(n)]
        information = [a + b for a, b in zip(information, packet.information)]
    solved = _solve(precision, information, constraints, values, n)
    if solved is None:
        return Result("inconsistent", labels)
    mean, covariance = solved
    out_mean = tuple(sum((row[i] * mean[i] for i in range(n)), Fraction(0))
                     for row in output_rows)
    out_cov = tuple(tuple(
        sum((a[i] * covariance[i][j] * b[j]
             for i in range(n) for j in range(n)), Fraction(0))
        for b in output_rows) for a in output_rows)
    return Result("ok", labels, out_mean, out_cov)


def _result_payload(result):
    return dumps({"status": result.status, "outputs": list(result.outputs),
                  "mean": [str(x) for x in result.mean],
                  "covariance": [[str(x) for x in row] for row in result.covariance]})


def _load_result(payload):
    document = json.loads(payload)
    mean = tuple(Fraction(x) for x in document["mean"])
    covariance = tuple(tuple(Fraction(x) for x in row) for row in document["covariance"])
    return Result(document["status"], tuple(document["outputs"]), mean, covariance)


class FederatedArchive:
    """Caller owns the connection; repeated initialization keeps prior cases."""

    def __init__(self, db):
        self.db = db
        install(db)

    def _case(self, case_id):
        row = self.db.execute("SELECT payload FROM fed_cases WHERE case_id=?",
                              (case_id,)).fetchone()
        if row is None:
            raise KeyError(case_id)
        return json.loads(row[0])

    def open_case(self, case_id, prior, outputs, sites):
        case_id = identifier(case_id)
        if not isinstance(prior, (Affine, Potential)):
            raise ValueError("prior must be Affine or Potential")
        normalized = decode_model(prior.payload())  # private copy; caller mutation is inert
        variables = tuple(normalized.variables)
        if not isinstance(outputs, dict) or not outputs:
            raise ValueError("nonempty ordered outputs required")
        labels, _ = linear_rows(outputs, variables, 8)
        sites = _site_names(sites)
        payload = dumps({"format": _FORMAT, "prior": normalized.payload(),
                         "outputs": _rows_payload(outputs),
                         "variables": list(variables), "sites": list(sites)})
        with atomic(self.db):
            row = self.db.execute("SELECT payload FROM fed_cases WHERE case_id=?",
                                  (case_id,)).fetchone()
            if row is not None:
                if row[0] != payload:
                    raise ValueError("case already opened with different content")
                return
            self.db.execute("INSERT INTO fed_cases VALUES (?,?)", (case_id, payload))

    def add_site(self, case_id, packet):
        case_id = identifier(case_id)
        with atomic(self.db):
            case = self._case(case_id)
            parsed = decode_packet(packet)  # validates structure and mathematics first
            if parsed.case_id != case_id:
                raise ValueError("packet case mismatch")
            if parsed.site_id not in case["sites"]:
                raise ValueError("undeclared site")
            if tuple(parsed.variables) != tuple(case["variables"]):
                raise ValueError("packet variable order mismatch")
            raw = bytes(packet)
            row = self.db.execute("SELECT packet FROM fed_sites WHERE case_id=? AND site_id=?",
                                  (case_id, parsed.site_id)).fetchone()
            if row is not None:
                if bytes(row[0]) != raw:
                    raise ValueError("different packet already received for this site")
                return bytes(row[0])
            self.db.execute("INSERT INTO fed_sites VALUES (?,?,?)",
                            (case_id, parsed.site_id, raw))
            return raw

    def finish(self, case_id):
        case_id = identifier(case_id)
        with atomic(self.db):
            case = self._case(case_id)
            row = self.db.execute("SELECT payload FROM fed_results WHERE case_id=?",
                                  (case_id,)).fetchone()
            if row is not None:
                return _load_result(row[0])
            received = {}
            for site_id, packet in self.db.execute(
                    "SELECT site_id,packet FROM fed_sites WHERE case_id=?", (case_id,)):
                received[site_id] = decode_packet(packet)
            if any(site not in received for site in case["sites"]):
                raise ValueError("missing site packets")
            labels, output_rows = linear_rows(case["outputs"], tuple(case["variables"]), 8)
            labels = tuple(labels)
            packets = [received[site] for site in case["sites"]]
            result = _combine(case["prior"], case["variables"], labels,
                              output_rows, packets)
            self.db.execute("INSERT INTO fed_results VALUES (?,?)",
                            (case_id, _result_payload(result)))
            return result
