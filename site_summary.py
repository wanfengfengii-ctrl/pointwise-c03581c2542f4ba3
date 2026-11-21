"""Fixed-size site summaries: exact support plus relative observation weights.

A site summary is a deterministic UTF-8 JSON packet.  It contains no original
observation labels, no per-observation value list and no observation-sized
arrays.  Everything is expressed on the fixed variable scale of the case:

    {"version": 1, "case_id": ..., "site_id": ..., "variables": [...],
     "body": {"feasible": bool,
              "constraints": [[rational, ...], ...],   # canonical RREF, <= n
              "values": [rational, ...],
              "precision": [[rational, ...], ...],     # n x n, PSD
              "information": [rational, ...]}}         # length n

Given observations Y = H x + noise with noise covariance R, the site likelihood
is non-zero only where y - H x lies in the range of R.  A basis N of the null
space of R turns that support into N^T H x = N^T y.  On the support the
log-likelihood differs from -x'Px/2 + q'x by a constant, with
P = H^T R^+ H and q = H^T R^+ y.  Marginal likelihood constants are dropped:
only the exact feasible set and the relative weight inside it are stored.

`feasible` is False exactly when the site's own measurements admit no variable
value at all.  The packet is still produced normally; the marker is different
from a malformed packet, which raises ValueError.
"""
import json
from dataclasses import dataclass
from fractions import Fraction

from database import dumps
from exact import (affine_solve, canonical_constraints, matrix, matvec,
                   multiply, psd, pseudo_inverse, rational, transpose, zeros)
from specs import linear_rows, names

VERSION = 1
_PACKET_FIELDS = {"version", "case_id", "site_id", "variables", "body"}
_BODY_FIELDS = {"feasible", "constraints", "values", "precision", "information"}


def _identifier(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(label + " must be a nonempty string")
    return value


def _rationals(items, size, label):
    if not isinstance(items, list) or len(items) != size:
        raise ValueError(label + " shape")
    return [rational(item) for item in items]


def _square(rows, size, label):
    if not isinstance(rows, list) or len(rows) != size:
        raise ValueError(label + " shape")
    return [_rationals(row, size, label) for row in rows]


def _empty_body(n):
    zero = str(Fraction(0))
    return {"feasible": False, "constraints": [], "values": [],
            "precision": [[zero] * n for _ in range(n)],
            "information": [zero] * n}


def _body(variables, h, y, noise):
    n, m = len(variables), len(y)
    kernel = affine_solve(noise, [Fraction(0)] * m, m)[1] if m else []
    rank = len(kernel[0]) if kernel else 0
    if rank:
        null_rows = transpose(kernel, rank)
        constraints = [[sum(null_rows[i][t] * h[t][j] for t in range(m))
                        for j in range(n)] for i in range(rank)]
        values = [sum(null_rows[i][t] * y[t] for t in range(m)) for i in range(rank)]
    else:
        constraints, values = [], []
    canonical = canonical_constraints(constraints, values, n)
    if canonical is None:
        return _empty_body(n)
    constraints, values = canonical
    if m:
        pseudoinverse = pseudo_inverse(noise)
        loader = multiply(transpose(h, n), pseudoinverse, m)
        precision = multiply(loader, h, n)
        information = matvec(loader, y)
    else:
        precision = zeros(n, n)
        information = [Fraction(0)] * n
    return {"feasible": True,
            "constraints": [[str(x) for x in row] for row in constraints],
            "values": [str(x) for x in values],
            "precision": [[str(x) for x in row] for row in precision],
            "information": [str(x) for x in information]}


def compile_site(case_id, site_id, variables, observations, values, noise):
    """Compile one site's fixed-size summary without any center prior."""
    case_id = _identifier(case_id, "case_id")
    site_id = _identifier(site_id, "site_id")
    variables = names(variables, 6)
    labels, h = linear_rows(observations, variables, 8)
    if not isinstance(values, dict) or set(values) != set(labels):
        raise ValueError("observation values must match labels")
    y = [rational(values[label]) for label in labels]
    try:
        covariance = matrix(noise, len(labels), len(labels))
    except TypeError as exc:
        raise ValueError("noise shape") from exc
    psd(covariance)
    packet = {"version": VERSION, "case_id": case_id, "site_id": site_id,
              "variables": list(variables), "body": _body(variables, h, y, covariance)}
    return dumps(packet).encode("utf-8")


@dataclass(frozen=True)
class Packet:
    case_id: str
    site_id: str
    variables: tuple
    feasible: bool
    constraints: tuple
    values: tuple
    precision: tuple
    information: tuple


def _pairs(items):
    result = {}
    for key, value in items:
        if not isinstance(key, str):
            raise ValueError("json keys must be strings")
        if key in result:
            raise ValueError("duplicate json key")
        result[key] = value
    return result


def _constant(value):
    raise ValueError("invalid json constant: " + value)


def _check_canonical(rows, n):
    previous = -1
    for index, row in enumerate(rows):
        pivot = next((j for j in range(n) if row[j]), None)
        if pivot is None or pivot <= previous or row[pivot] != 1:
            raise ValueError("constraints must be canonical")
        for other_index, other in enumerate(rows):
            if other_index != index and other[pivot]:
                raise ValueError("constraints must be canonical")
        previous = pivot


def decode_packet(raw):
    """Parse and fully validate a packet independent of any external original."""
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise ValueError("packet bytes required")
    try:
        text = bytes(raw).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("packet must be UTF-8") from exc
    try:
        document = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid packet json") from exc
    if not isinstance(document, dict) or set(document) != _PACKET_FIELDS:
        raise ValueError("packet fields")
    version = document["version"]
    if type(version) is not int or version != VERSION:
        raise ValueError("unsupported packet version")
    case_id = _identifier(document["case_id"], "case_id")
    site_id = _identifier(document["site_id"], "site_id")
    if not isinstance(document["variables"], list):
        raise ValueError("variables must be a list")
    variables = names(document["variables"], 6)
    n = len(variables)
    body = document["body"]
    if not isinstance(body, dict) or set(body) != _BODY_FIELDS:
        raise ValueError("body fields")
    feasible = body["feasible"]
    if not isinstance(feasible, bool):
        raise ValueError("feasible flag")
    raw_rows = body["constraints"]
    if not isinstance(raw_rows, list) or len(raw_rows) > n:
        raise ValueError("constraints shape")
    constraints = tuple(tuple(_rationals(row, n, "constraints")) for row in raw_rows)
    values = tuple(_rationals(body["values"], len(constraints), "values"))
    precision = tuple(tuple(row) for row in _square(body["precision"], n, "precision"))
    information = tuple(_rationals(body["information"], n, "information"))
    psd([list(row) for row in precision])
    if feasible:
        _check_canonical(constraints, n)
    elif (constraints or values or any(any(row) for row in precision)
          or any(information)):
        raise ValueError("no-solution packet must carry no support or weight")
    return Packet(case_id, site_id, variables, feasible, constraints,
                  values, precision, information)
