"""SDK2 native constrained Gaussian models and independent target storage.

No SDK1 import, covariance-to-model constructor, migration or legacy binding.
"""
import json
from exact import (add, affine_solve, inverse, matrix, matvec, multiply,
                   positive_definite, psd, transpose, vector)
from specs import Request, names
from conditioning import condition
from database import atomic, dumps


class Affine:
    """X = offset + loadings Z; independent Z_j ~ N(0, variances_j).

    Loadings must have independent columns; variances must be strictly positive.
    Zero latent dimensions represents a deterministic point.
    """
    kind = "affine"

    def __init__(self, variables, offset, loadings, variances):
        self.variables = names(variables)
        n = len(self.variables)
        self.offset = tuple(vector(offset, n))
        self.variances = tuple(vector(variances, len(variances)))
        k = len(self.variances)
        if k > n or any(v <= 0 for v in self.variances):
            raise ValueError("latent variances")
        self.loadings = tuple(tuple(row) for row in matrix(loadings, n, k))
        kernel = affine_solve(self.loadings, [0] * n, k)[1]
        if k and len(kernel[0]):
            raise ValueError("dependent latent columns")

    def _moments(self):
        k = len(self.variances)
        weighted = [[v * self.variances[j] for j, v in enumerate(row)]
                    for row in self.loadings]
        covariance = multiply(weighted, transpose(self.loadings, k), len(self.variables))
        return self.offset, covariance

    def evaluate(self, request):
        return condition(*self._moments(), request.compile(self.variables))

    def payload(self):
        return {"kind": self.kind, "variables": list(self.variables),
                "offset": [str(x) for x in self.offset],
                "loadings": [[str(x) for x in row] for row in self.loadings],
                "variances": [str(x) for x in self.variances]}


class Potential:
    """Density on {x: constraints x = values}, proportional to
    exp(-x' precision x / 2 + information' x).

    Precision is PSD and strictly positive on the constraint kernel.
    Redundant consistent constraints are allowed; improper distributions are not.
    """
    kind = "potential"

    def __init__(self, variables, precision, information, constraints, values):
        self.variables = names(variables)
        n = len(self.variables)
        self.precision = tuple(tuple(row) for row in matrix(precision, n, n))
        self.information = tuple(vector(information, n))
        if len(values) > 8:
            raise ValueError("too many constraints")
        self.constraints = tuple(tuple(row) for row in matrix(constraints, len(values), n))
        self.values = tuple(vector(values, len(values)))
        psd(self.precision)
        self._moments()  # Validate consistency and properness at construction.

    def _moments(self):
        n = len(self.variables)
        solved = affine_solve(self.constraints, self.values, n)
        if solved is None:
            raise ValueError("inconsistent constraints")
        origin, basis = solved
        r = len(basis[0])
        bt = transpose(basis, r)
        restricted = multiply(multiply(bt, self.precision), basis, r)
        positive_definite(restricted)
        covariance = multiply(multiply(basis, inverse(restricted), r), bt, n)
        adjusted = [h - q for h, q in zip(self.information, matvec(self.precision, origin))]
        shift = matvec(covariance, adjusted)
        return [x + y for x, y in zip(origin, shift)], covariance

    def evaluate(self, request):
        return condition(*self._moments(), request.compile(self.variables))

    def payload(self):
        return {"kind": self.kind, "variables": list(self.variables),
                "precision": [[str(x) for x in row] for row in self.precision],
                "information": [str(x) for x in self.information],
                "constraints": [[str(x) for x in row] for row in self.constraints],
                "values": [str(x) for x in self.values]}


def decode(payload):
    fields = dict(payload)
    kind = fields.pop("kind", None)
    if kind == "affine":
        if set(fields) != {"variables", "offset", "loadings", "variances"}:
            raise ValueError("affine fields")
        return Affine(**fields)
    if kind == "potential":
        if set(fields) != {"variables", "precision", "information", "constraints", "values"}:
            raise ValueError("potential fields")
        return Potential(**fields)
    raise ValueError("unknown SDK2 representation")


def identifier(value):
    if not isinstance(value, str) or not value:
        raise ValueError("nonempty ID required")
    return value


def install(db):
    with atomic(db):
        db.execute("CREATE TABLE IF NOT EXISTS g2_models (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        db.execute("""CREATE TABLE IF NOT EXISTS g2_analyses (
            id TEXT PRIMARY KEY, model_id TEXT NOT NULL REFERENCES g2_models(id),
            payload TEXT NOT NULL)""")


class NativeArchive:
    """Caller owns the connection. All reads resolve current model ID, without cache."""

    def __init__(self, db):
        self.db = db
        install(db)

    def put_model(self, model_id, model):
        model_id = identifier(model_id)
        if not isinstance(model, (Affine, Potential)):
            raise ValueError("SDK2 model required")
        # Decode a private normalized copy, so later object mutation cannot alter storage.
        model = decode(model.payload())
        with atomic(self.db):
            previous = self.db.execute("SELECT payload FROM g2_models WHERE id=?", (model_id,)).fetchone()
            if previous and set(decode(json.loads(previous[0])).variables) != set(model.variables):
                raise ValueError("model variable identity is immutable")
            self.db.execute("""INSERT INTO g2_models VALUES (?,?)
                ON CONFLICT(id) DO UPDATE SET payload=excluded.payload""",
                            (model_id, dumps(model.payload())))

    def get_model(self, model_id):
        row = self.db.execute("SELECT payload FROM g2_models WHERE id=?", (identifier(model_id),)).fetchone()
        if row is None:
            raise KeyError(model_id)
        return decode(json.loads(row[0]))

    def put_analysis(self, analysis_id, model_id, request):
        analysis_id, model_id = identifier(analysis_id), identifier(model_id)
        with atomic(self.db):
            request.compile(self.get_model(model_id).variables)
            payload = dumps(request.payload())
            self.db.execute("""INSERT INTO g2_analyses VALUES (?,?,?)
                ON CONFLICT(id) DO UPDATE SET model_id=excluded.model_id,payload=excluded.payload""",
                            (analysis_id, model_id, payload))

    def get_analysis(self, analysis_id):
        row = self.db.execute("SELECT model_id,payload FROM g2_analyses WHERE id=?",
                              (identifier(analysis_id),)).fetchone()
        if row is None:
            raise KeyError(analysis_id)
        return row[0], Request.from_payload(json.loads(row[1]))

    def evaluate(self, analysis_id):
        # One SQLite snapshot covers the binding and model.
        with atomic(self.db):
            model_id, request = self.get_analysis(analysis_id)
            return self.get_model(model_id).evaluate(request)
