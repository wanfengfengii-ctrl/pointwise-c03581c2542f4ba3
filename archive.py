"""Normal SDK1 application archive. A connection is explicitly owned by the caller."""
import json
from database import atomic, dumps
from gauss1 import Model
from specs import Request


def identifier(value):
    if not isinstance(value, str) or not value:
        raise ValueError("nonempty ID required")
    return value


def create_legacy(db):
    with atomic(db):
        db.execute("CREATE TABLE IF NOT EXISTS archive_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        db.execute("INSERT OR IGNORE INTO archive_meta VALUES ('format','1')")
        version = db.execute("SELECT value FROM archive_meta WHERE key='format'").fetchone()[0]
        if version != "1":
            raise ValueError("unsupported archive format")
        db.execute("CREATE TABLE IF NOT EXISTS models_v1 (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        db.execute("""CREATE TABLE IF NOT EXISTS analyses_v1 (
            id TEXT PRIMARY KEY, model_id TEXT NOT NULL REFERENCES models_v1(id),
            payload TEXT NOT NULL)""")


class Archive:
    def __init__(self, db):
        self.db = db
        create_legacy(db)

    def save_model(self, model_id, variables, mean, covariance):
        model_id = identifier(model_id)
        model = Model(variables, mean, covariance)
        with atomic(self.db):
            previous = self.db.execute("SELECT payload FROM models_v1 WHERE id=?", (model_id,)).fetchone()
            if previous and set(Model.from_payload(json.loads(previous[0])).variables) != set(model.variables):
                raise ValueError("model variable identity is immutable")
            self.db.execute("""INSERT INTO models_v1 VALUES (?,?)
                ON CONFLICT(id) DO UPDATE SET payload=excluded.payload""",
                            (model_id, dumps(model.payload())))

    def load_model(self, model_id):
        row = self.db.execute("SELECT payload FROM models_v1 WHERE id=?",
                              (identifier(model_id),)).fetchone()
        if row is None:
            raise KeyError(model_id)
        return Model.from_payload(json.loads(row[0]))

    def save_analysis(self, analysis_id, model_id, observations, values, noise, outputs):
        from analysis import Analysis
        analysis_id, model_id = identifier(analysis_id), identifier(model_id)
        request = Request(observations, values, noise, outputs)
        with atomic(self.db):
            request.compile(self.load_model(model_id).variables)
            self.db.execute("""INSERT INTO analyses_v1 VALUES (?,?,?)
                ON CONFLICT(id) DO UPDATE SET model_id=excluded.model_id,payload=excluded.payload""",
                            (analysis_id, model_id, dumps(request.payload())))
        return Analysis(self, analysis_id)

    def load_analysis(self, analysis_id):
        row = self.db.execute("SELECT model_id,payload FROM analyses_v1 WHERE id=?",
                              (identifier(analysis_id),)).fetchone()
        if row is None:
            raise KeyError(analysis_id)
        return row[0], Request.from_payload(json.loads(row[1]))

    def evaluate(self, analysis_id):
        from analysis import Analysis
        return Analysis(self, analysis_id).evaluate()
