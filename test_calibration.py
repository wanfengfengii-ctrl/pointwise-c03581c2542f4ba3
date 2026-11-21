"""Deterministic original regressions; disk tests use only unittest temporary paths."""
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import random
import tempfile
import unittest

from archive import Archive
from analysis import Analysis
from database import atomic, connect
from gauss1 import Model
from gauss2 import Affine, Potential, NativeArchive, decode
from samples import (affine_equal, build_legacy, equality_request, potential_equal)
from specs import Request


# Independent oracle: select a nonsingular principal block of observation covariance.
# It imports no production algebra or SDK to construct expected results.
def oracle_inverse(a):
    n = len(a)
    work = [[F(x) for x in row] + [F(i == j) for j in range(n)]
            for i, row in enumerate(a)]
    for j in range(n):
        p = next((i for i in range(j, n) if work[i][j]), None)
        if p is None:
            raise ValueError("singular")
        work[j], work[p] = work[p], work[j]
        d = work[j][j]
        work[j] = [x / d for x in work[j]]
        for i in range(n):
            if i != j:
                d = work[i][j]
                work[i] = [x - d * y for x, y in zip(work[i], work[j])]
    return [row[n:] for row in work]


def oracle(mean, covariance, h, y, noise, outputs):
    n, m = len(mean), len(y)
    cross = [[sum((F(covariance[i][j]) * h[k][j] for j in range(n)), F(0))
              for k in range(m)] for i in range(n)]
    s = [[sum((h[i][j] * cross[j][k] for j in range(n)), F(noise[i][k]))
          for k in range(m)] for i in range(m)]
    residual = [F(y[i]) - sum((h[i][j] * mean[j] for j in range(n)), F(0))
                for i in range(m)]
    chosen, inv = (), []
    for size in range(m, 0, -1):
        found = False
        for ix in combinations(range(m), size):
            try:
                inv = oracle_inverse([[s[i][j] for j in ix] for i in ix])
                chosen, found = ix, True
                break
            except ValueError:
                continue
        if found:
            break
    coefficients = [sum((inv[i][j] * residual[k] for j, k in enumerate(chosen)), F(0))
                    for i in range(len(chosen))]
    if any(sum((s[i][k] * coefficients[j] for j, k in enumerate(chosen)), F(0))
           != residual[i] for i in range(m)):
        return "inconsistent", (), ()
    gain = [[sum((cross[i][k] * inv[t][j] for t, k in enumerate(chosen)), F(0))
             for j in range(len(chosen))] for i in range(n)]
    mu = [F(mean[i]) + sum((cross[i][k] * coefficients[j]
                          for j, k in enumerate(chosen)), F(0)) for i in range(n)]
    cov = [[F(covariance[i][j]) - sum((gain[i][t] * cross[j][k]
                                     for t, k in enumerate(chosen)), F(0))
            for j in range(n)] for i in range(n)]
    out_mean = tuple(sum((F(a) * b for a, b in zip(row, mu)), F(0)) for row in outputs)
    out_cov = tuple(tuple(sum((F(a[i]) * cov[i][j] * F(b[j])
                              for i in range(n) for j in range(n)), F(0))
                          for b in outputs) for a in outputs)
    return "ok", out_mean, out_cov


def plain(result):
    return result.status, result.mean, result.covariance


class SemanticsTests(unittest.TestCase):
    def setUp(self):
        self.db = connect(":memory:")
        self.archive = build_legacy(self.db)

    def tearDown(self):
        self.db.close()

    def test_hand_calculated_archive(self):
        a = self.archive
        self.assertEqual(plain(a.evaluate("agree")), ("ok", (2, 0), ((0, 0), (0, 0))))
        self.assertEqual(plain(a.evaluate("conflict")), ("inconsistent", (), ()))
        self.assertEqual(plain(a.evaluate("prior")), ("ok", (0, 0), ((4, 0), (0, 0))))
        self.assertEqual(plain(a.evaluate("noisy")),
                         ("ok", (F(8, 5), F(-2, 5)), ((F(3, 5), F(3, 5)), (F(3, 5), F(3, 5)))))
        self.assertEqual(plain(a.evaluate("offset_check")), ("ok", (6, -1), ((0, 0), (0, 0))))

    def test_redundant_and_zero_rows(self):
        model = Model(["X", "Y"], [0, 0], [[1, 1], [1, 1]])
        request = Request({"a": {"X": 1}, "again": {"X": 2}, "zero": {}},
                          {"a": 0, "again": 0, "zero": 0},
                          [[0] * 3 for _ in range(3)], {"y": {"Y": 1}})
        self.assertEqual(plain(model.evaluate(request)), ("ok", (0,), ((0,),)))
        request.values["zero"] = "1/100000000000000000000000000"
        self.assertEqual(model.evaluate(request).status, "inconsistent")

    def test_old_handle_tracks_model_and_request(self):
        a = self.archive
        old = Analysis(a, "agree")
        a.save_model("pair", ["Y", "X"], [0, 0], [[1, -1], [-1, 1]])
        self.assertEqual(old.evaluate().status, "inconsistent")
        a.save_analysis("agree", "pair", {"x": {"X": 1}, "y": {"Y": 1}},
                        {"x": 1, "y": -1}, [[0, 0], [0, 0]], {"gap": {"X": 1, "Y": -1}})
        self.assertEqual(old.evaluate().mean, (2,))

    def test_zero_rank_and_eight_variables(self):
        point = Model(["X"], [3], [[0]])
        for value, expected in [(3, "ok"), ("3000000000000000001/1000000000000000000", "inconsistent")]:
            req = Request({"x": {"X": 1}}, {"x": value}, [[0]], {"x": {"X": 1}})
            self.assertEqual(point.evaluate(req).status, expected)
        variables = ["v" + str(i) for i in range(8)]
        model = Model(variables, [0] * 8, [[1] * 8 for _ in range(8)])
        req = Request({}, {}, [], {"s": dict.fromkeys(variables, 1)})
        self.assertEqual(plain(model.evaluate(req)), ("ok", (0,), ((64,),)))

    def test_validation_is_atomic(self):
        before = self.db.serialize()
        for covariance in ([[1, 2], [2, 1]], [[1, 0], [1, 1]]):
            with self.assertRaises(ValueError):
                self.archive.save_model("pair", ["X", "Y"], [0, 0], covariance)
        with self.assertRaises(ValueError):
            self.archive.save_model("pair", ["X", "Z"], [0, 0], [[1, 0], [0, 1]])
        with self.assertRaises(ValueError):
            self.archive.save_analysis("agree", "pair", {"z": {"Z": 1}}, {"z": 0}, [[0]], {"x": {"X": 1}})
        with self.assertRaises(ValueError):
            self.archive.save_analysis("bad", "pair", {}, {}, [], {})
        with self.assertRaises(ValueError):
            Model(["X"], [0.1], [[1]])
        with self.assertRaises(ValueError):
            Model(["X"], [False], [[1]])
        self.assertEqual(self.db.serialize(), before)

    def test_rollback_and_outer_transaction(self):
        old = self.db.serialize()
        def fail():
            raise RuntimeError("before commit")
        with self.assertRaisesRegex(RuntimeError, "before commit"):
            with atomic(self.db, before_commit=fail):
                self.archive.save_model("pair", ["X", "Y"], [0, 0], [[1, -1], [-1, 1]])
                self.assertEqual(self.archive.evaluate("agree").status, "inconsistent")
        self.assertEqual(self.db.serialize(), old)
        with self.assertRaisesRegex(RuntimeError, "outer"):
            with atomic(self.db):
                self.archive.save_analysis("new", "pair", {}, {}, [], {"x": {"X": 1}})
                raise RuntimeError("outer")
        with self.assertRaises(KeyError):
            self.archive.evaluate("new")
        self.assertEqual(self.archive.evaluate("agree").status, "ok")

    def test_serialized_database_reopen(self):
        restored = connect(":memory:")
        try:
            restored.deserialize(self.db.serialize())
            reopened = Archive(restored)
            self.assertEqual(reopened.evaluate("noisy"), self.archive.evaluate("noisy"))
            self.assertEqual(restored.execute("SELECT count(*) FROM models_v1").fetchone()[0], 4)
            self.assertEqual(restored.execute("SELECT count(*) FROM analyses_v1").fetchone()[0], 5)
        finally:
            restored.close()

    def test_seeded_independent_oracle(self):
        rng = random.Random(71821)
        for case in range(70):
            n = rng.randint(1, 4)
            m = rng.randint(0, 4)
            rank = rng.randint(0, n)
            variables = ["v" + str(i) for i in range(n)]
            loadings = [[int(i == j) if i < rank else rng.randint(-2, 2)
                         for j in range(rank)] for i in range(n)]
            variances = [F(rng.randint(1, 3), rng.randint(1, 3)) for _ in range(rank)]
            mean = [F(rng.randint(-3, 3), 2) for _ in range(n)]
            cov = [[sum((loadings[i][k] * variances[k] * loadings[j][k]
                         for k in range(rank)), F(0)) for j in range(n)] for i in range(n)]
            h = [[rng.randint(-2, 2) for _ in range(n)] for _ in range(m)]
            nf = [[rng.randint(-1, 1) for _ in range(m // 2)] for _ in range(m)]
            noise = [[sum((nf[i][k] * nf[j][k] for k in range(m // 2)), 0)
                      for j in range(m)] for i in range(m)]
            latent = [rng.randint(-2, 2) for _ in range(rank)]
            error = [rng.randint(-2, 2) for _ in range(m // 2)]
            x = [mean[i] + sum((loadings[i][k] * latent[k] for k in range(rank)), 0)
                 for i in range(n)]
            y = [sum((h[i][j] * x[j] for j in range(n)), F(0)) +
                 sum((nf[i][k] * error[k] for k in range(m // 2)), 0) for i in range(m)]
            if m and case % 3 == 0:
                y[0] += F(1, 7)
            outputs = [[rng.randint(-2, 2) for _ in range(n)] for _ in range(2)]
            observations = {"o" + str(i): dict(zip(variables, row)) for i, row in enumerate(h)}
            request = Request(observations, dict(zip(observations, y)), noise,
                              {"r" + str(i): dict(zip(variables, row)) for i, row in enumerate(outputs)})
            expected = oracle(mean, cov, h, y, noise, outputs)
            precision = [[F(1, 1) / variances[i] if i == j and i < rank else F(0)
                          for j in range(n)] for i in range(n)]
            information = [mean[i] / variances[i] if i < rank else F(0) for i in range(n)]
            constraints = [[F(int(j == i)) - (F(loadings[i][j]) if j < rank else F(0))
                            for j in range(n)] for i in range(rank, n)]
            constraint_values = [sum((a * b for a, b in zip(row, mean)), F(0))
                                 for row in constraints]
            for model in (Model(variables, mean, cov), Affine(variables, mean, loadings, variances),
                          Potential(variables, precision, information, constraints, constraint_values)):
                with self.subTest(case=case, model=type(model).__name__):
                    self.assertEqual(plain(model.evaluate(request)), expected)


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.db = connect(":memory:")
        self.native = NativeArchive(self.db)

    def tearDown(self):
        self.db.close()

    def test_native_representation_and_binding(self):
        native = self.native
        for model in (affine_equal(), potential_equal()):
            native.put_model("m", model)
            native.put_analysis("agree", "m", equality_request())
            native.put_analysis("reject", "m", equality_request(-1))
            self.assertEqual(plain(native.evaluate("agree")), ("ok", (2, 0), ((0, 0), (0, 0))))
            self.assertEqual(native.evaluate("reject").status, "inconsistent")
        for model in (affine_equal(-1), potential_equal(-1)):
            native.put_model("m", model)
            self.assertEqual(native.evaluate("agree").status, "inconsistent")
            self.assertEqual(plain(native.evaluate("reject")), ("ok", (0, 2), ((0, 0), (0, 0))))

    def test_potential_nonzero_information_affine_support(self):
        # X = 1+t, Y = 2+2t, C=-1; t~N(0,1).
        model = Potential(["A", "B", "C"], [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                          [1, 0, 0], [[-2, 1, 0], [0, 0, 1], [-4, 2, 0]], [0, -1, 0])
        request = Request({}, {}, [], {"a": {"A": 1}, "b": {"B": 1}, "c": {"C": 1}})
        self.assertEqual(plain(model.evaluate(request)), ("ok", (1, 2, -1),
                                                         ((1, 2, 0), (2, 4, 0), (0, 0, 0))))
        observed = Request({"a": {"A": 1}, "b": {"B": 1}}, {"a": 3, "b": 6},
                           [[1, 1], [1, 1]], {"a": {"A": 1}, "b": {"B": 1}})
        expected = oracle([1, 2, -1], [[1, 2, 0], [2, 4, 0], [0, 0, 0]],
                          [[1, 0, 0], [0, 1, 0]], [3, 6], [[1, 1], [1, 1]], [[1, 0, 0], [0, 1, 0]])
        self.assertEqual(plain(model.evaluate(observed)), expected)

    def test_potential_full_and_point(self):
        request = Request({}, {}, [], {"x": {"X": 1}, "y": {"Y": 1}})
        full = Potential(["X", "Y"], [[2, 1], [1, 2]], [1, 0], [], [])
        self.assertEqual(plain(full.evaluate(request)), ("ok", (F(2, 3), F(-1, 3)),
                                                        ((F(2, 3), F(-1, 3)), (F(-1, 3), F(2, 3)))))
        for model in (Affine(["X", "Y"], [3, -2], [[], []], []),
                      Potential(["X", "Y"], [[0, 0], [0, 0]], [8, 9],
                                [[1, 0], [0, 1]], [3, -2])):
            self.assertEqual(plain(model.evaluate(request)), ("ok", (3, -2), ((0, 0), (0, 0))))
            self.assertEqual(decode(model.payload()).evaluate(request), model.evaluate(request))

    def test_invalid_models_and_unknown_fields(self):
        with self.assertRaises(ValueError):
            Affine(["X", "Y"], [0, 0], [[1, 2], [1, 2]], [1, 1])
        with self.assertRaises(ValueError):
            Affine(["X"], [0], [[1]], [0])
        with self.assertRaises(ValueError):
            Potential(["X"], [[0]], [0], [], [])
        with self.assertRaises(ValueError):
            Potential(["X"], [[1]], [0], [[1], [1]], [0, 1])
        payload = affine_equal().payload()
        payload["covariance"] = [[1, 1], [1, 1]]
        with self.assertRaises(ValueError):
            decode(payload)

    def test_targets_are_independent_of_legacy_archive(self):
        legacy = build_legacy(self.db)
        self.native.put_model("pair", affine_equal(-1))
        self.native.put_analysis("agree", "pair", equality_request())
        self.assertEqual(legacy.evaluate("agree").status, "ok")
        self.assertEqual(self.native.evaluate("agree").status, "inconsistent")
        self.native.put_model("pair", potential_equal())
        self.assertEqual(self.native.evaluate("agree").status, "ok")
        self.assertEqual(self.db.execute("SELECT value FROM archive_meta WHERE key='format'").fetchone()[0], "1")


class PersistenceTests(unittest.TestCase):
    def test_file_reopen_and_failed_batch(self):
        with tempfile.TemporaryDirectory(prefix="calibration-test-") as folder:
            path = str(Path(folder) / "archive.sqlite")
            db = connect(path)
            try:
                archive = build_legacy(db)
                with self.assertRaisesRegex(RuntimeError, "batch"):
                    with atomic(db):
                        archive.save_model("pair", ["X", "Y"], [0, 0], [[1, -1], [-1, 1]])
                        raise RuntimeError("batch")
            finally:
                db.close()
            db = connect(path)
            try:
                archive = Archive(db)
                self.assertEqual(archive.evaluate("agree").status, "ok")
                self.assertEqual(archive.evaluate("conflict").status, "inconsistent")
                native = NativeArchive(db)
                native.put_model("native", potential_equal(-1))
                native.put_analysis("check", "native", equality_request())
            finally:
                db.close()
            db = connect(path)
            try:
                self.assertEqual(NativeArchive(db).evaluate("check").status, "inconsistent")
                self.assertEqual(Archive(db).evaluate("agree").status, "ok")
                self.assertEqual(db.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
