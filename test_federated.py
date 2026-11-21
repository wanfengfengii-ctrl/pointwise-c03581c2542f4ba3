"""Federated site-summary tests: compilation, offline merge, persistence, rollback.

The independent oracle path builds an equivalent concatenated-observation
Request with block-diagonal noise and uses the existing SDK to compute the
expected answer, without importing site_summary or federated internals.
"""
import json
import tempfile
import unittest
from fractions import Fraction as F
from pathlib import Path
import random

from database import atomic, connect
from federated import FederatedArchive
from gauss2 import Affine, Potential
from site_summary import compile_site, decode_packet
from specs import Request

VARIABLES = ["X", "Z"]
OUTPUTS = {"x": {"X": 1}, "z": {"Z": 1}}


def prior_independent(variables=VARIABLES):
    n = len(variables)
    return Affine(variables, [0] * n, [[int(i == j) for j in range(n)] for i in range(n)],
                  [1] * n)


def prior_equal(variables=VARIABLES):
    loadings = [[1] for _ in variables]
    return Affine(variables, [0] * len(variables), loadings, [1])


def site_a(case="case", site="A"):
    return compile_site(case, site, VARIABLES,
                        {"a": {"X": 1}, "b": {"Z": 1}}, {"a": 3, "b": 1},
                        [[1, 1], [1, 1]])


def site_b(case="case", site="B"):
    return compile_site(case, site, VARIABLES, {"z": {"Z": 1}}, {"z": 2}, [[1]])


def site_b_rigid(case="case", site="B"):
    return compile_site(case, site, VARIABLES, {"d": {"X": 1, "Z": -1}}, {"d": 1}, [[0]])


def expected_by_concat(prior, specs, outputs):
    """Concatenate site observations with block-diagonal noise and use SDK2."""
    observations, values, sizes = {}, {}, []
    for site_id, obs, vals, noise in specs:
        for label, terms in obs.items():
            observations[site_id + "|" + label] = terms
            values[site_id + "|" + label] = vals[label]
        sizes.append((len(obs), noise))
    total = sum(size for size, _ in sizes)
    big = [[F(0)] * total for _ in range(total)]
    offset = 0
    for size, noise in sizes:
        for i in range(size):
            for j in range(size):
                big[offset + i][offset + j] = F(noise[i][j])
        offset += size
    return prior.evaluate(Request(observations, values, big, outputs))


def run_case(tag, prior, specs, outputs, packets, site_order=None):
    db = connect(":memory:")
    try:
        center = FederatedArchive(db)
        order = site_order or [s[0] for s in specs]
        center.open_case(tag, prior, outputs, order)
        for packet in packets:
            center.add_site(tag, packet)
        return center.finish(tag)
    finally:
        db.close()


class CompileTests(unittest.TestCase):
    def test_deterministic_bytes_and_field_shape(self):
        first = site_a()
        second = site_a()
        self.assertIsInstance(first, bytes)
        self.assertEqual(first, second)
        document = json.loads(first)
        self.assertEqual(set(document), {"version", "case_id", "site_id",
                                         "variables", "body"})
        self.assertEqual(document["version"], 1)
        body = document["body"]
        # No original observation labels or observation-sized arrays leak out.
        self.assertNotIn("a", body)
        self.assertNotIn("b", body)
        self.assertEqual(len(body["precision"]), 2)
        self.assertEqual(len(body["information"]), 2)
        self.assertLessEqual(len(body["constraints"]), len(VARIABLES))

    def test_zero_noise_is_hard_support(self):
        packet = decode_packet(site_b_rigid())
        self.assertTrue(packet.feasible)
        self.assertEqual([list(row) for row in packet.constraints], [[F(1), F(-1)]])
        self.assertEqual(list(packet.values), [F(1)])
        # Rank-one zero noise carries no relative weight.
        self.assertTrue(all(x == 0 for row in packet.precision for x in row))
        self.assertTrue(all(x == 0 for x in packet.information))

    def test_site_local_contradiction_is_marked_not_raised(self):
        packet = compile_site("c", "A", VARIABLES, {"p": {"X": 1}, "q": {"X": 1}},
                              {"p": 1, "q": 2}, [[0, 0], [0, 0]])
        decoded = decode_packet(packet)
        self.assertFalse(decoded.feasible)
        self.assertEqual(decoded.constraints, ())
        # Malformed input still raises.
        with self.assertRaises(ValueError):
            compile_site("c", "A", VARIABLES, {"p": {"Q": 1}}, {"p": 1}, [[1]])

    def test_empty_observation_site_is_valid(self):
        packet = compile_site("c", "A", VARIABLES, {}, {}, [])
        decoded = decode_packet(packet)
        self.assertTrue(decoded.feasible)
        self.assertEqual(decoded.constraints, ())
        self.assertTrue(all(x == 0 for row in decoded.precision for x in row))

    def test_packet_scale_is_bounded_by_variable_count(self):
        # Eight observations but only two variables -> summary stays 2x2.
        observations = {}
        values = {}
        rows = []
        for i in range(8):
            observations["o" + str(i)] = {"X": 1}
            values["o" + str(i)] = i
            rows.append([1, 0])
        noise = [[int(i == j) for j in range(8)] for i in range(8)]
        decoded = decode_packet(compile_site("c", "A", VARIABLES, observations, values, noise))
        self.assertEqual(len(decoded.precision), 2)
        self.assertEqual(len(decoded.information), 2)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            compile_site("", "A", VARIABLES, {}, {}, [])
        with self.assertRaises(ValueError):
            compile_site("c", "", VARIABLES, {}, {}, [])
        with self.assertRaises(ValueError):
            compile_site("c", "A", [], {}, {}, [])
        with self.assertRaises(ValueError):
            compile_site("c", "A", ["A"] * 2, {}, {}, [])
        with self.assertRaises(ValueError):
            compile_site("c", "A", ["v" + str(i) for i in range(7)], {}, {}, [])
        with self.assertRaises(ValueError):
            compile_site("c", "A", VARIABLES, {"o": {"Q": 1}}, {"o": 0}, [[1]])
        with self.assertRaises(ValueError):
            compile_site("c", "A", VARIABLES, {"o": {"X": 1}}, {"o": 0}, [[-1]])
        with self.assertRaises(ValueError):
            compile_site("c", "A", VARIABLES, {"o": {"X": 1}}, {"o": 0}, [[1, 0]])

    def test_decode_rejects_structural_and_math_errors(self):
        good = json.loads(site_a())

        def mutated(**changes):
            copy = json.loads(site_a())
            copy.update(changes)
            return json.dumps(copy).encode()

        with self.assertRaises(ValueError):
            decode_packet(mutated(extra=1))
        with self.assertRaises(ValueError):
            decode_packet(mutated(version=2))
        with self.assertRaises(ValueError):
            decode_packet(mutated(version=True))
        broken = json.loads(site_a())
        del broken["body"]["feasible"]
        with self.assertRaises(ValueError):
            decode_packet(json.dumps(broken).encode())
        nonsymmetric = json.loads(site_a())
        nonsymmetric["body"]["precision"] = [["1", "0"], ["1", "1"]]
        with self.assertRaises(ValueError):
            decode_packet(json.dumps(nonsymmetric).encode())
        negative = json.loads(site_a())
        negative["body"]["precision"] = [["-1", "0"], ["0", "0"]]
        with self.assertRaises(ValueError):
            decode_packet(json.dumps(negative).encode())
        unsorted = json.loads(site_a())
        unsorted["body"]["constraints"] = [["0", "1"], ["1", "0"]]
        unsorted["body"]["values"] = ["1", "1"]
        with self.assertRaises(ValueError):
            decode_packet(json.dumps(unsorted).encode())
        with self.assertRaises(ValueError):
            decode_packet(b'{"version":1,"version":1,"case_id":"c",'
                          b'"site_id":"A","variables":["X"],"body":{}}')
        with self.assertRaises(ValueError):
            decode_packet(b'{"version":NaN}')
        with self.assertRaises(ValueError):
            decode_packet(b'\xff\xfe')


class MergeTests(unittest.TestCase):
    def test_published_example(self):
        result = run_case("case", prior_independent(),
                          [("A", None, None, None), ("B", None, None, None)],
                          OUTPUTS, [site_a(), site_b()])
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.mean, (F(9, 4), F(1, 4)))
        self.assertEqual(result.covariance, ((F(1, 4), F(1, 4)),
                                             (F(1, 4), F(1, 4))))

    def test_rigid_conflict_between_sites(self):
        result = run_case("case", prior_independent(),
                          [("A", None, None, None), ("B", None, None, None)],
                          OUTPUTS, [site_a(), site_b_rigid()])
        self.assertEqual(result.status, "inconsistent")
        self.assertEqual(result.outputs, ("x", "z"))
        self.assertEqual(result.mean, ())
        self.assertEqual(result.covariance, ())

    def test_degenerate_prior_conflict(self):
        result = run_case("case", prior_equal(),
                          [("A", None, None, None), ("B", None, None, None)],
                          OUTPUTS, [site_a(), site_b()])
        self.assertEqual(result.status, "inconsistent")

    def test_infeasible_site_forces_inconsistent(self):
        bad = compile_site("case", "A", VARIABLES, {"p": {"X": 1}, "q": {"X": 1}},
                           {"p": 1, "q": 2}, [[0, 0], [0, 0]])
        result = run_case("case", prior_independent(),
                          [("A", None, None, None), ("B", None, None, None)],
                          OUTPUTS, [bad, site_b("case")])
        self.assertEqual(result.status, "inconsistent")

    def test_empty_site_is_neutral(self):
        empty = compile_site("case", "A", VARIABLES, {}, {}, [])
        result = run_case("case", prior_independent(),
                          [("A", None, None, None), ("B", None, None, None)],
                          OUTPUTS, [empty, site_b("case")])
        # Only Z=2 with noise 1 pulls Z from its prior; X stays at its prior.
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.mean, (F(0), F(1)))
        self.assertEqual(result.covariance, ((F(1), F(0)), (F(0), F(1, 2))))

    def test_three_sites(self):
        third = compile_site("case", "C", VARIABLES, {"z": {"Z": 1}}, {"z": 0}, [[1]])
        result = run_case("case", prior_independent(),
                          [("A", None, None, None), ("B", None, None, None),
                           ("C", None, None, None)],
                          OUTPUTS, [site_a("case"), site_b("case"), third])
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.mean, (F(11, 5), F(1, 5)))
        self.assertEqual(result.covariance, ((F(1, 5), F(1, 5)), (F(1, 5), F(1, 5))))

    def test_output_order_and_linear_combination(self):
        outputs = {"sum": {"X": 1, "Z": 1}, "x": {"X": 1}, "minus": {"X": -1, "Z": 1}}
        result = run_case("case", prior_independent(),
                          [("A", None, None, None), ("B", None, None, None)],
                          outputs, [site_a(), site_b()])
        self.assertEqual(result.outputs, ("sum", "x", "minus"))
        self.assertEqual(result.mean, (F(5, 2), F(9, 4), F(-2)))
        self.assertEqual(result.covariance,
                         ((F(1), F(1, 2), F(0)),
                          (F(1, 2), F(1, 4), F(0)),
                          (F(0), F(0), F(0))))

    def test_potential_prior(self):
        # Same prior via Potential: X, Z independent N(0,1).
        prior = Potential(VARIABLES, [[1, 0], [0, 1]], [0, 0], [], [])
        result = run_case("case", prior,
                          [("A", None, None, None), ("B", None, None, None)],
                          OUTPUTS, [site_a(), site_b()])
        self.assertEqual(result.mean, (F(9, 4), F(1, 4)))
        self.assertEqual(result.covariance, ((F(1, 4), F(1, 4)), (F(1, 4), F(1, 4))))

    def test_prior_degenerate_potential_constraint(self):
        # Prior X = Z via Potential constraint.
        prior = Potential(VARIABLES, [[1, 1], [1, 1]], [0, 0], [[1, -1]], [0])
        result = run_case("case", prior,
                          [("A", None, None, None), ("B", None, None, None)],
                          OUTPUTS, [site_a(), site_b()])
        self.assertEqual(result.status, "inconsistent")

    def test_three_frame_independence(self):
        # Same packet works under different priors and output sets.
        packet = site_b()
        for prior, outputs in [
            (prior_independent(), OUTPUTS),
            (Affine(VARIABLES, [1, 2], [[1, 0], [0, 1]], [2, 3]),
             {"z": {"Z": 1}}),
            (prior_equal(), {"diff": {"X": -1, "Z": 1}}),
        ]:
            db = connect(":memory:")
            try:
                center = FederatedArchive(db)
                other = compile_site("case", "A", VARIABLES, {}, {}, [])
                center.open_case("case", prior, outputs, ["A", "B"])
                center.add_site("case", other)
                center.add_site("case", packet)
                self.assertEqual(center.finish("case").status, "ok")
            finally:
                db.close()

    def test_independent_block_diagonal_oracle(self):
        rng = random.Random(90210)
        checked = 0
        for case in range(120):
            n = rng.randint(1, 4)
            variables = ["v" + str(i) for i in range(n)]
            rank = rng.randint(0, n)
            loadings = [[F(int(i == j)) + (F(rng.randint(-1, 1)) if j < i else F(0))
                         for j in range(rank)] for i in range(n)]
            variances = [F(rng.randint(1, 3)) for _ in range(rank)]
            prior = Affine(variables, [F(rng.randint(-2, 2)) for _ in range(n)],
                           loadings, variances)
            specs, packets = [], []
            total_rows = 0
            for s in range(rng.randint(2, 3)):
                m = rng.randint(0, 2)
                total_rows += m
                h = [[rng.randint(-2, 2) for _ in range(n)] for _ in range(m)]
                observations = {"o" + str(i): dict(zip(variables, row))
                                for i, row in enumerate(h)}
                values = {"o" + str(i): F(rng.randint(-2, 2)) for i in range(m)}
                k = rng.randint(0, m)
                factor = [[F(rng.randint(-1, 1)) for _ in range(k)] for _ in range(m)]
                noise = [[sum((factor[i][t] * factor[j][t] for t in range(k)), F(0))
                          for j in range(m)] for i in range(m)]
                specs.append(("S" + str(s), observations, values, noise))
                packets.append(compile_site("case", "S" + str(s), variables,
                                            observations, values, noise))
            if total_rows > 8:
                continue  # Request caps concatenated rows; separate concern.
            outputs = {"r0": dict(zip(variables, [rng.randint(-1, 1) for _ in range(n)])),
                       "r1": dict(zip(variables, [rng.randint(-1, 1) for _ in range(n)]))}
            expected = expected_by_concat(prior, specs, outputs)
            result = run_case("case", prior,
                              [(sid, None, None, None) for sid, _, _, _ in specs],
                              outputs, packets)
            self.assertEqual(result.status, expected.status)
            self.assertEqual(result.mean, expected.mean)
            self.assertEqual(result.covariance, expected.covariance)
            checked += 1
        self.assertGreater(checked, 80)

    def test_correlated_prior_matches_concatenated(self):
        from exact import inverse
        covariance = [[5, 3], [3, 3]]
        precision = inverse(covariance)
        information = [sum(precision[i][j] * [1, 2][j] for j in range(2)) for i in range(2)]
        priors = (Affine(VARIABLES, [1, 2], [[1, 1], [0, 1]], [2, 3]),
                  Potential(VARIABLES, precision, information, [], []))
        # Same observations as the published example, but a nonzero correlated prior.
        specs = [("A", {"a": {"X": 1}, "b": {"Z": 1}}, {"a": 3, "b": 1}, [[1, 1], [1, 1]]),
                 ("B", {"z": {"Z": 1}}, {"z": 2}, [[1]])]
        for prior in priors:
            expected = expected_by_concat(prior, specs, OUTPUTS)
            result = run_case("case", prior,
                              [(sid, None, None, None) for sid, _, _, _ in specs],
                              OUTPUTS, [compile_site("case", sid, VARIABLES, obs, vals, noise)
                                        for sid, obs, vals, noise in specs])
            self.assertEqual(result.status, expected.status)
            self.assertEqual(result.mean, expected.mean)
            self.assertEqual(result.covariance, expected.covariance)
        self.assertEqual(result.mean, (F(25, 7), F(11, 7)))

    def test_rank_deficient_multiple_rows_match(self):
        # Three correlated (rank-one) rows carry the same evidence as one.
        site = compile_site("case", "B", VARIABLES,
                            {"b1": {"Z": 1}, "b2": {"Z": 1}, "b3": {"Z": 1}},
                            {"b1": 2, "b2": 2, "b3": 2},
                            [[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        fixed = compile_site("case", "A", VARIABLES, {"a": {"X": 1}}, {"a": 0}, [[0]])
        specs = [("A", {"a": {"X": 1}}, {"a": 0}, [[0]]),
                 ("B", {"b1": {"Z": 1}, "b2": {"Z": 1}, "b3": {"Z": 1}},
                  {"b1": 2, "b2": 2, "b3": 2}, [[1, 1, 1], [1, 1, 1], [1, 1, 1]])]
        expected = expected_by_concat(prior_independent(), specs, OUTPUTS)
        result = run_case("case", prior_independent(),
                          [("A", None, None, None), ("B", None, None, None)],
                          OUTPUTS, [fixed, site])
        self.assertEqual(result.mean, expected.mean)
        self.assertEqual(result.covariance, expected.covariance)
        self.assertEqual(result.mean, (F(0), F(1)))

    def test_six_variables(self):
        variables = ["v" + str(i) for i in range(6)]
        prior = Affine(variables, [0] * 6,
                       [[int(i == j) for j in range(6)] for i in range(6)], [1] * 6)
        outputs = {"o" + str(j): {"v" + str(i): int(i == j) for i in range(6)}
                   for j in range(6)}
        observations = {"a" + str(i): {"v" + str(i): 1} for i in range(6)}
        values = {"a" + str(i): i + 1 for i in range(6)}
        site = compile_site("case", "A", variables, observations, values,
                            [[int(i == j) for j in range(6)] for i in range(6)])
        fixed = compile_site("case", "B", variables, {"b": {"v0": 1}}, {"b": 10}, [[0]])
        result = run_case("case", prior,
                          [("A", None, None, None), ("B", None, None, None)],
                          outputs, [site, fixed])
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.mean, (F(10), F(1), F(3, 2), F(2), F(5, 2), F(3)))


class ExactHelperTests(unittest.TestCase):
    def test_pseudo_inverse_moore_penrose(self):
        from exact import multiply, pseudo_inverse, transpose
        rng = random.Random(2024)
        for _ in range(300):
            n = rng.randint(1, 5)
            k = rng.randint(0, n)
            factor = [[F(rng.randint(-2, 2)) for _ in range(k)] for _ in range(n)]
            weights = [F(rng.randint(1, 3)) for _ in range(k)]
            m = [[sum((factor[i][t] * weights[t] * factor[j][t] for t in range(k)), F(0))
                  for j in range(n)] for i in range(n)]
            p = pseudo_inverse(m)
            self.assertEqual(multiply(multiply(m, p, n), m, n), m)
            self.assertEqual(multiply(multiply(p, m, n), p, n), p)
            self.assertEqual(multiply(m, p, n), transpose(multiply(m, p, n), n))
            self.assertEqual(multiply(p, m, n), transpose(multiply(p, m, n), n))

    def test_pseudo_inverse_rejects_non_psd(self):
        from exact import pseudo_inverse
        for bad in ([[1, 2], [2, 1]], [[-1]], [[0, 1], [1, 0]]):
            with self.assertRaises(ValueError):
                pseudo_inverse(bad)

    def test_canonical_constraints(self):
        from exact import canonical_constraints
        self.assertEqual(canonical_constraints([[1, 1], [2, 2]], [1, 2], 2),
                         ([[F(1), F(1)]], [F(1)]))
        self.assertIsNone(canonical_constraints([[1, 1], [2, 2]], [1, 3], 2))


class ArchiveTests(unittest.TestCase):
    def test_reopen_after_source_removed(self):
        with tempfile.TemporaryDirectory(prefix="federated-test-") as folder:
            path = str(Path(folder) / "center.sqlite")
            db = connect(path)
            try:
                center = FederatedArchive(db)
                center.open_case("case", prior_independent(), OUTPUTS, ["A", "B"])
                center.add_site("case", site_a())
                center.add_site("case", site_b())
                first = center.finish("case")
            finally:
                db.close()
            db = connect(path)
            try:
                center = FederatedArchive(db)  # re-initialization keeps the case
                self.assertEqual(center.finish("case"), first)
                self.assertEqual(center.add_site("case", site_a()), site_a())
                self.assertEqual(center.finish("case"), first)
            finally:
                db.close()

    def test_order_of_arrival_is_irrelevant(self):
        first = run_case("case", prior_independent(),
                         [("A", None, None, None), ("B", None, None, None)],
                         OUTPUTS, [site_a(), site_b()])
        second = run_case("case", prior_independent(),
                          [("A", None, None, None), ("B", None, None, None)],
                          OUTPUTS, [site_b(), site_a()])
        self.assertEqual(first, second)

    def test_reorder_declaration(self):
        # Declaration order affects only the site list identity, not the result.
        a, b = site_a(), site_b()
        db = connect(":memory:")
        try:
            center = FederatedArchive(db)
            center.open_case("case", prior_independent(), OUTPUTS, ["A", "B"])
            center.add_site("case", a)
            center.add_site("case", b)
            one = center.finish("case")
        finally:
            db.close()
        self.assertEqual(one.mean, (F(9, 4), F(1, 4)))

    def test_missing_site_does_not_freeze_and_can_continue(self):
        db = connect(":memory:")
        try:
            center = FederatedArchive(db)
            center.open_case("case", prior_independent(), OUTPUTS, ["A", "B"])
            center.add_site("case", site_a())
            with self.assertRaises(ValueError):
                center.finish("case")
            center.add_site("case", site_b())
            self.assertEqual(center.finish("case").status, "ok")
        finally:
            db.close()

    def test_duplicate_case_same_content_and_conflict(self):
        db = connect(":memory:")
        try:
            center = FederatedArchive(db)
            center.open_case("case", prior_independent(), OUTPUTS, ["A", "B"])
            center.open_case("case", prior_independent(), OUTPUTS, ["A", "B"])
            with self.assertRaises(ValueError):
                center.open_case("case", prior_independent(), OUTPUTS, ["B", "A"])
            with self.assertRaises(ValueError):
                center.open_case("case", prior_equal(), OUTPUTS, ["A", "B"])
        finally:
            db.close()

    def test_unknown_case(self):
        db = connect(":memory:")
        try:
            center = FederatedArchive(db)
            with self.assertRaises(KeyError):
                center.add_site("nope", site_a())
            with self.assertRaises(KeyError):
                center.finish("nope")
        finally:
            db.close()

    def test_same_site_same_bytes_returns_first_and_different_rejected(self):
        db = connect(":memory:")
        try:
            center = FederatedArchive(db)
            center.open_case("case", prior_independent(), OUTPUTS, ["A", "B"])
            self.assertEqual(center.add_site("case", site_a()), site_a())
            self.assertEqual(center.add_site("case", site_a()), site_a())
            other = compile_site("case", "A", VARIABLES, {"a": {"X": 1}}, {"a": 3}, [[1]])
            with self.assertRaises(ValueError):
                center.add_site("case", other)
        finally:
            db.close()

    def test_binding_errors(self):
        db = connect(":memory:")
        try:
            center = FederatedArchive(db)
            center.open_case("case", prior_independent(), OUTPUTS, ["A", "B"])
            with self.assertRaises(ValueError):
                center.add_site("case", site_a(case="other"))
            with self.assertRaises(ValueError):
                center.add_site("case", compile_site("case", "Z", VARIABLES, {}, {}, []))
            swapped = compile_site("case", "A", ["Z", "X"],
                                   {"a": {"X": 1}}, {"a": 1}, [[1]])
            with self.assertRaises(ValueError):
                center.add_site("case", swapped)
            with self.assertRaises(ValueError):
                center.add_site("case", b"{}")
        finally:
            db.close()

    def test_failure_leaves_no_partial_state(self):
        db = connect(":memory:")
        try:
            center = FederatedArchive(db)
            center.open_case("case", prior_independent(), OUTPUTS, ["A", "B"])
            center.add_site("case", site_a())
            before = db.serialize()
            with self.assertRaises(ValueError):
                center.add_site("case", site_a(case="other"))
            with self.assertRaises(ValueError):
                center.add_site("case", b"not-json")
            self.assertEqual(db.serialize(), before)
            with self.assertRaises(ValueError):
                center.finish("case")
            self.assertEqual(db.serialize(), before)
            center.add_site("case", site_b())
            self.assertEqual(center.finish("case").status, "ok")
        finally:
            db.close()

    def test_open_case_atomic_on_failure(self):
        db = connect(":memory:")
        try:
            center = FederatedArchive(db)
            before = db.serialize()
            with self.assertRaises(ValueError):
                center.open_case("case", prior_independent(), {}, ["A", "B"])
            with self.assertRaises(ValueError):
                center.open_case("case", prior_independent(), OUTPUTS, ["A"])
            with self.assertRaises(ValueError):
                center.open_case("case", prior_independent(), OUTPUTS, ["A", "A"])
            with self.assertRaises(ValueError):
                center.open_case("case", object(), OUTPUTS, ["A", "B"])
            self.assertEqual(db.serialize(), before)
        finally:
            db.close()

    def test_caller_mutation_does_not_change_case(self):
        db = connect(":memory:")
        try:
            prior = prior_independent()
            outputs = {"x": {"X": 1}, "z": {"Z": 1}}
            center = FederatedArchive(db)
            center.open_case("case", prior, outputs, ["A", "B"])
            outputs["x"]["X"] = 5
            outputs["new"] = {"Z": 1}
            center.add_site("case", site_a())
            center.add_site("case", site_b())
            result = center.finish("case")
            self.assertEqual(result.outputs, ("x", "z"))
            self.assertEqual(result.mean, (F(9, 4), F(1, 4)))
        finally:
            db.close()

    def test_result_stable_after_new_connection(self):
        with tempfile.TemporaryDirectory(prefix="federated-test-") as folder:
            path = str(Path(folder) / "center.sqlite")
            db = connect(path)
            try:
                center = FederatedArchive(db)
                center.open_case("case", prior_independent(), OUTPUTS, ["A", "B"])
                center.add_site("case", site_a())
                center.add_site("case", site_b())
                first = center.finish("case")
            finally:
                db.close()
            db = connect(path)
            try:
                center = FederatedArchive(db)
                self.assertEqual(center.finish("case"), first)
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
