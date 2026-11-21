"""Finite hand-auditable calibration fixtures and a complete SDK1 database builder."""
from archive import Archive
from database import atomic
from gauss2 import Affine, Potential
from specs import Request


def equality_request(second=1):
    return Request({"left": {"X": 1}, "right": {"Y": 1}},
                   {"left": 1, "right": second}, [[0, 0], [0, 0]],
                   {"sum": {"X": 1, "Y": 1}, "gap": {"X": 1, "Y": -1}})


def affine_equal(sign=1):
    return Affine(["X", "Y"], [0, 0], [[1], [sign]], [1])


def potential_equal(sign=1):
    return Potential(["X", "Y"], [["1/4", str(sign) + "/4"],
                                    [str(sign) + "/4", "1/4"]],
                     [0, 0], [[1, -sign]], [0])


def build_legacy(db):
    """Populate a caller-supplied empty database, including actual saved requests."""
    with atomic(db):
        archive = Archive(db)
        archive.save_model("pair", ["X", "Y"], [0, 0], [[1, 1], [1, 1]])
        archive.save_model("offset", ["A", "B", "C"], [1, 2, -1],
                           [[1, 2, 0], [2, 4, 0], [0, 0, 0]])
        archive.save_model("full", ["X", "Y"], [0, 0], [[2, 1], [1, 2]])
        archive.save_model("point", ["X"], [3], [[0]])
        for name, req in [("agree", equality_request()), ("conflict", equality_request(-1))]:
            archive.save_analysis(name, "pair", **req.payload())
        archive.save_analysis("noisy", "full", {"u": {"X": 1}, "v": {"Y": 1}},
                              {"u": 2, "v": 0}, [[1, 1], [1, 1]],
                              {"x": {"X": 1}, "y": {"Y": 1}})
        archive.save_analysis("offset_check", "offset", {"a": {"A": 1}, "b": {"B": 1}},
                              {"a": 3, "b": 6}, [[0, 0], [0, 0]],
                              {"b": {"B": 1}, "c": {"C": 1}})
        archive.save_analysis("prior", "pair", {}, {}, [],
                              {"sum": {"X": 1, "Y": 1}, "gap": {"X": 1, "Y": -1}})
    return archive
