"""Site-summary demo: offline compilation, center merge, and a support conflict.

Each site compiles its own packet independently, with no center prior or output
definition.  The demo persists only those bytes, then a fresh center archive
reads nothing but the bytes, merges two sites, and reopens the published result.
"""
import json
import tempfile
from pathlib import Path

from database import connect
from federated import FederatedArchive
from gauss2 import Affine
from site_summary import compile_site

VARIABLES = ["X", "Z"]
OUTPUTS = {"x": {"X": 1}, "z": {"Z": 1}}


def display(result):
    return {"status": result.status, "outputs": list(result.outputs),
            "mean": [str(x) for x in result.mean],
            "covariance": [[str(x) for x in row] for row in result.covariance]}


def independent_prior():
    return Affine(VARIABLES, [0, 0], [[1, 0], [0, 1]], [1, 1])


def site_packets(case_id, rigid_site):
    """Compile each site independently; no center prior or outputs are visible."""
    alpha = compile_site(case_id, "alpha", VARIABLES,
                         {"x": {"X": 1}, "z": {"Z": 1}}, {"x": 3, "z": 1},
                         [[1, 1], [1, 1]])
    beta = compile_site(case_id, "beta", VARIABLES, {"z": {"Z": 1}}, {"z": 2}, [[1]])
    rigid = compile_site(case_id, rigid_site, VARIABLES,
                         {"gap": {"X": 1, "Z": -1}}, {"gap": 1}, [[0]])
    return alpha, beta, rigid


def run():
    with tempfile.TemporaryDirectory(prefix="site-demo-") as folder:
        alpha, beta, _ = site_packets("positive", "beta-rigid")
        conflict_alpha, _, rigid = site_packets("conflict", "beta-rigid")
        alpha_path = Path(folder) / "alpha.json"
        beta_path = Path(folder) / "beta.json"
        alpha_path.write_bytes(alpha)
        beta_path.write_bytes(beta)
        # Source sites are gone: only the two files remain.
        alpha = alpha_path.read_bytes()
        beta = beta_path.read_bytes()

        db = connect(str(Path(folder) / "center.sqlite"))
        try:
            center = FederatedArchive(db)
            center.open_case("positive", independent_prior(), OUTPUTS, ["alpha", "beta"])
            center.add_site("positive", alpha)
            center.add_site("positive", beta)
            merged = center.finish("positive")
            center.open_case("conflict", independent_prior(), OUTPUTS,
                             ["alpha", "beta-rigid"])
            center.add_site("conflict", conflict_alpha)
            center.add_site("conflict", rigid)
            conflict = center.finish("conflict")
        finally:
            db.close()

        db = connect(str(Path(folder) / "center.sqlite"))
        try:
            reopened = FederatedArchive(db).finish("positive")
        finally:
            db.close()
        return {"merged": display(merged), "support_conflict": display(conflict),
                "reopened_matches": reopened == merged}


if __name__ == "__main__":
    import demo
    print(json.dumps({"legacy_demo": demo.run(), "federated": run()},
                     ensure_ascii=False, indent=2))
