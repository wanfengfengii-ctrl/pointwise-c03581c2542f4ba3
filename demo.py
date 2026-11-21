"""Actual old archive and separate SDK2 native examples; no simulated migration."""
from pathlib import Path
import json
import tempfile
from analysis import Analysis
from archive import Archive
from database import connect
from gauss2 import NativeArchive
from samples import (affine_equal, build_legacy, equality_request, potential_equal)


def display(result):
    return {"status": result.status, "outputs": list(result.outputs),
            "mean": [str(x) for x in result.mean],
            "covariance": [[str(x) for x in row] for row in result.covariance]}


def run():
    with tempfile.TemporaryDirectory(prefix="calibration-demo-") as folder:
        path = str(Path(folder) / "legacy.sqlite")
        db = connect(path)
        try:
            archive = build_legacy(db)
            report = {"legacy_agree": display(archive.evaluate("agree")),
                      "legacy_conflict": display(archive.evaluate("conflict")),
                      "legacy_correlated_noise": display(archive.evaluate("noisy"))}
            held = Analysis(archive, "agree")
            archive.save_model("pair", ["X", "Y"], [0, 0], [[1, -1], [-1, 1]])
            report["legacy_replacement"] = display(held.evaluate())
        finally:
            db.close()
        db = connect(path)
        try:
            report["legacy_reopened"] = display(Archive(db).evaluate("agree"))
        finally:
            db.close()
        db = connect(str(Path(folder) / "native.sqlite"))
        try:
            native = NativeArchive(db)
            native.put_model("pair", affine_equal())
            native.put_analysis("check", "pair", equality_request())
            report["native_affine"] = display(native.evaluate("check"))
            native.put_model("pair", potential_equal(-1))
            report["native_potential_replacement"] = display(native.evaluate("check"))
        finally:
            db.close()
        return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
