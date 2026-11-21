"""Saved analysis handles carry identity, never a frozen model or evidence verdict."""
from database import atomic


class Analysis:
    def __init__(self, archive, analysis_id):
        self.archive = archive
        self.analysis_id = analysis_id

    def evaluate(self):
        with atomic(self.archive.db):
            model_id, request = self.archive.load_analysis(self.analysis_id)
            model = self.archive.load_model(model_id)
            return model.evaluate(request)
