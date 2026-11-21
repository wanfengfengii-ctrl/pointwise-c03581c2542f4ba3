"""SDK1: at most eight rational variables, mean and PSD covariance."""
from dataclasses import dataclass
from exact import matrix, psd, vector
from specs import names
from conditioning import condition


@dataclass(frozen=True)
class Model:
    variables: tuple
    mean: tuple
    covariance: tuple

    def __init__(self, variables, mean, covariance):
        variables = names(variables)
        mean = vector(mean, len(variables))
        covariance = matrix(covariance, len(variables), len(variables))
        psd(covariance)
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "mean", tuple(mean))
        object.__setattr__(self, "covariance", tuple(tuple(row) for row in covariance))

    def evaluate(self, request):
        return condition(self.mean, self.covariance, request.compile(self.variables))

    def payload(self):
        return {"variables": list(self.variables),
                "mean": [str(x) for x in self.mean],
                "covariance": [[str(x) for x in row] for row in self.covariance]}

    @classmethod
    def from_payload(cls, payload):
        if set(payload) != {"variables", "mean", "covariance"}:
            raise ValueError("SDK1 fields")
        return cls(**payload)
