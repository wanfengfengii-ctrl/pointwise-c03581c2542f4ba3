"""Named, serializable linear observations and outputs."""
from dataclasses import dataclass
from fractions import Fraction
from exact import matrix, psd, rational


def names(values, limit=8, empty=False):
    values = tuple(values)
    if (not empty and not values) or len(values) > limit:
        raise ValueError("name count")
    if any(not isinstance(x, str) or not x for x in values) or len(set(values)) != len(values):
        raise ValueError("names must be unique nonempty strings")
    return values


def linear_rows(rows, variables, limit):
    if not isinstance(rows, dict):
        raise ValueError("named rows must be a dict")
    labels = names(rows, limit, empty=True)
    result = []
    for label in labels:
        terms = rows[label]
        if not isinstance(terms, dict) or set(terms) - set(variables):
            raise ValueError("unknown variable")
        result.append([rational(terms.get(v, 0)) for v in variables])
    return labels, result


@dataclass(frozen=True)
class Result:
    status: str
    outputs: tuple
    mean: tuple = ()
    covariance: tuple = ()


@dataclass(frozen=True)
class Request:
    observations: dict
    values: dict
    noise: object
    outputs: dict

    def compile(self, variables):
        labels, h = linear_rows(self.observations, variables, 8)
        if not isinstance(self.values, dict) or set(self.values) != set(labels):
            raise ValueError("observation values must match labels")
        y = [rational(self.values[label]) for label in labels]
        noise = matrix(self.noise, len(labels), len(labels))
        psd(noise)
        output_labels, l = linear_rows(self.outputs, variables, 8)
        if not output_labels:
            raise ValueError("at least one output required")
        return labels, h, y, noise, output_labels, l

    def payload(self):
        def rows(value):
            return {label: {v: str(rational(c)) for v, c in terms.items()}
                    for label, terms in value.items()}
        return {"observations": rows(self.observations),
                "values": {label: str(rational(v)) for label, v in self.values.items()},
                "noise": [[str(rational(v)) for v in row] for row in self.noise],
                "outputs": rows(self.outputs)}

    @classmethod
    def from_payload(cls, value):
        if set(value) != {"observations", "values", "noise", "outputs"}:
            raise ValueError("request fields")
        return cls(**value)
