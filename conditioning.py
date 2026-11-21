"""Exact Gaussian conditional moments, including singular evidence covariance."""
from exact import (add, affine_solve, matvec, multiply, subtract, transpose)
from specs import Result


def condition(mean, covariance, compiled):
    _, h, values, noise, labels, outputs = compiled
    n = len(mean)
    m = len(values)
    hc = multiply(h, covariance)
    evidence_cov = add(multiply(hc, transpose(h, n)), noise)
    residual = [v - expected for v, expected in zip(values, matvec(h, mean))]
    solved = affine_solve(evidence_cov, residual, m)
    if solved is None:
        return Result("inconsistent", labels)
    # Any exact solution is valid: covariance cross terms annihilate the kernel.
    cross = multiply(covariance, transpose(h, n))
    shift = matvec(cross, solved[0])
    posterior_mean = [a + b for a, b in zip(mean, shift)]
    solved_columns = []
    for j in range(n):
        column = affine_solve(evidence_cov, [row[j] for row in hc], m)
        if column is None:
            raise ArithmeticError("invalid joint covariance")
        solved_columns.append(column[0])
    correction = multiply(cross, transpose(solved_columns, m), n)
    posterior_cov = subtract(covariance, correction)
    out_mean = matvec(outputs, posterior_mean)
    out_cov = multiply(multiply(outputs, posterior_cov), transpose(outputs, n))
    return Result("ok", labels, tuple(out_mean), tuple(tuple(row) for row in out_cov))
