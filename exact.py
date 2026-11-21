"""Small exact rational linear algebra; no probabilistic representation conversion."""
from fractions import Fraction
from itertools import combinations


def rational(value):
    if isinstance(value, bool) or not isinstance(value, (int, str, Fraction)):
        raise ValueError("expected integer, fraction string or Fraction")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("invalid rational") from exc


def vector(values, size):
    result = [rational(x) for x in values]
    if len(result) != size:
        raise ValueError("vector shape")
    return result


def matrix(values, rows, cols):
    result = [vector(row, cols) for row in values]
    if len(result) != rows:
        raise ValueError("matrix shape")
    return result


def zeros(rows, cols):
    return [[Fraction(0) for _ in range(cols)] for _ in range(rows)]


def identity(n):
    return [[Fraction(i == j) for j in range(n)] for i in range(n)]


def transpose(a, cols=None):
    n = len(a[0]) if a else (0 if cols is None else cols)
    return [[a[i][j] for i in range(len(a))] for j in range(n)]


def multiply(a, b, cols=None):
    k = len(b)
    n = len(b[0]) if b else (0 if cols is None else cols)
    if any(len(row) != k for row in a):
        raise ValueError("product shape")
    return [[sum((row[t] * b[t][j] for t in range(k)), Fraction(0))
             for j in range(n)] for row in a]


def matvec(a, x):
    return [sum((u * v for u, v in zip(row, x)), Fraction(0)) for row in a]


def add(a, b):
    return [[u + v for u, v in zip(x, y)] for x, y in zip(a, b)]


def subtract(a, b):
    return [[u - v for u, v in zip(x, y)] for x, y in zip(a, b)]


def affine_solve(a, b, n):
    """Return one solution and a matrix whose columns span the homogeneous kernel.

    None means inconsistent. Free coordinates of the particular solution are zero.
    Empty equation systems and zero-dimensional solution spaces are supported.
    """
    a = matrix(a, len(a), n)
    b = vector(b, len(a))
    work = [row + [rhs] for row, rhs in zip(a, b)]
    pivots = []
    lead = 0
    for col in range(n):
        pivot = next((r for r in range(lead, len(work)) if work[r][col]), None)
        if pivot is None:
            continue
        work[lead], work[pivot] = work[pivot], work[lead]
        scale = work[lead][col]
        work[lead] = [v / scale for v in work[lead]]
        for r in range(len(work)):
            if r != lead and work[r][col]:
                scale = work[r][col]
                work[r] = [x - scale * y for x, y in zip(work[r], work[lead])]
        pivots.append(col)
        lead += 1
    if any(not any(row[:n]) and row[n] for row in work):
        return None
    particular = [Fraction(0)] * n
    for r, col in enumerate(pivots):
        particular[col] = work[r][n]
    free = [j for j in range(n) if j not in pivots]
    basis = zeros(n, len(free))
    for j, col in enumerate(free):
        basis[col][j] = Fraction(1)
        for r, pivot in enumerate(pivots):
            basis[pivot][j] = -work[r][col]
    return particular, basis


def inverse(a):
    n = len(a)
    a = matrix(a, n, n)
    columns = []
    for j in range(n):
        solved = affine_solve(a, [int(i == j) for i in range(n)], n)
        if solved is None or (n and len(solved[1][0])):
            raise ValueError("singular matrix")
        columns.append(solved[0])
    return transpose(columns, n)


def canonical_constraints(a, b, n):
    """Reduce an affine system to independent reduced-row-echelon equations.

    Returns (rows, values) with rows spanning the same solution set, or None
    when no point satisfies the system.  The result contains at most n rows.
    """
    if len(a) != len(b):
        raise ValueError("system shape")
    work = [list(row) + [rhs] for row, rhs in zip(matrix(a, len(a), n), vector(b, len(b)))]
    pivots = []
    lead = 0
    for col in range(n):
        pivot = next((r for r in range(lead, len(work)) if work[r][col]), None)
        if pivot is None:
            continue
        work[lead], work[pivot] = work[pivot], work[lead]
        scale = work[lead][col]
        work[lead] = [v / scale for v in work[lead]]
        for r in range(len(work)):
            if r != lead and work[r][col]:
                factor = work[r][col]
                work[r] = [x - factor * y for x, y in zip(work[r], work[lead])]
        pivots.append(col)
        lead += 1
    for r in range(lead, len(work)):
        if work[r][n]:
            return None
    return [work[r][:n] for r in range(lead)], [work[r][n] for r in range(lead)]


def psd_factor(a):
    """Factor a symmetric PSD matrix as sum(weights[j] outer(v_j, v_j)).

    Exact diagonal-pivoted elimination; raises ValueError when the input is not
    symmetric positive semidefinite.
    """
    n = len(a)
    s = [list(row) for row in matrix(a, n, n)]
    symmetric(s)
    columns, weights = [], []
    for _ in range(n):
        pivot = next((i for i in range(n) if s[i][i]), None)
        if pivot is None:
            break
        d = s[pivot][pivot]
        if d < 0:
            raise ValueError("not positive semidefinite")
        v = [s[i][pivot] for i in range(n)]
        for i in range(n):
            if v[i]:
                factor = v[i] / d
                s[i] = [x - factor * y for x, y in zip(s[i], v)]
        columns.append(v)
        weights.append(Fraction(1) / d)
    if any(any(x for x in row) for row in s):
        raise ValueError("not positive semidefinite")
    return columns, weights


def pseudo_inverse(a):
    """Exact Moore-Penrose inverse of a symmetric positive semidefinite matrix."""
    n = len(a)
    a = matrix(a, n, n)
    columns, weights = psd_factor(a)
    r = len(columns)
    if r == 0:
        return zeros(n, n)
    basis = [[columns[j][i] for j in range(r)] for i in range(n)]
    bt = transpose(basis, r)
    gram_inverse = inverse(multiply(bt, basis, r))
    work = multiply(basis, gram_inverse, r)
    work = [[work[i][j] / weights[j] for j in range(r)] for i in range(n)]
    work = multiply(work, gram_inverse, r)
    return multiply(work, bt, n)


def determinant(a):
    n = len(a)
    a = matrix(a, n, n)
    result = Fraction(1)
    for j in range(n):
        pivot = next((i for i in range(j, n) if a[i][j]), None)
        if pivot is None:
            return Fraction(0)
        if pivot != j:
            a[j], a[pivot] = a[pivot], a[j]
            result = -result
        value = a[j][j]
        result *= value
        for i in range(j + 1, n):
            scale = a[i][j] / value
            for k in range(j + 1, n):
                a[i][k] -= scale * a[j][k]
    return result


def symmetric(a):
    if any(len(row) != len(a) for row in a):
        raise ValueError("square matrix required")
    if any(a[i][j] != a[j][i] for i in range(len(a)) for j in range(len(a))):
        raise ValueError("not symmetric")


def psd(a):
    """Exact PSD validation by all principal minors (dimension at most eight)."""
    symmetric(a)
    for size in range(1, len(a) + 1):
        for ix in combinations(range(len(a)), size):
            if determinant([[a[i][j] for j in ix] for i in ix]) < 0:
                raise ValueError("not positive semidefinite")


def positive_definite(a):
    symmetric(a)
    for size in range(1, len(a) + 1):
        if determinant([row[:size] for row in a[:size]]) <= 0:
            raise ValueError("not positive definite")
