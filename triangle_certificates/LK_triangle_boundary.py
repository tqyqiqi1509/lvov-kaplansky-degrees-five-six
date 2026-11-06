"""Exact boundary checks for the triangle appendix, with python-flint."""

from itertools import combinations
from pathlib import Path
import json

from flint import fmpq, fmpq_mat, nmod_mat


def zero(n):
    return fmpq_mat(n, n)


def dense(rows):
    return fmpq_mat([[fmpq(str(x)) for x in row] for row in rows])


def unpack(n, sparse):
    result = zero(n)
    for i, j, x in sparse:
        result[i, j] = fmpq(str(x))
    return result


def commutator(a, b):
    return a*b - b*a


def alternating(a, b, c):
    return a*b*c - a*c*b - b*a*c + b*c*a + c*a*b - c*b*a


def coeffs(source, direction):
    x, y, z = source
    u, v, w = direction
    a, b, c = commutator(x, y), commutator(x, z), commutator(y, z)
    aa = commutator(u, y) + commutator(x, v)
    bb = commutator(u, z) + commutator(x, w)
    cc = commutator(v, z) + commutator(y, w)
    linear = alternating(aa, b, c) + alternating(a, bb, c) + alternating(a, b, cc)
    quadratic = (
        alternating(commutator(u, v), b, c)
        + alternating(a, commutator(u, w), c)
        + alternating(a, b, commutator(v, w))
        + alternating(aa, bb, c)
        + alternating(aa, b, cc)
        + alternating(a, bb, cc)
    )
    return linear, quadratic


def even_source(n):
    m = n // 2
    x, y, z = (zero(n) for _ in range(3))
    for i in range(m):
        x[i, m+i] = 1
        if i < m-1:
            x[m+i, i+1] = 1
            y[i+1, m+i] = z[i+1, m+i] = i+1
            z[m+i, i] = i+1
        if i:
            y[m+i, i] = i
    y[0, 2*m-1] = 1
    return [x, y, z]


def even_normals(source, n):
    m = n // 2
    a = commutator(*source[:2])
    c = commutator(*source[1:])
    aa = [a[m+j, m+j] for j in range(m)]
    cc = [c[m+(j+1)%m, m+j] for j in range(m)]
    n0, n1 = zero(n), zero(n)
    for j in range(m):
        k, l = (j+1)%m, (j+2)%m
        n0[m+k, m+j] = 1/(cc[j]*aa[j]*aa[k])
        n1[m+l, m+j] = 1/(cc[j]*cc[k]*(aa[j]+aa[l]-aa[k]))
    return [n0, n1]


def pair(a, b):
    return sum(x*y for x, y in zip(a.entries(), b.entries()))


def reduce_mod_101(x):
    return int(x.numerator) * pow(int(x.denominator), -1, 101) % 101


def pivot_columns(matrix):
    reduced, rank = matrix.rref()
    return [next(j for j in range(matrix.ncols()) if reduced[i, j])
            for i in range(rank)]


def odd_directions(n, columns, free_coordinates):
    derivative = fmpq_mat(columns).transpose()
    mod = nmod_mat([[reduce_mod_101(derivative[i, j])
                     for j in range(derivative.ncols())]
                    for i in range(derivative.nrows())], 101)
    pivot = pivot_columns(mod)
    rows = pivot_columns(mod.transpose())
    assert len(pivot) == len(rows) == n*n-2
    minor = fmpq_mat([[derivative[i, j] for j in pivot] for i in rows])
    assert reduce_mod_101(minor.det()) != 0
    inverse = minor.inv()
    directions = []
    for slot, i, j in free_coordinates:
        free = slot*n*n + i*n + j
        assert free not in pivot
        values = inverse * fmpq_mat([[-derivative[r, free]] for r in rows])
        flat = [fmpq(0)] * (3*n*n)
        flat[free] = 1
        for k, col in enumerate(pivot):
            flat[col] = values[k, 0]
        directions.append([fmpq_mat(n, n, flat[k*n*n:(k+1)*n*n]) for k in range(3)])
    return directions


def check(n, record):
    source = [dense(x) for x in record['source']] if 'source' in record else even_source(n)
    normals = (
        even_normals(source, n) if n%2 == 0
        else [unpack(n, normal) for normal in record['normals']]
    )
    assert fmpq_mat([nn.entries() for nn in normals]).rank() == 2
    value = alternating(commutator(*source[:2]),
                        commutator(source[0], source[2]),
                        commutator(*source[1:]))
    if 'value' in record:
        assert value == unpack(n, record['value'])
        assert value.rank() == 1 and value*value == zero(n)
    else:
        assert value == zero(n)

    columns = []
    for k in range(3):
        for i in range(n):
            for j in range(n):
                d = [zero(n) for _ in range(3)]
                d[k][i,j] = 1
                linear, _ = coeffs(source, d)
                assert all(pair(nn, linear) == 0 for nn in normals)
                columns.append(linear.entries())
    rank = fmpq_mat(columns).rank()
    if n not in (6, 8):
        assert rank == n*n-2

    if n % 2 == 0:
        directions = [
            [unpack(n, ((i,j,x) for w,i,j,x in d if w == k)) for k in range(3)]
            for d in record['directions']
        ]
    else:
        directions = odd_directions(n, columns, record['free_coordinates'])

    quads = []
    for d in directions:
        linear, q = coeffs(source, d)
        assert linear == zero(n)
        quads.append(q)
    size = len(directions)
    grams = [zero(size), zero(size)]
    for gram, normal in zip(grams, normals):
        for i, q in enumerate(quads):
            gram[i, i] = pair(normal, q)
    for i, j in combinations(range(size), 2):
        mixed = coeffs(source, [a+b for a,b in zip(directions[i], directions[j])])[1]
        mixed -= quads[i]+quads[j]
        for gram, normal in zip(grams, normals):
            gram[i,j] = gram[j,i] = pair(normal, mixed)/2

    if n%2 == 0:
        cross = fmpq_mat([[2*grams[0][i,2+j] for j in range(2)]for i in range(2)])
        assert cross == dense(record['cross'])
        assert cross.det() == fmpq(record['det'])
        assert all(grams[0][i,j] == grams[0][i+2,j+2] == 0
                   for i in range(2) for j in range(2))
        witness = 2 if n == 4 else 3
        assert fmpq(record['Q1']) == grams[1][witness, witness]
    else:
        modular_grams = [nmod_mat([[reduce_mod_101(q[i, j]) for j in range(size)]
                                  for i in range(size)], 101) for q in grams]
        assert modular_grams == [nmod_mat(q, 101) for q in record['gram_mod_101']]
        assert int(modular_grams[0].det()) == record['gram_determinant_mod_101']
    assert max(q.rank() for q in grams) >= 3
    assert fmpq_mat([q.entries() for q in grams]).rank() == 2
    print(n, 'Jacobian rank', rank, 'restricted Gram ranks', [q.rank() for q in grams])


if __name__ == '__main__':
    data = json.loads(Path(__file__).with_name('triangle_boundary_data.json').read_text())
    for n in range(4,10):
        check(n, data[str(n)])
