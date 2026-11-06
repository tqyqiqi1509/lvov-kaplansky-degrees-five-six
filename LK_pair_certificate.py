"""Exact rank certificates for the two-slot lifting argument.

Requires python-flint. All linear algebra is over the rationals.
"""
from collections import Counter
from itertools import combinations, permutations, product

from flint import fmpq, fmpq_mat


def matrix_from_keys(keys, width):
    rows = {}
    for column, key in keys:
        if key is not None:
            rows.setdefault(key, [0] * width)[column] += 1
    return matrix(rows.values(), width)


def matrix(rows, width):
    rows = list(rows)
    return fmpq_mat(rows) if rows else fmpq_mat(0, width)


def kernel(a):
    reduced, rank = a.rref()
    pivots = [next(j for j in range(a.ncols()) if reduced[i, j])
              for i in range(rank)]
    free = [j for j in range(a.ncols()) if j not in pivots]
    basis = fmpq_mat(a.ncols(), len(free))
    for column, j in enumerate(free):
        basis[j, column] = 1
        for row, pivot in enumerate(pivots):
            basis[pivot, column] = -reduced[row, j]
    return basis


def reduced_rows(a):
    reduced, rank = a.rref()
    return reduced.tolist()[:rank]


def cyclic_key(word):
    zero = word.index(0)
    return word[zero:] + word[:zero]


def triangular_keys(words, degree):
    units = ((0, 0), (0, 1), (1, 1))
    for inputs in product(units, repeat=degree):
        for column, word in enumerate(words):
            path = [inputs[i] for i in word]
            if all(left[1] == right[0] for left, right in zip(path, path[1:])):
                yield column, (inputs, path[0][0], path[-1][1])


def detector_key(word, a, b):
    first = word.index(a)
    second = word.index(b)
    order = first < second
    lo, hi = sorted((first, second))
    return (order, tuple(sorted(word[:lo])), tuple(sorted(word[lo + 1:hi])),
            tuple(sorted(word[hi + 1:])))


def support_ranks(detectors, basis):
    width = basis.ncols()
    blocks = [reduced_rows(detector * basis) for detector in detectors]
    spaces = [[]]
    for mask in range(1, 1 << len(blocks)):
        bit = mask & -mask
        edge = bit.bit_length() - 1
        spaces.append(reduced_rows(matrix(spaces[mask ^ bit] + blocks[edge], width)))
    return [len(space) for space in spaces]


def check(degree, expected_dimensions):
    words = list(permutations(range(degree)))
    width = len(words)
    traces = [(i, ('trace', cyclic_key(w))) for i, w in enumerate(words)]
    identities = matrix_from_keys(traces + list(triangular_keys(words, degree)), width)
    deletion_keys = [(i, ('unit', label, tuple(x for x in w if x != label)))
                     for i, w in enumerate(words) for label in range(degree)]
    proper = matrix_from_keys(traces + deletion_keys, width)
    g_basis, h_basis = kernel(identities), kernel(proper)
    dims = (g_basis.ncols(), h_basis.ncols())
    assert dims == expected_dimensions
    detectors = [
        matrix_from_keys(((i, detector_key(w, a, b)) for i, w in enumerate(words)), width)
        for a, b in combinations(range(degree), 2)
    ]
    g_ranks = support_ranks(detectors, g_basis)
    h_ranks = support_ranks(detectors, h_basis)
    full_mask = len(g_ranks) - 1
    bad = [s for s in range(full_mask + 1)
           if g_ranks[full_mask ^ s] < dims[0] and h_ranks[s] < dims[1]]
    assert bad == []
    kept = {}
    for mask in range(full_mask + 1):
        if g_ranks[full_mask ^ mask] < dims[0]:
            kept.setdefault(mask.bit_count(), []).append(h_ranks[mask])
    profile = [(size, len(ranks), min(ranks)) for size, ranks in sorted(kept.items())]
    expected = {
        4: [(4, 3, 7), (5, 6, 7), (6, 1, 7)],
        5: [(4, 15, 38), (5, 90, 38), (6, 195, 38), (7, 120, 38),
            (8, 45, 38), (9, 10, 38), (10, 1, 38)],
    }
    assert profile == expected[degree]
    print(f'degree {degree}: dimensions {dims}; {full_mask + 1} subsets; {len(bad)} failures')


def proper_basis(words):
    degree = len(words[0])
    rows = ((i, (label, tuple(x for x in w if x != label)))
            for i, w in enumerate(words) for label in range(degree))
    return kernel(matrix_from_keys(rows, len(words)))


def cyclic_detector(words, free):
    def key(word):
        start = word.index(free[0])
        word = word[start:] + word[:start]
        groups = []
        for label in word:
            if label in free:
                groups.append((label, []))
            else:
                groups[-1][1].append(label)
        return tuple((label, tuple(sorted(frozen))) for label, frozen in groups)
    return matrix_from_keys(((i, key(w)) for i, w in enumerate(words)), len(words))


def check_quintic_traces():
    words = list(permutations(range(5)))
    basis = proper_basis(words)
    trace = matrix_from_keys(((i, cyclic_key(w)) for i, w in enumerate(words)), len(words))
    rows = []
    for triple in combinations(range(5), 3):
        labels = dict.fromkeys(triple, 0)
        labels.update(zip((j for j in range(5) if j not in triple), (1, 2)))
        def key(word):
            repeated = tuple(labels[x] for x in word)
            return min(repeated[j:] + repeated[:j] for j in range(5))
        block = matrix_from_keys(((i, key(w)) for i, w in enumerate(words)), len(words))
        rows.extend(reduced_rows(block * basis))
    trace_rows = reduced_rows(trace * basis)
    repeated = matrix(rows, basis.ncols())
    assert (basis.ncols(), len(trace_rows), repeated.rank()) == (44, 6, 6)
    assert matrix(rows + trace_rows, basis.ncols()).rank() == 6
    print('proper quintics: dimension 44, trace rank 6, triple trace rank 6')


def check_proper_sextics():
    words = list(permutations(range(6)))
    pairs = list(combinations(range(6), 2))
    triples = list(combinations(range(6), 3))
    basis = proper_basis(words)
    returns = []
    for pair in pairs:
        returns.extend(reduced_rows(cyclic_detector(words, pair) * basis))
    remaining = basis * kernel(matrix(returns, basis.ncols()))
    assert (basis.ncols(), remaining.ncols()) == (265, 256)
    width = remaining.ncols()
    path_rows = {}
    for a, b in pairs:
        block = matrix_from_keys(((i, detector_key(w, a, b))
                                  for i, w in enumerate(words)), len(words))
        path_rows[a, b] = reduced_rows(block * remaining)
    matching = {(0, 1), (2, 3), (4, 5)}
    all_paths = [row for rows in path_rows.values() for row in rows]
    outside_matching = [row for pair, rows in path_rows.items() if pair not in matching
                        for row in rows]
    assert matrix(all_paths, width).rank() == matrix(outside_matching, width).rank() == 166
    trace = matrix_from_keys(((i, cyclic_key(w)) for i, w in enumerate(words)), len(words))
    trace_rows = reduced_rows(trace * remaining)
    adjacent = []
    for triple in triples:
        if {0, 1} <= set(triple) or {0, 2} <= set(triple):
            adjacent.extend(reduced_rows(cyclic_detector(words, triple) * remaining))
    assert len(trace_rows) == matrix(adjacent, width).rank() == 29
    assert matrix(adjacent + trace_rows, width).rank() == 29
    print('proper sextics: dimensions 265 and 256, path rank 166, trace rank 29')


def triangle_polynomial(inputs, order):
    def multiply(left, right):
        result = [fmpq_mat(3, 3) for _ in range(order + 1)]
        for i, a in enumerate(left):
            for j, b in enumerate(right[:order + 1 - i]):
                result[i + j] += a * b
        return result

    def commutator(left, right):
        return [a - b for a, b in zip(multiply(left, right), multiply(right, left))]

    x, y, z = inputs
    arguments = (commutator(x, y), commutator(x, z), commutator(y, z))
    output = [fmpq_mat(3, 3) for _ in range(order + 1)]
    for perm in permutations(range(3)):
        parity = sum(a > b for a, b in combinations(perm, 2))
        first, second, third = (arguments[i] for i in perm)
        for i, coefficient in enumerate(multiply(multiply(first, second), third)):
            output[i] += (-1) ** parity * coefficient
    return output


def triple_at(vector):
    return [fmpq_mat(3, 3, vector[i:i + 9]) for i in (0, 9, 18)]


def check_triangle_order_three():
    sparse_arc = (
        ((1, '1'), (5, '1'), (12, '2'), (16, '1'), (25, '1')),
        ((1, '3'), (3, '2'), (4, '-1'), (5, '-2'), (6, '-1'), (7, '1'), (8, '2'), (9, '-3'), (10, '2'), (12, '-2'), (13, '-3'), (14, '2'), (15, '-1'), (16, '-1'), (17, '-2'), (19, '2'), (21, '-1'), (23, '2'), (24, '2'), (26, '1')),
        ((1, '59/4'), (2, '-3'), (9, '-3'), (10, '10'), (13, '5'), (19, '10')),
        ((1, '57'), (2, '-21'), (9, '-28'), (10, '97/2'), (11, '-6'), (13, '28'), (19, '97/2'), (20, '-6')),
        ((1, '4669/16'), (2, '-339/2'), (9, '-145'), (10, '207'), (11, '-54'), (13, '262'), (19, '206'), (20, '-54')),
        ((1, '26353/16'), (2, '-10115/8'), (9, '-650'), (10, '7745/8'), (11, '-442'), (13, '4099/2'), (19, '7609/8'), (20, '-442')),
        ((1, '647135/64'), (2, '-75179/8'), (9, '-2290'), (10, '20417/4'), (11, '-13635/4'), (13, '61877/4'), (19, '19259/4'), (20, '-13519/4')),
        ((1, '2015285/32'), (2, '-1093575/16'), (10, '971615/32'), (11, '-25790'), (13, '113675'), (19, '844955/32'), (20, '-25251')),
    )
    coefficients = []
    for row in sparse_arc:
        vector = [fmpq(0)] * 27
        for i, value in row:
            vector[i] = fmpq(value)
        coefficients.append(triple_at(vector))
    base = coefficients[0]
    result = triangle_polynomial(list(zip(*coefficients)), 7)
    expected = [fmpq_mat(3, 3) for _ in range(8)]
    expected[0][2, 0] = 4
    expected[7][0, 2] = 32
    assert result == expected

    def jet(vector, degree):
        direction = triple_at(vector)
        inputs = [(constant, linear) for constant, linear in zip(base, direction)]
        return triangle_polynomial(inputs, degree)[degree].entries()

    basis = [[int(i == j) for i in range(27)] for j in range(27)]
    derivative = fmpq_mat([jet(v, 1) for v in basis]).transpose()
    assert derivative.rank() == 7
    assert all(derivative[i, j] == 0 for i in (2, 4) for j in range(27))
    rows, cols = (0, 1, 3, 5, 6, 7, 8), (1, 2, 10, 11, 13, 19, 20)
    minor = fmpq_mat([[derivative[i, j] for j in cols] for i in rows])
    assert minor.det() != 0
    tangent = kernel(derivative).transpose().tolist()
    diagonal = [jet(v, 2)[4] for v in tangent]
    quadratic = fmpq_mat(20, 20)
    for i, value in enumerate(diagonal):
        quadratic[i, i] = value
    for i, j in combinations(range(20), 2):
        value = (jet([x + y for x, y in zip(tangent[i], tangent[j])], 2)[4]
                 - diagonal[i] - diagonal[j]) / 2
        quadratic[i, j] = quadratic[j, i] = value
    assert quadratic.rank() == 3
    print(f'triangle in order 3: rank 7, minor {minor.det()}, quadratic rank 3, arc order 7')


def matchings(labels):
    if not labels:
        yield ()
        return
    first, *others = labels
    for second in others:
        rest = [x for x in others if x != second]
        for matching in matchings(rest):
            yield ((first, second),) + matching


def product_of_commutators(pairs):
    for flips in product((0, 1), repeat=len(pairs)):
        yield tuple(x for pair, flip in zip(pairs, flips)
                    for x in (pair[::-1] if flip else pair)), (-1) ** sum(flips)


def restriction_map(words, labels, cyclic=False):
    def key(word):
        image = tuple(labels[x] for x in word)
        if cyclic:
            return min(image[i:] + image[:i] for i in range(len(image)))
        return image
    return matrix_from_keys(((i, key(w)) for i, w in enumerate(words)), len(words))


def role_map(groups):
    return {slot: label for label, group in enumerate(groups) for slot in group}


def standard_terms(labels):
    for w in permutations(labels):
        yield w, (-1) ** sum(a > b for a, b in combinations(w, 2))


def two_pair_pencils():
    pairs = dict(zip('ABCDEF', combinations(range(4), 2)))
    formulas = (
        '-AAF +AFA +BAE -BEA -CAD +CDA -DAC +DCA +EAB -EBA',
        '-ABE +ACD +ADC -AEB +AFA +BAE -CAD -DAC +EAB -FAA',
    )
    pencils = []
    for formula in formulas:
        polynomial = Counter()
        for term in formula.split():
            sign = 1 if term[0] == '+' else -1
            for w, c in product_of_commutators(tuple(pairs[x] for x in term[1:])):
                polynomial[w] += sign * c
        pencils.append(polynomial)
    anticommutator = Counter()
    for w, c in standard_terms(range(4)):
        for v, d in product_of_commutators(((0, 1),)):
            anticommutator[w + v] += c * d
            anticommutator[v + w] += c * d
    return pencils + [anticommutator]


def single_pair_pencils():
    right, left = Counter(), Counter()
    for label in range(1, 5):
        slots = tuple(i for i in range(5) if i != label)
        for w, c in standard_terms(slots):
            for v, d in product_of_commutators(((0, label),)):
                coefficient = (-1) ** (4 - label) * c * d
                right[w + v] += coefficient
                left[v + w] += coefficient
    return [right, left]


def verify_span(image, image_words, polynomials):
    columns = fmpq_mat([[p[w] for p in polynomials] for w in image_words])
    rank = len(polynomials)
    assert columns.rank() == rank
    joined = [list(a) + list(b) for a, b in zip(columns.tolist(), image.tolist())]
    assert fmpq_mat(joined).rank() == rank


def check_quartic_traces():
    labels = tuple(range(4))
    words = list(permutations(labels))
    indices = {w: i for i, w in enumerate(words)}
    basis = fmpq_mat(24, 6)
    orders = [pairs for matching in matchings(labels) for pairs in permutations(matching)]
    for j, pairs in enumerate(orders):
        for w, coeff in product_of_commutators(pairs):
            basis[indices[w], j] = coeff
    trace = matrix_from_keys(((i, cyclic_key(w)) for i, w in enumerate(words)), 24)
    traces = reduced_rows(trace * basis)
    repeated = []
    for pair in combinations(labels, 2):
        roles = role_map((pair,) + tuple((i,) for i in labels if i not in pair))
        repeated.extend(reduced_rows(restriction_map(words, roles, True) * basis))
    assert len(traces) == matrix(repeated, 6).rank() == matrix(traces + repeated, 6).rank() == 2
    print('quartic commutator products: trace rank 2, repeated trace rank 2')


def coordinate_unit(i, j):
    value = fmpq_mat(3, 3)
    value[i - 1, j - 1] = 1
    return value


def pivot_columns(reduced_rows):
    return [next(j for j, entry in enumerate(row) if entry) for row in reduced_rows]


def select_columns(a, columns):
    return fmpq_mat([[row[j] for j in columns] for row in a.tolist()])


def trace_release_blocks(orders, after_triples, labels):
    unit = coordinate_unit
    source = (unit(1, 2) + unit(2, 3),
              unit(2, 1) + 3 * unit(3, 2),
              2 * unit(2, 1) + 5 * unit(3, 2)
              + 7 * (unit(2, 2) + unit(3, 3)) + 11 * unit(2, 3))
    values = {}
    for word in product(range(3), repeat=5):
        value = source[word[0]]
        for letter in word[1:]:
            value = value * source[letter]
        values[word] = value.entries()

    # Compute on the ordered products of three commutators, then restrict
    # to the common kernel of the triple traces in the 90 product coordinates.
    expansions = [list(product_of_commutators(pairs)) for pairs in orders]
    coefficient_blocks = []
    trace_blocks = []
    for triple in combinations(labels, 3):
        rest = tuple(i for i in labels if i not in triple)
        for pair in combinations(rest, 2):
            single = tuple(i for i in rest if i not in pair)
            roles = role_map((triple, pair, single))
            coefficient = fmpq_mat(3, 90)
            for j, pairs in enumerate(orders):
                oriented = [(roles[x], roles[y]) for x, y in pairs]
                if any((a == 0) == (b == 0) for a, b in oriented):
                    continue
                sign = (-1) ** sum(a != 0 for a, _ in oriented)
                position = next(i for i, edge in enumerate(oriented) if 2 in edge)
                coefficient[position, j] = sign
            coefficient_blocks.append(coefficient * after_triples)

            release = fmpq_mat(27, 90)
            for j, expansion in enumerate(expansions):
                for k, slot in enumerate(triple):
                    for word, sign in expansion:
                        i = word.index(slot)
                        tail = word[i + 1:] + word[:i]
                        entries = values[tuple(roles[x] for x in tail)]
                        for r, entry in enumerate(entries):
                            release[9 * k + r, j] += sign * entry
            trace_blocks.append(release * after_triples)
    return coefficient_blocks, trace_blocks


def check_trace_release(orders, triple_kernel, labels, formal_trace):
    coeff, trace = trace_release_blocks(orders, triple_kernel, labels)
    k_rows = [row for b in coeff for row in b.tolist()]
    h_rows = [row for b in trace for row in b.tolist()]
    rank_k = reduced_rows(matrix(k_rows, 75))
    rank_h = reduced_rows(matrix(h_rows, 75))
    assert (len(rank_k), len(rank_h), formal_trace.rank()) == (32, 14, 14)
    assert matrix(rank_h + formal_trace.tolist(), 75).rank() == 14
    assert all(not any(sum(b[i, j] for i in range(3)) for j in range(75)) for b in coeff)
    k_cols = pivot_columns(rank_k)
    h_cols = pivot_columns(rank_h)
    k = [select_columns(b, k_cols).tolist() for b in coeff]
    h = [reduced_rows(select_columns(b, h_cols)) for b in trace]

    def contains(equations, row):
        residue = list(row)
        for equation in equations:
            pivot = next(i for i, value in enumerate(equation) if value)
            multiplier = residue[pivot]
            if multiplier:
                for j in range(pivot, 14):
                    residue[j] -= multiplier * equation[j]
        return not any(residue)

    visited = set()

    def visit(equations):
        key = tuple(tuple(str(entry) for entry in row) for row in equations)
        if key in visited:
            return
        visited.add(key)
        if len(equations) == 14:
            return
        active = [i for i, block in enumerate(h)
                  if not all(contains(equations, row) for row in block)]
        active_rows = [row for i in active for row in k[i]]
        dual_basis = reduced_rows(matrix(active_rows, 32).transpose())
        assert len(dual_basis) == 32
        assignments = dict.fromkeys(active[j // 3] for j in pivot_columns(dual_basis))
        for i in assignments:
            child = reduced_rows(matrix(equations + h[i], 14))
            assert len(child) > len(equations)
            visit(child)

    visit([])
    assert len(visited) == 669
    assert Counter(map(len, visited)) == {
        0: 1, 4: 16, 5: 6, 6: 6, 7: 32, 8: 72, 9: 81,
        10: 76, 11: 166, 12: 159, 13: 53, 14: 1,
    }
    print(f'trace-release compatibility: {len(visited)} rational row spaces')


def check_triple_commutators():
    labels = tuple(range(6))
    words = list(permutations(labels))
    indices = {w: i for i, w in enumerate(words)}
    basis = fmpq_mat(720, 90)
    orders = [pairs for matching in matchings(labels) for pairs in permutations(matching)]
    for j, pairs in enumerate(orders):
        for w, coeff in product_of_commutators(pairs):
            basis[indices[w], j] = coeff
    assert basis.rank() == 90

    trace = matrix_from_keys(((i, cyclic_key(w)) for i, w in enumerate(words)), 720)
    trace_zero = basis * kernel(trace * basis)
    assert trace_zero.ncols() == 61
    triple_rows = []
    for triple in combinations(labels, 3):
        other = tuple((i,) for i in labels if i not in triple)
        triple_rows.extend(reduced_rows(
            restriction_map(words, role_map((triple,) + other), True) * basis))
    triple_kernel = kernel(matrix(triple_rows, 90))
    after_triples = basis * triple_kernel
    assert after_triples.ncols() == 75
    check_trace_release(orders, triple_kernel, labels, trace * after_triples)

    def shape_321_rows(space):
        rows = []
        for triple in combinations(labels, 3):
            rest = tuple(i for i in labels if i not in triple)
            for pair in combinations(rest, 2):
                single = tuple(i for i in rest if i not in pair)
                image = restriction_map(words, role_map((triple, pair, single))) * space
                rows.extend(reduced_rows(image))
        return matrix(rows, space.ncols())

    after_321 = after_triples * kernel(shape_321_rows(after_triples))
    trace_zero_after_321 = trace_zero * kernel(shape_321_rows(trace_zero))
    assert (after_321.ncols(), trace_zero_after_321.ncols()) == (43, 29)

    def triangle(word):
        a, b, c = ((0, 1), (0, 2), (1, 2))
        return sum(sign * (-1) ** sum(x > y for x, y in combinations(order, 2))
                   for order in permutations((a, b, c))
                   for w, sign in product_of_commutators(order) if w == word)

    rows = []
    for matching in matchings(labels):
        roles = role_map(matching)
        image = restriction_map(words, roles) * after_321
        # The row order of restriction_map is first occurrence, not lexicographic order.
        repeated_words = list(dict.fromkeys(tuple(roles[x] for x in w) for w in words))
        alternator = [triangle(w) for w in repeated_words]
        i = next(i for i, x in enumerate(alternator) if x)
        scalar = [image[i, j] / alternator[i] for j in range(image.ncols())]
        assert all(image[k, j] == x * scalar[j] for k, x in enumerate(alternator)
                   for j in range(image.ncols()))
        rows.append(scalar)
    final = after_321 * kernel(matrix(rows, 43))
    assert final.ncols() == 38
    assert (trace * final).rank() == 9

    traceful_2211_rows, traceless_2211_rows = [], []
    two_pair = two_pair_pencils()
    for first in combinations(labels, 2):
        for second in combinations((i for i in labels if i not in first), 2):
            others = tuple((i,) for i in labels if i not in first + second)
            roles = role_map((first, second) + others)
            traceful_2211_rows.extend(reduced_rows(restriction_map(words, roles, True) * final))
            image_words = list(dict.fromkeys(tuple(roles[x] for x in w) for w in words))
            actual = restriction_map(words, roles)
            verify_span(actual * final, image_words, two_pair)
            image = actual * trace_zero_after_321
            verify_span(image, image_words, two_pair[:2])
            traceless_2211_rows.extend(reduced_rows(image))
    assert matrix(traceful_2211_rows, 38).rank() == 9
    after_2211 = trace_zero_after_321 * kernel(matrix(traceless_2211_rows, 29))
    assert after_2211.ncols() == 11
    single_pair_rows = []
    single_pair = single_pair_pencils()
    for pair in combinations(labels, 2):
        roles = role_map((pair,) + tuple((i,) for i in labels if i not in pair))
        image_words = list(dict.fromkeys(tuple(roles[x] for x in w) for w in words))
        image = restriction_map(words, roles) * after_2211
        verify_span(image, image_words, single_pair)
        single_pair_rows.extend(reduced_rows(image))
    assert matrix(single_pair_rows, 11).rank() == 10
    print('triple commutators: 90 -> 75 -> 43 -> 38; trace zero: 61 -> 29 -> 11 -> 1')


def check_standard_six_order_four():
    def unit(i, j):
        result = fmpq_mat(4, 4)
        result[i - 1, j - 1] = 1
        return result

    def standard(inputs):
        result = fmpq_mat(4, 4)
        for word, sign in terms:
            term = inputs[word[0]]
            for index in word[1:]:
                term = term * inputs[index]
            result += sign * term
        return result

    terms = list(standard_terms(range(6)))
    inputs = [3 * unit(1, 1) - unit(2, 2) - unit(3, 3) - unit(4, 4),
              unit(1, 2), unit(1, 3), unit(1, 4), unit(3, 1), unit(4, 1)]
    assert standard(inputs) == -24 * unit(1, 2)
    columns = []
    for slot in range(6):
        for i, j in product(range(1, 5), repeat=2):
            inputs_with_variation = inputs.copy()
            inputs_with_variation[slot] = unit(i, j)
            columns.append(standard(inputs_with_variation).entries())
    jacobian = fmpq_mat(columns).transpose()
    selected = (0, 4, 20, 22, 23, 25, 29, 36, 38, 39, 52, 54, 55, 68, 84)
    minor = fmpq_mat([[jacobian[i, j] for j in selected] for i in range(15)])
    assert minor.det() == -4809263859892224
    assert jacobian.rank() == 15
    print('standard polynomial in degree 6, order 4: value -24 E12, rank 15')


def polynomial_value(polynomial, inputs):
    n = inputs[0].nrows()
    value = fmpq_mat(n, n)
    for word, coefficient in polynomial.items():
        if coefficient:
            term = inputs[word[0]]
            for label in word[1:]:
                term = term * inputs[label]
            value += coefficient * term
    return value


def polynomial_direction(polynomial, inputs, slot, direction):
    n = inputs[0].nrows()
    value = fmpq_mat(n, n)
    for word, coefficient in polynomial.items():
        if coefficient:
            for position, label in enumerate(word):
                if label != slot:
                    continue
                term = direction if position == 0 else inputs[word[0]]
                for j, letter in enumerate(word[1:], 1):
                    term = term * (direction if j == position else inputs[letter])
                value += coefficient * term
    return value


def check_exceptional_two_pair():
    g0, g1, anticommutator = two_pair_pencils()
    polynomial = Counter()
    for f, scale in ((g0, 1), (g1, 1), (anticommutator, -3)):
        for word, coefficient in f.items():
            polynomial[word] += scale * coefficient

    e = coordinate_unit
    inputs = (e(1, 2) + e(2, 3), 2 * e(2, 1) + e(3, 2),
              e(2, 2) + 3 * e(3, 3), e(1, 2))
    assert polynomial_value(polynomial, inputs) == -24 * e(2, 3)
    directions = [(0, i, j) for i, j in ((1, 1), (1, 2), (1, 3),
                  (2, 1), (2, 2), (2, 3), (3, 1))]
    directions += [(1, 3, 1), (2, 3, 1)]
    columns = [polynomial_direction(polynomial, inputs, slot, e(i, j)).entries()
               for slot, i, j in directions]
    jacobian = fmpq_mat(columns).transpose()
    expected = fmpq_mat([
        [4, 0, 0, 0, 0, 0, 0, -10, 0],
        [0, -4, 0, 0, 0, 4, 0, 0, 0],
        [0, 0, -32, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 4, 0, 0, 0, 0, -12],
        [0, 0, 0, 0, -12, 0, 0, -24, 0],
        [0, -32, 0, 0, 0, -16, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 36, 0, 0],
        [0, 0, 0, -8, 0, 0, 0, 0, 0],
        [8, 0, 0, 0, -12, 0, 0, -20, 0],
    ])
    assert jacobian == expected
    assert jacobian.det() == 24461180928
    print('exceptional two-pair polynomial: value -24 E23, minor 24461180928')


def mixed_standard_terms():
    right, left = Counter(), Counter()
    for w, c in standard_terms(range(4)):
        for v, d in product_of_commutators(((0, 1),)):
            right[w + v] += c * d
            left[v + w] += c * d
    return right, left


def check_mixed_standard_order_three():
    right, left = mixed_standard_terms()
    e = coordinate_unit
    x = e(1, 3) + e(2, 1) + e(3, 2)
    z = e(1, 3)
    selected = (0, 6, 9, 12, 13, 15, 18, 19, 24)
    for r, first in ((3, 25), (4, 18)):
        inputs = (x, e(1, 2) + r * e(2, 3), z,
                  fmpq_mat([[first, 3, 4], [6, 1, 4], [0, 0, 6]]))
        assert polynomial_value(right, inputs) == r * e(1, 3)
        assert polynomial_value(left, inputs) == -e(1, 3)
        directions = [(j // 9, (j % 9) // 3 + 1, j % 3 + 1) for j in selected]
        matrix_r = fmpq_mat([polynomial_direction(right, inputs, j, e(i, k)).entries()
                             for j, i, k in directions]).transpose()
        matrix_l = fmpq_mat([polynomial_direction(left, inputs, j, e(i, k)).entries()
                             for j, i, k in directions]).transpose()
        # A determinant of a 9 by 9 affine pencil has degree at most nine.
        # Agreement at ten rational parameters certifies the factorization.
        for t in range(10):
            if r == 3:
                expected = 15552 * (t - 3)**3 * (t + 1)**2 * (2*t + 1) * (3*t - 2) * (3*t - 1)
            else:
                expected = 31104 * (t - 4)**3 * (t + 1)**2 * (3*t + 1) * (4*t - 3) * (4*t - 1)
            assert (matrix_r + t * matrix_l).det() == expected

    bracket = right.copy()
    for w, c in left.items():
        bracket[w] -= c
    inputs = (e(1, 2) + e(2, 3), 2 * e(2, 1) + e(3, 2), e(3, 3), e(1, 2))
    assert polynomial_value(bracket, inputs) == -3 * e(1, 2)
    directions = [(0, i, j) for i, j in ((1, 1), (1, 2), (1, 3),
                  (2, 1), (2, 2), (2, 3), (3, 1))] + [(3, 2, 1)]
    columns = [polynomial_direction(bracket, inputs, j, e(i, k)).entries()[:8]
               for j, i, k in directions]
    assert fmpq_mat(columns).transpose().det() == 10368
    print('mixed standard polynomial in order 3: both affine determinant factorizations and commutator minor')


def check_mixed_standard_order_four():
    from flint import nmod_mat

    def e(i, j):
        result = nmod_mat(4, 4, 101)
        result[i, j] = 1
        return result

    x = sum((e((i + 1) % 4, i) for i in range(4)), nmod_mat(4, 4, 101))
    diagonal = sum((pow(10, i, 101) * e(i, i) for i in range(4)), nmod_mat(4, 4, 101))
    y = x**3 * diagonal
    z = diagonal**2
    t = (-10 * pow(8, -1, 101)) * e(0, 2)
    inputs = (x, y, z, t)
    right, _ = mixed_standard_terms()

    def direction(polynomial, slot, variation):
        result = nmod_mat(4, 4, 101)
        for word, coefficient in polynomial.items():
            for position, label in enumerate(word):
                if label == slot:
                    term = variation if position == 0 else inputs[word[0]]
                    for j, letter in enumerate(word[1:], 1):
                        term = term * (variation if position == j else inputs[letter])
                    result += coefficient * term
        return result.entries()

    commutator = {(0, 1): 1, (1, 0): -1}
    rows = list(range(24)) + list(range(25, 31))
    cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 16, 18, 19, 21, 23, 24,
            32, 36, 37, 39, 40, 42, 43, 45, 49, 51, 56]
    matrix_cols = []
    for col in cols:
        slot, ij = divmod(col, 16)
        i, j = divmod(ij, 4)
        full_col = direction(right, slot, e(i, j)) + direction(commutator, slot, e(i, j))
        matrix_cols.append([full_col[row] for row in rows])
    assert nmod_mat(matrix_cols, 101).transpose().det() == 59
    print('mixed standard joint map in order 4: minor 59 modulo 101')


def check_hook_order_three():
    right, left = single_pair_pencils()
    e = coordinate_unit
    inputs = (e(1, 1) + e(2, 3), e(1, 1) + e(2, 2),
              e(1, 2), e(2, 1), e(1, 3) + e(2, 3))
    assert polynomial_value(right, inputs) == fmpq_mat(3, 3)
    assert polynomial_value(left, inputs) == e(1, 3) - e(2, 3)
    directions = [(slot, e(3, col)) for slot in range(5) for col in (1, 2)]
    mr = [polynomial_direction(right, inputs, j, delta).entries()[:8]
          for j, delta in directions]
    ml = [polynomial_direction(left, inputs, j, delta).entries()[:8]
          for j, delta in directions]
    columns = ((0, 1, 2, 3, 4, 5, 8, 9), (0, 1, 2, 3, 4, 6, 8, 9),
               (0, 1, 2, 3, 4, 7, 8, 9), (0, 1, 2, 3, 5, 6, 8, 9))
    for t in range(9):
        expected = (12*t*(1-t)*(9-2*t+2*t*t),
                    -2*t*(1-t)*(32+37*t-11*t*t),
                    4*(1-t)*(12-25*t+8*t*t),
                    2*t*t*(80-69*t-30*t*t+18*t**3+5*t**4))
        for selection, value in zip(columns, expected):
            a = fmpq_mat([mr[i] for i in selection])
            b = fmpq_mat([ml[i] for i in selection])
            assert (a + t*b).det() == value
    resultant = fmpq_mat([[2, -2, 9, 0], [0, 2, -2, 9],
                          [-11, 37, 32, 0], [0, -11, 37, 32]]).det()
    assert resultant == 47213
    print('hook polynomial in order 3: four determinant factorizations, resultant 47213')


if __name__ == '__main__':
    check(4, (4, 7))
    check(5, (53, 38))
    check_quartic_traces()
    check_quintic_traces()
    check_proper_sextics()
    check_triangle_order_three()
    check_triple_commutators()
    check_standard_six_order_four()
    check_exceptional_two_pair()
    check_mixed_standard_order_three()
    check_mixed_standard_order_four()
    check_hook_order_three()
