# L'vov–Kaplansky conjecture in degrees five and six

Exact arithmetic checks for Qiyue Tang's manuscript of November 6, 2026.

## Run

Install the tested dependencies in a Python environment, then run the five checks from the repository root.

```sh
python -m pip install -r requirements.txt
python LK_pair_certificate.py
python triangle_certificates/LK_triangle_boundary.py
python triangle_certificates/audit_even_original_words.py
python triangle_certificates/independent_hessian_commutator.py
python triangle_certificates/independent_hessian_boundaries.py
```

Run with assertions enabled (the Python default). Each command exits with a nonzero status if a check fails. The scripts print their results and read only the two JSON inputs shipped here.

| Script | Checks |
| --- | --- |
| `LK_pair_certificate.py` | Word-coefficient ranks, trace-release enumeration, the order-three triangle arc, and the standard and repeated-input matrix certificates. |
| `LK_triangle_boundary.py` | Boundary orders four through nine, including the published odd-order Gram matrices modulo 101 and the even-order quadratic tests. |
| `audit_even_original_words.py` | Symbolic even-order quadratic coefficients and the finite overlap cases. |
| `independent_hessian_commutator.py` | Symbolic odd-order derivative and quadratic coefficients and the two determinant factorizations. |
| `independent_hessian_boundaries.py` | Odd-order overlap cases with block parameter five through thirteen. |

`triangle_boundary_data.json` supplies the boundary inputs and expected arithmetic. Its matrix coordinates are zero-based. `odd_compact_certificate.json` supplies the compact odd-order witness data and is shared by the two odd-order scripts. The manuscript gives the general-order arguments that use these calculations.
