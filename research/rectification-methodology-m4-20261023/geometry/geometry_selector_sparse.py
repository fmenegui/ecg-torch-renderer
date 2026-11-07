"""Sparse numerical realization of the frozen M4 residual equations.

The adapter temporarily replaces only stencil assembly and field solve inside
an otherwise identical selector. Use in one process at a time. Experimental,
not thread safe and not production code.
"""
from contextlib import contextmanager
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import geometry_selector as dense


def lattice_sparse(ids):
    ids = np.asarray(ids)
    if ids.ndim != 2 or ids.shape[1] != 2 or not np.isfinite(ids).all() or not np.equal(ids, np.round(ids)).all():
        raise ValueError('finite integer [N,2] IDs required')
    if len(np.unique(ids, axis=0)) != len(ids):
        raise ValueError('duplicate integer IDs')
    low, high = ids.min(0).astype(int), ids.max(0).astype(int)
    nr, nc = high-low+1
    if nr*nc > 20000 or min(nr, nc) < 2:
        raise ValueError('unsupported rectangular ID lattice extent')
    rr, cc = np.mgrid[low[0]:high[0]+1, low[1]:high[1]+1]
    all_ids = np.c_[rr.ravel(), cc.ravel()]
    index = (ids[:, 0].astype(int)-low[0])*nc+ids[:, 1].astype(int)-low[1]
    row_indices, col_indices, data = [], [], []
    row = 0
    def add(indices, weights):
        nonlocal row
        row_indices.extend([row]*len(indices))
        col_indices.extend(indices)
        data.extend(weights)
        row += 1
    for r in range(nr):
        for c in range(nc):
            i = r*nc+c
            if c+2 < nc:
                add([i, i+1, i+2], [1, -2, 1])
            if r+2 < nr:
                add([i, i+nc, i+2*nc], [1, -2, 1])
            if c+1 < nc and r+1 < nr:
                add([i, i+1, i+nc, i+nc+1], np.sqrt(2)*np.array([1, -1, -1, 1]))
    d = sparse.coo_matrix((data, (row_indices, col_indices)), shape=(row, nr*nc)).tocsr()
    return all_ids, index, (d.T@d).tocsr()


def fit_field_sparse(residual, selected_indices, n_vertices, penalty, strength):
    w = np.ones(len(selected_indices))
    value = np.zeros((n_vertices, 2))
    for _ in range(4):
        diagonal = np.full(n_vertices, 1e-8)
        diagonal[selected_indices] += w
        rhs = np.zeros_like(value)
        rhs[selected_indices] = residual*w[:, None]
        a = (strength*penalty+sparse.diags(diagonal)).tocsc()
        value = spsolve(a, rhs, permc_spec='MMD_AT_PLUS_A')
        error = np.linalg.norm(value[selected_indices]-residual, axis=1)
        w = np.minimum(1., 1./np.maximum(error, 1e-12))
    return value


@contextmanager
def sparse_backend():
    original_lattice, original_fit_field = dense.lattice, dense.fit_field
    try:
        dense.lattice, dense.fit_field = lattice_sparse, fit_field_sparse
        yield
    finally:
        dense.lattice, dense.fit_field = original_lattice, original_fit_field


def fit_select(observed, ids, allowed_cells, pitch_mm=5.):
    with sparse_backend():
        return dense.fit_select(observed, ids, allowed_cells, pitch_mm)
