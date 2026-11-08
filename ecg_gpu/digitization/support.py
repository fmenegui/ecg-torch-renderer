"""Image-only coverage diagnostics for explicitly partial lattice outputs.

The denominator is the observed relative lattice extent, never the full page.
These measurements do not certify grid identities or clinical usability.
"""
from collections import deque

import numpy as np


def cell_support_summary(topology):
    """Count four-connected supported cells without inventing missing cells."""
    ids = np.asarray(topology['ids'], dtype=int)
    cells = {tuple(map(int, cell)) for cell in topology['cells']}
    if len(ids) == 0:
        extent = (0, 0)
    elif ids.ndim != 2 or ids.shape[1] != 2:
        raise ValueError('Expected relative row/column identities')
    else:
        extent = tuple(int(x) for x in np.ptp(ids, axis=0))
    total = extent[0]*extent[1]
    remaining, sizes = cells.copy(), []
    while remaining:
        pending = deque([remaining.pop()])
        count = 0
        while pending:
            row, col = pending.popleft()
            count += 1
            for candidate in ((row-1, col), (row+1, col), (row, col-1), (row, col+1)):
                if candidate in remaining:
                    remaining.remove(candidate)
                    pending.append(candidate)
        sizes.append(count)
    largest = max(sizes, default=0)
    return {'observed_extent_cells': total, 'supported_cells': len(cells),
            'cell_coverage_fraction': len(cells)/total if total else 0.,
            'connected_components': len(sizes), 'largest_component_cells': largest,
            'largest_component_fraction': largest/len(cells) if cells else 0.,
            'largest_component_extent_fraction': largest/total if total else 0.,
            'denominator': 'bounding rectangle of observed relative grid identities',
            'full_page_coverage_measured': False,
            'grid_id_accuracy_measured': False}
