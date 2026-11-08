"""Experimental local support filter for a candidate relative grid.

This preserves explicitly supported cells after a global graph rejection.
It does not certify the remaining integer identities or global injectivity.
"""
import numpy as np


def locally_supported_cells(xy, topology):
    selected = np.asarray(topology['selected_indices'], int)
    ids = np.asarray(topology['ids'], int)
    if len(selected) < 4 or not topology.get('spacing_px'):
        return dict(topology, cells=[], partial_status='rejected', local_filter=True)
    xy = np.asarray(xy, float)[selected]
    lookup = {tuple(k): i for i, k in enumerate(ids)}
    theta = topology['orientation_radians']
    axes = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    spacing = np.asarray(topology['spacing_px'])
    cells = []
    for r, c in topology['cells']:
        quad = xy[[lookup[k] for k in ((r, c), (r, c+1), (r+1, c+1), (r+1, c))]]
        edges = (np.roll(quad, -1, axis=0)-quad)@axes.T
        good = True
        for i, (axis, sign) in enumerate(((0, 1), (1, 1), (0, -1), (1, -1))):
            along, across = sign*edges[i, axis], abs(edges[i, 1-axis])
            good &= .55*spacing[axis] < along < 1.5*spacing[axis] and across < .45*spacing[1-axis]
        if good:
            cells.append([int(r), int(c)])
    return dict(topology, cells=cells, partial_status='partial' if len(cells) >= 16 else 'rejected',
                native_status=topology['status'], local_filter=True,
                removed_cells=len(topology['cells'])-len(cells), global_injectivity_checked=False)
