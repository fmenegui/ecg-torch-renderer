"""Isolated CPU prototype. Production modules are not modified.

The wrapper snapshots the convex hull. Rebuild it if mapping.observed changes.
Each floating-point cross product has the baseline's operand order and dtype.
Only reduction of Boolean predicates is split across query/hull blocks.
"""
import cv2
import numpy as np


class BoundedEvidence:
    def __init__(self, mapping, pair_budget=131072):
        if int(pair_budget) != pair_budget or pair_budget < 1:
            raise ValueError('Positive integer pair budget required')
        self.mapping = mapping
        self.pair_budget = int(pair_budget)
        self.hull = cv2.convexHull(mapping.observed.astype(np.float32)).reshape(-1, 2)
        self.edges = np.roll(self.hull, -1, axis=0)-self.hull
        self.maximum_pairs_used = 0

    def _inside(self, points):
        n = len(points)
        inside = np.empty(n, bool)
        hull_block = min(len(self.hull), self.pair_budget)
        query_block = max(1, self.pair_budget//max(1, hull_block))
        for start in range(0, n, query_block):
            query = points[start:start+query_block]
            positive, negative = np.ones(len(query), bool), np.ones(len(query), bool)
            for first in range(0, len(self.hull), max(1, hull_block)):
                hull = self.hull[first:first+hull_block]
                edges = self.edges[first:first+hull_block]
                self.maximum_pairs_used = max(self.maximum_pairs_used, len(query)*len(hull))
                delta = query[:, None]-hull
                cross = edges[None, :, 0]*delta[..., 1]-edges[None, :, 1]*delta[..., 0]
                positive &= (cross >= -1e-5).all(1)
                negative &= (cross <= 1e-5).all(1)
            inside[start:start+len(query)] = positive | negative
        return inside

    def classify(self, points, *, radius=.75, mapped=None, return_eligibility=True):
        """Return provenance and optionally eligibility with one hull evaluation.

        No point-array cache persists between calls, so query mutation/reordering
        does not introduce stale classifications. Original CellMaps are used.
        """
        points = np.asarray(points)
        inside = self._inside(points)
        evidence = np.where(inside, 2, 1).astype(np.uint8)
        support = np.isfinite(self.mapping.observed_cells(points)).all(1)
        evidence[support] = 3
        if not return_eligibility:
            return evidence
        local = self.mapping.cell_coordinates(points)
        observed = np.isfinite(local).all(1)
        prediction = self.mapping.forward(points) if mapped is None else mapped
        agreement = np.linalg.norm(local-prediction, axis=1) <= .05
        eligible = observed & agreement
        if self.mapping.tree is not None:
            distance, nearest = self.mapping.tree.query(points)
            nearby = distance <= radius*self.mapping.local_spacing[nearest]
            eligible |= ~observed & nearby & (evidence == 2)
        return evidence, eligible & np.isfinite(prediction).all(1)

    def evidence(self, points):
        return self.classify(points, return_eligibility=False)

    def eligible_geometry(self, points, *, radius=.75, mapped=None):
        return self.classify(points, radius=radius, mapped=mapped)[1]
