"""Text evidence and independent integer rotations of incomplete grids."""

import unittest

import numpy as np

from ecg_gpu.digitization.cell_mesh import forward_cell_map
from ecg_gpu.digitization.page_orientation import (
    _text_evidence,
    choose_text_rotation,
    rotate_lattice_ids,
)


class PageOrientationTests(unittest.TestCase):
    def evidence(self, labels):
        tsv = "level\tconf\ttext\n"
        tsv += "".join("5\t90\t" + word + "\n" for word in labels)
        return _text_evidence(tsv)

    def test_readable_distinct_lead_labels_identify_rotation(self):
        candidates = [self.evidence([]) for _ in range(4)]
        candidates[3] = self.evidence(["V1", "V2", "aVR"])
        result = choose_text_rotation(candidates)
        self.assertEqual(result["quarter_turns_ccw"], 3)
        self.assertEqual(result["status"], "resolved-from-text")
        self.assertFalse(result["reference_used"])

    def test_one_repeated_label_does_not_create_redundant_evidence(self):
        candidates = [
            self.evidence(["V1"] * 40),
            *[self.evidence([]) for _ in range(3)],
        ]
        self.assertEqual(len(candidates[0]["strong_labels"]), 1)
        self.assertEqual(choose_text_rotation(candidates)["status"], "unresolved")

    def test_ambiguous_opposite_rotations_abstain(self):
        strong = self.evidence(["V1", "V2"])
        candidates = [strong, self.evidence([]), strong, self.evidence([])]
        result = choose_text_rotation(candidates)
        self.assertEqual(result["status"], "unresolved")
        self.assertFalse(result["semantic_orientation_inferred"])

    def test_missing_ocr_alternative_prevents_a_false_winner(self):
        candidates = [
            self.evidence(["V1", "V2"]),
            *[self.evidence([]) for _ in range(3)],
        ]
        candidates[1]["error"] = "timeout"
        result = choose_text_rotation(candidates)
        self.assertEqual(result["status"], "unresolved")
        self.assertEqual(result["reason"], "incomplete OCR execution")

    def test_low_confidence_noise_and_bad_confidence_values_do_not_vote(self):
        data = "level\tconf\ttext\n5\t20\tV1\n5\tbad\tV2\n5\t95\tWAVE\n"
        self.assertEqual(_text_evidence(data)["score"], 0.0)

    def test_all_rotations_preserve_exact_cells_and_do_not_close_a_hole(self):
        row, col = np.mgrid[:6, :8]
        ids = np.column_stack((row.ravel(), col.ravel()))
        ids = ids[(ids != [2, 3]).any(1)]
        present = {tuple(k) for k in ids}
        cells = [
            [r, c]
            for r, c in sorted(present)
            if all(k in present for k in ((r, c + 1), (r + 1, c + 1), (r + 1, c)))
        ]
        original = {
            "ids": ids.tolist(),
            "cells": cells,
            "selected_indices": list(range(len(ids))),
        }
        xy = ids[:, ::-1] * 10.0 + [20.0, 30.0]
        query = np.array([[1.2, 1.3], [3.0, 2.0]]) * 10.0 + [20.0, 30.0]
        rotation = np.array([[0.0, 1.0], [-1.0, 0.0]])
        for k in range(4):
            rotated = rotate_lattice_ids(original, k)
            changed_ids = np.asarray(rotated["ids"])
            mapped = forward_cell_map(
                xy, changed_ids[:, ::-1] * 5.0, changed_ids, query, rotated["cells"]
            )
            expected = (
                np.array([[1.2, 1.3]]) * 5.0 @ np.linalg.matrix_power(rotation, k).T
            )
            offset = (ids[:, ::-1] * 5.0 @ np.linalg.matrix_power(rotation, k).T).min(0)
            np.testing.assert_allclose(mapped[:1], expected - offset, atol=1e-10)
            self.assertTrue(np.isnan(mapped[1]).all())
            self.assertEqual(len(cells), len(rotated["cells"]))
            self.assertEqual(original["ids"], ids.tolist())


if __name__ == "__main__":
    unittest.main()
