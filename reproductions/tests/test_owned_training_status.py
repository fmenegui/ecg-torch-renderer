import unittest

from reproductions.runpod.owned_training_status import (
    TRAINING_NAME,
    select_training_pod,
)


class OwnedTrainingStatusChecks(unittest.TestCase):
    def test_exact_execution_name_and_unique_identity_are_required(self):
        self.assertIsNone(select_training_pod([{'name': TRAINING_NAME + '-other', 'id': 'other'}]))
        self.assertEqual(select_training_pod([{'name': TRAINING_NAME, 'id': 'exact123'}]), 'exact123')
        for rows in [[{'name': TRAINING_NAME, 'id': 'a'}, {'name': TRAINING_NAME, 'id': 'b'}],
                     [{'name': TRAINING_NAME, 'id': '--another-resource'}]]:
            with self.assertRaises(ValueError):
                select_training_pod(rows)


if __name__ == '__main__':
    unittest.main()
