import unittest

from reproductions.runpod.owned_training_status import (
    TRAINING_NAME,
    TRAINING_VOLUME,
    select_training_pod,
    verify_training_volume,
)


class OwnedTrainingStatusChecks(unittest.TestCase):
    def test_exact_execution_name_and_unique_identity_are_required(self):
        self.assertIsNone(select_training_pod([{'name': TRAINING_NAME + '-other', 'id': 'other'}]))
        self.assertEqual(select_training_pod([{'name': TRAINING_NAME, 'id': 'exact123'}]), 'exact123')
        self.assertEqual(select_training_pod([{'name': TRAINING_NAME, 'id': 'stopped123', 'desiredStatus': 'EXITED'}]), 'stopped123')
        for rows in [[{'name': TRAINING_NAME, 'id': 'a'}, {'name': TRAINING_NAME, 'id': 'b'}],
                     [{'name': TRAINING_NAME, 'id': '--another-resource'}]]:
            with self.assertRaises(ValueError):
                select_training_pod(rows)

    def test_volume_status_cannot_expand_beyond_the_verified_training_volume(self):
        original = {'id': TRAINING_VOLUME, 'name': TRAINING_NAME, 'size': 220, 'dataCenterId': 'US-KS-2'}
        self.assertEqual(verify_training_volume(original), original)
        for field, value in [('id', 'another'), ('name', TRAINING_NAME + '-other'), ('size', 50), ('dataCenterId', 'US-CA-1')]:
            with self.assertRaises(ValueError):
                verify_training_volume(dict(original, **{field: value}))


if __name__ == '__main__':
    unittest.main()
