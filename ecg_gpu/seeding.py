"""Stable offline seed derivation, independent of Python's salted hash()."""
import hashlib
import json


def render_seed(base_seed, source_id, *, view_id=0, stream_id=0):
    """Derive a replayable 63-bit seed outside the rendering hot path.

    Use patient/source identifiers without exporting them. Pass a distinct
    stream_id for independent workers/ranks, or keep it zero when a capture
    must be invariant to worker assignment and dataset ordering.
    """
    for name, value in (('base_seed', base_seed), ('view_id', view_id), ('stream_id', stream_id)):
        if type(value) is not int or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    if not isinstance(source_id, str) or not source_id:
        raise ValueError('source_id must be a nonempty string')
    payload = json.dumps([base_seed, source_id, view_id, stream_id], ensure_ascii=False,
                         separators=(',', ':')).encode('utf-8')
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8,
                          person=b'ecg-capture-v1').digest(), 'little') % (2**63-1)
