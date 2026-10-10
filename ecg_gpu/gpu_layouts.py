"""Declarative ECG layouts, including Cabrera and configurable rhythm strips."""
from dataclasses import dataclass

from .schema_v05 import LEADS


@dataclass(frozen=True)
class LayoutSpec:
    """Rows contain standard lead names or '-aVR' for an inverted lead.

    Rows may have different column counts. Leads absent from rows are not
    printed. Each lead occurs at most once in the short rows. Up to three
    full-duration rhythm strips follow them. Their lead IDs can be overridden
    per sample with parameters['rhythm_leads'], a [B,3] integer tensor.
    """
    name: str
    rows: tuple
    rhythm: tuple = ()

    def __post_init__(self):
        if not self.name or not self.rows or len(self.rhythm) > 3:
            raise ValueError('Named nonempty layout with at most three rhythm strips required')
        names = []
        for row in self.rows:
            if not row:
                raise ValueError('Empty layout row')
            for token in row:
                name = token.removeprefix('-')
                if name not in LEADS:
                    raise ValueError(f'Invalid lead: {token}')
                names.append(name)
        if len(set(names)) != len(names):
            raise ValueError('Short-row leads must be unique. Use rhythm for repetitions')
        if any(name not in LEADS for name in self.rhythm):
            raise ValueError('Invalid rhythm lead')


STANDARD = (('I','aVR','V1','V4'), ('II','aVL','V2','V5'), ('III','aVF','V3','V6'))
CABRERA_ORDER = ('aVL','I','-aVR','II','aVF','III','V1','V2','V3','V4','V5','V6')
CABRERA = (('aVL','II','V1','V4'), ('I','aVF','V2','V5'), ('-aVR','III','V3','V6'))
TWO_COLUMNS = tuple((a,b) for a,b in zip(LEADS[:6], LEADS[6:]))
TWO_ROWS = (('I','III','aVL','V1','V3','V5'), ('II','aVR','aVF','V2','V4','V6'))
FOUR_ROWS = tuple(tuple(LEADS[i:i+3]) for i in range(0,12,3))
RHYTHM = ('II','V1','V5')


def standard_layouts():
    # Preserve the original five indices to keep their meanings stable.
    specs = [LayoutSpec('3x4', STANDARD), LayoutSpec('6x2', TWO_COLUMNS),
             LayoutSpec('12x1', tuple((name,) for name in LEADS)),
             LayoutSpec('2x6', TWO_ROWS), LayoutSpec('3x4+r1', STANDARD, RHYTHM[:1])]
    specs += [LayoutSpec(f'3x4+r{n}', STANDARD, RHYTHM[:n]) for n in (2,3)]
    for name, rows in [('6x2',TWO_COLUMNS), ('12x1',tuple((x,) for x in LEADS)),
                       ('2x6',TWO_ROWS), ('4x3',FOUR_ROWS)]:
        if name == '4x3':
            specs.append(LayoutSpec(name,rows))
        specs += [LayoutSpec(f'{name}+r{n}',rows,RHYTHM[:n]) for n in (1,2,3)]
    specs += [LayoutSpec('3x4-cabrera',CABRERA)]
    specs += [LayoutSpec(f'3x4-cabrera+r{n}',CABRERA,RHYTHM[:n]) for n in (1,2,3)]
    specs += [LayoutSpec('6x2-cabrera',tuple(zip(CABRERA_ORDER[:6], CABRERA_ORDER[6:]))),
              LayoutSpec('12x1-cabrera',tuple((x,) for x in CABRERA_ORDER)),
              LayoutSpec('2x1-long',(('II',),('V1',))),
              LayoutSpec('3x1-long',(('II',),('V1',),('V5',))),
              LayoutSpec('1x1-long',(('II',),)), LayoutSpec('1x12',(tuple(LEADS),))]
    return specs
