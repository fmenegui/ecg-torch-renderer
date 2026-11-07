"""Generated report text: header fields, measurements and filter footers.

Real ECG pages carry several lines of text: fictional-here patient and
administrative fields, a measurement block and a footer with filter and
calibration settings. The legacy renderer printed about forty fixed tokens.

Construction (CPU, once) renders a per-character glyph atlas and encodes a
bank of line templates. Each template mixes literal characters with slots
filled on device per record: random digits, and the ventricular rate digits,
which are estimated from the printed lead II signal itself. Footer lines are
chosen to agree with the record's paper speed, gain and, when device filters
are enabled, its filter settings. Everything per batch runs in torch.

All names and identifiers are fictional and generated from small word banks.
No interpretation statements are printed: the renderer never synthesizes
diagnoses.
"""
import math
import random

import torch
from torch import nn
from torch.nn import functional as F

from .device_filters import HIGHPASS_HZ, LOWPASS_HZ, NOTCH_HZ
from .schema_v05 import GAINS, SPEEDS

# Slot markers inside templates. Literal digits stay literal.
DIGIT, NONZERO, RATE_HUNDREDS, RATE_TENS, RATE_ONES = '\x01', '\x02', '\x03', '\x04', '\x05'
SLOT_KINDS = {DIGIT: 1, NONZERO: 2, RATE_HUNDREDS: 3, RATE_TENS: 4, RATE_ONES: 5}
CHARSET = ''.join(chr(code) for code in range(32, 127)) + 'ÁÂÃÀÇÉÊÍÓÔÕÚáâãàçéêíóôõúñÑ°µ'
MAX_LINE = 44
RATE = RATE_HUNDREDS + RATE_TENS + RATE_ONES
D, N = DIGIT, NONZERO

GIVEN = ('ANA', 'JOAO', 'MARIA', 'JOSE', 'LUCAS', 'JULIA', 'PEDRO', 'CLARA', 'RAFAEL', 'LAURA',
         'CARLOS', 'HELENA', 'DIEGO', 'SOFIA', 'MATEUS', 'ALICE', 'TOMAS', 'IRENE', 'SAMUEL', 'NORA')
FAMILY = ('SILVA', 'SOUZA', 'COSTA', 'PEREIRA', 'LIMA', 'ALVES', 'ROCHA', 'GOMES', 'MARTINS', 'BARROS',
          'GARCIA', 'LOPEZ', 'MORENO', 'TORRES', 'SMITH', 'BROWN', 'MILLER', 'WILSON', 'TAYLOR', 'CLARK')

HEADER_TEMPLATES = (
    'Name: {name}', 'Nome: {name}', 'Nombre: {name}', 'Patient: {name}', 'Paciente: {name}',
    f'ID: {N}{D * 7}', f'Prontuario: {N}{D * 6}', f'MRN {N}{D * 8}', f'Registro: {N}{D * 5}',
    f'DOB: {D}{D}/{D}{D}/19{D}{D}', f'Nasc.: {D}{D}/{D}{D}/19{D}{D}', f'Age: {N}{D} yr',
    f'Idade: {N}{D} anos', f'Edad: {N}{D} a', f'Date: {D}{D}/{D}{D}/20{D}{D} {D}{D}:{D}{D}',
    f'Data: {D}{D}/{D}{D}/20{D}{D} {D}{D}:{D}{D}:{D}{D}', f'Fecha: {D}{D}-{D}{D}-20{D}{D}',
    f'Room {N}{D}{D}  Bed {N}', f'Leito: {N}{D}', f'Loc: {N}{D}  Option: {N}{D}',
    'Referred by: Dr. {name}', 'Solicitante: Dr(a). {name}', f'Device: ECG-{N}{D}{D}{D} v{N}.{D}',
    f'Height: 1{D}{D} cm  Weight: {N}{D} kg', f'Altura: 1{D}{D} cm  Peso: {N}{D} kg',
)
MEASURE_TEMPLATES = (
    f'Vent. rate  {RATE} BPM', f'FC: {RATE} bpm', f'Frec. vent.: {RATE} lpm', f'HR {RATE}/min',
    f'PR int.    1{D}{D} ms', f'PR: 1{D}{D} ms', f'QRS dur.   {N}{D} ms', f'QRS: {N}{D} ms',
    f'QT/QTc  3{D}{D}/4{D}{D} ms', f'QT/QTcB: 3{D}{D}/4{D}{D} ms', f'P-R-T axes  {N}{D} {N}{D} {N}{D}',
    f'Eixos P-R-T: {N}{D} {N}{D} {N}{D}', f'RR: {N}{D}{D} ms',
)


def _format_hz(value):
    return f'{value:g}'


def footer_lines():
    """Footer strings indexed by (speed, gain, highpass, lowpass, notch)."""
    lines = []
    for speed in SPEEDS[1:]:
        for gain in GAINS[1:]:
            for high in HIGHPASS_HZ:
                for low in LOWPASS_HZ:
                    for notch in NOTCH_HZ:
                        mains = f'  {notch:g} Hz' if notch else ''
                        lines.append(f'{speed:g} mm/s  {gain:g} mm/mV  {_format_hz(high)}-{_format_hz(low)} Hz{mains}')
    return lines


def template_bank(seed=1973, names_per_template=12):
    """Return ``(header, measure, footer)`` lists of template strings."""
    rng = random.Random(seed)
    header = []
    for template in HEADER_TEMPLATES:
        if '{name}' in template:
            for _ in range(names_per_template):
                name = f'{rng.choice(FAMILY)}, {rng.choice(GIVEN)}'
                header.append(template.format(name=name))
        else:
            header.append(template)
    return header, list(MEASURE_TEMPLATES), footer_lines()


def encode(templates):
    """Encode templates as glyph ids and slot kinds, each ``[N, MAX_LINE]``."""
    glyphs, kinds = [], []
    for template in templates:
        if len(template) > MAX_LINE:
            raise ValueError(f'Template longer than {MAX_LINE} characters: {template!r}')
        ids, slots = [], []
        for char in template:
            kind = SLOT_KINDS.get(char, 0)
            if kind == 0 and char not in CHARSET:
                raise ValueError(f'Character {char!r} is not in the glyph set')
            ids.append(CHARSET.index('0') if kind else CHARSET.index(char))
            slots.append(kind)
        pad = MAX_LINE - len(ids)
        glyphs.append(ids + [0] * pad)
        kinds.append(slots + [-1] * pad)
    return glyphs, kinds


def glyph_atlas(font_names):
    """Per-font glyph atlas ``[F*G,1,40,W]`` and advances ``[F,G]`` at 24 px."""
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    fonts = []
    for name in font_names:
        try:
            fonts.append(ImageFont.truetype(str(name), 24))
        except OSError:
            fonts.append(ImageFont.load_default(size=24))
    width = math.ceil(max(font.getlength(char) for font in fonts for char in CHARSET)) + 4
    rows, advances = [], []
    for font in fonts:
        for char in CHARSET:
            image = Image.new('L', (width, 40), 0)
            ImageDraw.Draw(image).text((1, 1), char, font=font, fill=255, anchor='lt')
            rows.append(torch.from_numpy(np.asarray(image, dtype=np.float32).copy() / 255)[None])
        advances.append([font.getlength(char) for char in CHARSET])
    return torch.stack(rows), torch.tensor(advances, dtype=torch.float32)


def ventricular_rate(signals, rates, lengths, lead=1):
    """Beats per minute from the autocorrelation of one lead, ``[B]`` on device.

    Searches RR intervals of 0.33-1.5 s (40-180 bpm). Short or flat records
    fall back to the strongest lag in range.
    """
    b, _, n = signals.shape
    valid = torch.arange(n, device=signals.device)[None] < lengths[:, None]
    x = torch.where(valid, signals[:, lead], 0.)
    x = x - x.sum(-1, keepdim=True) / lengths[:, None].clamp_min(1)
    x = torch.where(valid, x, 0.)
    spectrum = torch.fft.rfft(x, n=2 * n, dim=-1)
    correlation = torch.fft.irfft(spectrum.abs().square(), n=2 * n, dim=-1)[:, :n]
    lag = torch.arange(n, device=signals.device, dtype=signals.dtype)[None]
    seconds = lag / rates[:, None]
    window = (seconds >= .33) & (seconds <= 1.5)
    best = torch.where(window, correlation, -torch.inf).argmax(-1).to(signals.dtype)
    return (60 * rates / best.clamp_min(1)).round().clamp(20, 250)


class ReportText(nn.Module):
    """Render generated report lines into a ``[B,1,H,W]`` ink mask on device."""

    rows, columns = 2, 3

    def __init__(self, font_names, height, width, font_limit, seed=1973):
        super().__init__()
        header, measure, footer = template_bank(seed)
        glyphs, kinds = encode(header + measure + footer)
        self.header_count, self.measure_count, self.footer_count = len(header), len(measure), len(footer)
        atlas, advances = glyph_atlas(font_names)
        self.register_buffer('atlas', atlas)
        self.register_buffer('advances', advances)
        self.register_buffer('glyphs', torch.tensor(glyphs, dtype=torch.long))
        self.register_buffer('kinds', torch.tensor(kinds, dtype=torch.long))
        self.glyph_count = len(CHARSET)
        self.height, self.width = height, width
        self.patch = (math.ceil(40 * font_limit / 24) + 2, math.ceil(atlas.shape[-1] * font_limit / 24) + 2)
        self.zero, self.space = CHARSET.index('0'), CHARSET.index(' ')
        self.lines = self.rows * self.columns

    def specification(self):
        return {'schema': 'ecg-report-text-v1', 'charset_size': self.glyph_count,
                'templates': {'header': self.header_count, 'measure': self.measure_count,
                              'footer': self.footer_count}, 'max_line': MAX_LINE,
                'rate_source': 'lead II autocorrelation, 40-180 bpm',
                'interpretation_statements': False}

    def sample(self, b, rand):
        """Per-record controls; ``rand(*shape)`` draws uniforms on the renderer device."""
        lines = torch.cat([
            (rand(b, self.lines // 2) * self.header_count).long().clamp(max=self.header_count - 1),
            self.header_count + (rand(b, self.lines - self.lines // 2) * self.measure_count).long()
            .clamp(max=self.measure_count - 1)], 1)
        return {'report_lines': lines,
                'report_present': (rand(b, self.lines) < .85).float(),
                'report_digits': (rand(b, self.lines, MAX_LINE) * 10).long().clamp(max=9),
                'report_footer_present': (rand(b) < .8).float(),
                'report_text_mm': 1.8 + rand(b) * .8}

    def _fill(self, line_ids, digits, rate):
        """Glyph ids ``[B,K,L]`` with digit and rate slots filled."""
        glyphs = self.glyphs[line_ids]
        kinds = self.kinds[line_ids]
        rate = rate.long()[:, None, None]
        hundreds, tens, ones = rate // 100, (rate // 10) % 10, rate % 10
        nonzero = 1 + digits % 9
        filled = torch.where(kinds == 1, self.zero + digits, glyphs)
        filled = torch.where(kinds == 2, self.zero + nonzero, filled)
        filled = torch.where(kinds == 3, torch.where(hundreds > 0, self.zero + hundreds, self.space), filled)
        filled = torch.where(kinds == 4, self.zero + tens, filled)
        filled = torch.where(kinds == 5, self.zero + ones, filled)
        return filled, kinds >= 0

    def footer_index(self, p, filters_enabled):
        if filters_enabled:
            def nearest(values, choices):
                bank = torch.tensor(choices, device=values.device, dtype=values.dtype)
                return (values[:, None] - bank[None]).abs().argmin(1)
            high = nearest(p['filter_highpass_hz'], HIGHPASS_HZ)
            low = nearest(p['filter_lowpass_hz'], LOWPASS_HZ)
            notch = nearest(p['filter_notch_hz'], NOTCH_HZ)
        else:
            # Without filtering the footer still states typical settings.
            high = (p['report_digits'][:, 0, 0] % 3)
            low = (p['report_digits'][:, 0, 1] % 3)
            notch = (p['report_digits'][:, 0, 2] % 3)
        speed, gain = p['speed_index'].long(), p['gain_index'].long()
        return self.header_count + self.measure_count + (((speed * 3 + gain) * 3 + high) * 3 + low) * 3 + notch

    def forward(self, p, instances, rate, header_bottom, filters_enabled=False):
        """Return ``(ink [B,1,H,W], metadata)``; ``rate`` is ``[B]`` bpm."""
        ppm = instances['ppm']
        size = (p['report_text_mm'] * ppm).clamp(max=self.patch[0] * 24 / 40 - 1)
        line_ids = torch.cat([p['report_lines'], self.footer_index(p, filters_enabled)[:, None]], 1)
        digits = torch.cat([p['report_digits'], p['report_digits'][:, :1]], 1)
        glyphs, used = self._fill(line_ids, digits, rate)
        font = p['font'].long()[:, None, None]
        advance = self.advances[font.expand_as(glyphs), glyphs] * (size / 24)[:, None, None]
        offset = advance.cumsum(-1) - advance
        left = instances['left']
        plot = (self.width - 1) - left - (instances['ppm'] * 7)
        slot = torch.arange(self.lines, device=rate.device)
        column = (slot % self.columns).float()
        row = (slot // self.columns).float()
        x0 = left[:, None] + column[None] * plot[:, None] / self.columns
        pitch = size * 1.35
        y0 = header_bottom[:, None] + row[None] * pitch[:, None]
        fits = (y0 + size[:, None] * 1.1) < (instances['top'][:, None] - .5 * ppm[:, None])
        column_end = x0 + plot[:, None] / self.columns - 2 * ppm[:, None]
        bottom = self.height - 1 - (ppm * 6).clamp(max=self.height * .07)
        footer_y = (bottom + (self.height - 1 - bottom - size) / 2)[:, None]
        footer_fits = (self.height - 1 - bottom) > size * 1.2
        x_lines = torch.cat([x0, left[:, None]], 1)
        y_lines = torch.cat([y0, footer_y], 1)
        line_present = torch.cat([p['report_present'] * fits, (p['report_footer_present'] * footer_fits)[:, None]], 1)
        end = torch.cat([column_end, (left + plot)[:, None]], 1)
        x = x_lines[:, :, None] + offset
        glyph_present = used & ((x + advance) <= end[:, :, None]) & (line_present[:, :, None] > 0)
        ink = self._splat(font * self.glyph_count + glyphs, x, y_lines[:, :, None].expand_as(x),
                          size, glyph_present.float())
        boxes = torch.stack([x_lines, y_lines, torch.where(glyph_present, x + advance, -torch.inf).amax(-1),
                             y_lines + size[:, None] * 1.1], -1)
        meta = {'schema': 'ecg-report-text-v1', 'glyphs': glyphs, 'glyph_present': glyph_present,
                'line_present': glyph_present.any(-1), 'boxes': boxes, 'ventricular_rate_bpm': rate,
                'charset': CHARSET}
        return ink, meta

    def _splat(self, tokens, x, y, size, present):
        b = tokens.shape[0]
        tokens, x, y, present = (value.reshape(b, -1) for value in (tokens, x, y, present))
        jobs = tokens.shape[1]
        ph, pw = self.patch
        device = tokens.device
        px = torch.arange(pw, device=device)[None, None, :].float()
        py = torch.arange(ph, device=device)[None, :, None].float()
        scale = size[:, None].expand(b, jobs).reshape(-1, 1, 1) / 24
        gx = 2 * ((px + .5) / scale) / self.atlas.shape[-1] - 1
        gy = 2 * ((py + .5) / scale) / self.atlas.shape[-2] - 1
        field = torch.stack([gx.expand(-1, ph, -1), gy.expand(-1, -1, pw)], -1)
        glyph = F.grid_sample(self.atlas[tokens.reshape(-1)], field, align_corners=False)[:, 0]
        glyph = glyph.reshape(b, jobs, ph, pw) * present[..., None, None]
        ix = (x[..., None, None] + torch.arange(pw, device=device)[None, None, None, :]).round().long()
        iy = (y[..., None, None] + torch.arange(ph, device=device)[None, None, :, None]).round().long()
        inside = (ix >= 0) & (ix < self.width) & (iy >= 0) & (iy < self.height)
        offsets = torch.arange(b, device=device)[:, None, None, None]
        index = (offsets * self.height + iy.clamp(0, self.height - 1)) * self.width + ix.clamp(0, self.width - 1)
        output = torch.zeros(b * self.height * self.width, device=device)
        output.scatter_reduce_(0, index.expand_as(glyph).reshape(-1), (glyph * inside).reshape(-1),
                               reduce='amax', include_self=True)
        return output.reshape(b, 1, self.height, self.width)


def decode_report_text(meta):
    """Host-side transcripts ``list[list[str]]`` from report metadata, outside training."""
    glyphs = meta['glyphs'].cpu().tolist()
    present = meta['glyph_present'].cpu().tolist()
    charset = meta['charset']
    return [[''.join(charset[g] for g, keep in zip(line, mask) if keep).rstrip()
             for line, mask in zip(record, record_mask)] for record, record_mask in zip(glyphs, present)]
