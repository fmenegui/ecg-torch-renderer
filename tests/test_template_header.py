"""Typography regressions found by reviewing enlarged rendered examples."""

import unittest

import torch

from ecg_gpu import ECGRenderer, RenderConfig, DocumentTemplate, DOCUMENT_TEMPLATES


class TemplateHeaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_units_and_template_text_have_disjoint_ink_rows(self):
        templates = list(DOCUMENT_TEMPLATES.values()) + [
            DocumentTemplate(
                "large-type",
                top_mm=34.0,
                font_mm=3.2,
                header_lines=(
                    "SYNTHETIC RECORD",
                    "SIMULATED SUBJECT",
                    "RESEARCH",
                    "NO DIAGNOSIS",
                ),
            )
        ]
        for width in (512, 1024):
            for template in templates:
                renderer = ECGRenderer(
                    RenderConfig(
                        width=width,
                        height=width * 3 // 4,
                        profile="clean",
                        surface_mode="off",
                    ),
                    document_template=template,
                )
                for duration in (5.0, 10.0, 20.0):
                    for font in range(renderer.font_count):
                        with self.subTest(
                            width=width,
                            template=template.name,
                            duration=duration,
                            font=font,
                        ):
                            p = renderer.sample_parameters(
                                1,
                                generator=torch.Generator().manual_seed(7),
                                photo=False,
                            )
                            p["font"].fill_(font)
                            p["speed_index"].fill_(1)
                            p["gain_index"].fill_(1)
                            p["header_present"].fill_(1)
                            p["label_present"].zero_()
                            instances = renderer._instances(p, torch.tensor([duration]))
                            p["font_size"] = instances["ppm"] * template.font_mm
                            text, _, units = renderer._text(
                                instances, p, need_labels=True
                            )
                            title = (text - units).clamp_min(0)
                            unit_rows = (units[0, 0] > 0.05).nonzero()[:, 0]
                            title_rows = (title[0, 0] > 0.05).nonzero()[:, 0]
                            self.assertGreater(len(unit_rows), 0)
                            self.assertGreater(len(title_rows), 0)
                            self.assertLess(int(unit_rows.max()), int(title_rows.min()))


if __name__ == "__main__":
    unittest.main()
