"""Build completed M6 tables, figures and a self-contained Portuguese report."""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
NAMES = {
    "baseline": "Atual B0",
    "baseline-text": "Atual + OCR",
    "consensus": "Consenso",
    "consensus-text": "Consenso + OCR",
    "M2": "M2 anterior",
}
ORDER = ("baseline", "baseline-text", "consensus", "consensus-text")
COLORS = {"good": "#16866f", "wrong": "#c95a44", "missing": "#e3e8ed", "ink": "#21364b"}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def percent(value):
    return f"{100 * value:.2f}%".replace(".", ",")


def bars(ax, names, metrics):
    y = np.arange(len(names))
    good = np.array([r["g"] for r in metrics]) * 100
    wrong = np.array([r["w"] for r in metrics]) * 100
    ax.barh(y, good, color=COLORS["good"], label="Correta")
    ax.barh(y, wrong, left=good, color=COLORS["wrong"], label="Errada")
    ax.barh(
        y,
        100 - good - wrong,
        left=good + wrong,
        color=COLORS["missing"],
        label="Não retornada",
    )
    for row, g, m in zip(y, good, metrics):
        ax.text(
            101.5,
            row,
            f"G {percent(m['g'])}   R {percent(m['r']) if m['r'] is not None else 'indef.'}",
            va="center",
            fontsize=9,
            color=COLORS["ink"],
        )
        if g > 12:
            ax.text(
                g / 2,
                row,
                percent(g / 100),
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 146)
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_bounds(0, 100)
    ax.tick_params(axis="y", length=0)


def save_figure(fig, name):
    fig.savefig(HERE / (name + ".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(HERE / (name + ".png"), bbox_inches="tight", facecolor="white", dpi=160)
    plt.close(fig)


def balanced(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["group_id"], row["backend"]), []).append(row)
    g, w, c = np.mean(
        [
            [np.mean([r[k] for r in group]) for k in ("g", "w", "c")]
            for group in groups.values()
        ],
        axis=0,
    )
    return {
        "g": float(g),
        "w": float(w),
        "c": float(c),
        "r": float(w / c) if c else None,
    }


def build(work, comparison, bank, report):
    validation, test = work / "validation-final", work / "test-final-cached"
    vs, ts = read(validation / "summary.json"), read(test / "summary.json")
    tr = read(test / "cases.json")
    vp, tp = read(validation / "protocol.json"), read(test / "protocol.json")
    assert len(vp["cases"]) == 24 and len(tp["cases"]) == 32
    assert not (
        {r["group_id"] for r in vp["cases"]} & {r["group_id"] for r in tp["cases"]}
    )
    va, ta = (
        read(work / "validation-independent-audit.json"),
        read(work / "test-independent-audit.json"),
    )
    assert va["passed"] and ta["passed"]
    assert all(r["coordinates_and_eligibility_exact"] for r in ta["baseline_replay"])
    for source, prefix in ((validation, "validation"), (test, "test")):
        for name in ("protocol.json", "summary.json", "cases.csv", "details.json"):
            shutil.copy2(source / name, HERE / (prefix + "-" + name))
    for name in ("validation-independent-audit.json", "test-independent-audit.json"):
        shutil.copy2(work / name, HERE / name)
    legacy = []
    for case in tp["cases"]:
        old = read(comparison / "cases" / case["key"] / "our-M2/result.json")
        legacy.append(
            {
                **case,
                **{
                    k: old["scores"]["primary"]["domains"]["paper"][k]
                    for k in ("g", "w", "c", "r")
                },
            }
        )
    legacy_mean = balanced(legacy)
    write(
        HERE / "legacy-M2-reference.json",
        {
            "source": "unchanged frozen M3 our-M2 results",
            "means": legacy_mean,
            "rows": legacy,
        },
    )
    vm = vs["primary"]["paper"]["means"]
    tm = ts["primary"]["paper"]["means"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.6))
    bars(
        axes[0], [NAMES[m] for m in ORDER], [vm[m + "-cells"]["overall"] for m in ORDER]
    )
    axes[0].set_title(
        "Validação: 24 capturas, 6 pacientes", loc="left", fontweight="bold", pad=14
    )
    bars(
        axes[1], [NAMES[m] for m in ORDER], [tm[m + "-cells"]["overall"] for m in ORDER]
    )
    axes[1].set_title(
        "Avaliação retrospectiva: 32 capturas, 8 outros pacientes",
        loc="left",
        fontweight="bold",
        pad=14,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.46, 0.045),
        frameon=False,
    )
    fig.text(
        0.04,
        0.02,
        "Células observadas, mesmas detecções. G: fração correta até 0,5 mm. R: erradas entre as retornadas. Origem ajustada por âncoras.",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.19, right=0.99, top=0.94, bottom=0.17, hspace=0.65)
    save_figure(fig, "partial-coordinates")
    full_names = [
        "baseline-full",
        "M2",
        "baseline-text-full",
        "consensus-full",
        "consensus-text-full",
    ]
    full_metrics = [legacy_mean if m == "M2" else tm[m]["overall"] for m in full_names]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    labels = [NAMES[m.replace("-full", "")] for m in full_names]
    bars(ax, labels, full_metrics)
    ax.set_title(
        "Pipeline com saída completa ou preservação do original",
        loc="left",
        fontweight="bold",
        pad=16,
    )
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, -0.33), frameon=False)
    fig.text(
        0.04,
        0.025,
        "32 capturas sintéticas. Elegibilidade geométrica medida nas mesmas consultas. M2 é a referência histórica com refinamento opcional.",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.19, right=0.99, top=0.85, bottom=0.27)
    save_figure(fig, "full-pipeline")
    selected = [
        r
        for r in tr
        if r["domain"] == "paper"
        and r["registration"] == "primary"
        and r["method"]
        in ("baseline-full", "baseline-text-full", "consensus-text-full")
    ]
    pairs = {}
    for row in selected:
        pairs.setdefault(row["key"], {})[row["method"]] = row
    keys = [case["key"] for case in tp["cases"]]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    delta = (
        np.array(
            [
                pairs[k]["consensus-text-full"]["g"] - pairs[k]["baseline-full"]["g"]
                for k in keys
            ]
        )
        * 100
    )
    ax.bar(
        np.arange(len(keys)),
        delta,
        color=[COLORS["good"] if d >= 0 else COLORS["wrong"] for d in delta],
    )
    ax.axhline(0, color="#70808f", lw=0.8)
    ax.set_xticks(np.arange(len(keys)), keys, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("Variação de G em pontos percentuais")
    ax.set_title(
        "Consenso + OCR: todos os 32 casos, incluindo perdas",
        loc="left",
        fontweight="bold",
        pad=15,
    )
    fig.subplots_adjust(bottom=0.28, left=0.1, right=0.98, top=0.86)
    save_figure(fig, "case-differences")
    ocr_deltas = [
        pairs[k]["baseline-text-full"]["g"] - pairs[k]["baseline-full"]["g"]
        for k in keys
    ]
    ocr_case_changes = {
        "improved": sum(d > 0 for d in ocr_deltas),
        "worse": sum(d < 0 for d in ocr_deltas),
        "tied": sum(d == 0 for d in ocr_deltas),
    }
    details = read(test / "details.json")
    counts = {
        m: sum(d["resampled"] for d in details if d["method"] == m)
        for m in tm
        if m.endswith("-full")
    }
    availability = ", ".join(
        NAMES[m.replace("-full", "")] + ": " + str(n) + "/32" for m, n in counts.items()
    )
    ocr = {
        k: read(work / "orientation-test-final" / k / "orientation.json") for k in keys
    }
    ocr_summary = {
        "test_cases": 32,
        "statuses": dict(Counter(v["status"] for v in ocr.values())),
        "incomplete_OCR_cases": [
            k for k, v in ocr.items() if any(c.get("error") for c in v["candidates"])
        ],
        "latency_seconds": {
            "median": float(np.median([v["elapsed_seconds"] for v in ocr.values()])),
            "max": float(max(v["elapsed_seconds"] for v in ocr.values())),
        },
        "receipts": ocr,
    }
    write(HERE / "orientation-summary.json", ocr_summary)
    # Explicit post-hoc examples, never an additional estimate of performance.
    resampled = {
        d["key"]
        for d in details
        if d["method"] == "consensus-text-full" and d["resampled"]
    }
    eligible = [k for k in keys if k in resampled]
    best = max(
        eligible or keys,
        key=lambda k: (
            pairs[k]["consensus-text-full"]["g"] - pairs[k]["baseline-full"]["g"]
        ),
    )
    worst = min(
        keys,
        key=lambda k: (
            pairs[k]["consensus-text-full"]["g"] - pairs[k]["baseline-full"]["g"]
        ),
    )
    examples = list(dict.fromkeys(["220288-0", best, worst]))
    write(
        HERE / "example-selection.json",
        {
            "scope": "post-hoc illustration, no representative-sample claim",
            "fixed_example": "220288-0",
            "largest_gain_among_resampled": best,
            "largest_loss_all_cases": worst,
            "examples": examples,
        },
    )
    font = ImageFont.truetype("DejaVuSans.ttf", 18)
    canvas = Image.new("RGB", (1440, len(examples) * 365), "#e9edf0")
    draw = ImageDraw.Draw(canvas)
    schedule = {c["key"]: c for c in tp["cases"]}
    for row, key in enumerate(examples):
        with np.load(bank / schedule[key]["file"], allow_pickle=False) as z:
            original = Image.fromarray(z["image"])
        for column, method in enumerate((None, "baseline-full", "consensus-text-full")):
            if method is None:
                image, title = original, key + "  Entrada"
            else:
                folder = test / "cases" / key / method
                meta = read(folder / "report.json")
                name = meta.get("output_image", "original-preserved-failure.png")
                image = Image.open(folder / name).convert("RGB")
                status = (
                    "retificada"
                    if meta["complete_geometry"]["resampled"]
                    else "original preservado"
                )
                title = NAMES[method.replace("-full", "")] + "  " + status
            draw.text(
                (column * 480 + 12, row * 365 + 12), title, font=font, fill="#21364b"
            )
            tile = ImageOps.contain(image, (464, 310), Image.Resampling.LANCZOS)
            x = column * 480 + 8 + (464 - tile.width) // 2
            y = row * 365 + 48 + (310 - tile.height) // 2
            canvas.paste(tile, (x, y))
    canvas.save(HERE / "output-examples.jpg", quality=92, subsampling=0)
    combo_pair = ts["primary"]["paper"]["paired"]["full"]["methods"][
        "consensus-text-full"
    ]
    pair = ts["primary"]["paper"]["paired"]["full"]["methods"]["baseline-text-full"]
    interval = pair["delta_95_descriptive_interval"]["g"]

    def embedded(name):
        suffix = Path(name).suffix
        mime = "image/jpeg" if suffix == ".jpg" else "image/png"
        return f'<img alt="{html.escape(name)}" src="data:{mime};base64,{base64.b64encode((HERE / name).read_bytes()).decode()}">'

    def row(label, value):
        return (
            "<tr><td>"
            + label
            + "</td>"
            + "".join(
                "<td>" + percent(value[k]) + "</td>" for k in ("g", "w", "c", "r")
            )
            + "</tr>"
        )

    table = "".join(row(label, value) for label, value in zip(labels, full_metrics))
    combined = tm["consensus-text-full"]["overall"]
    recommended = tm["baseline-text-full"]["overall"]
    baseline = tm["baseline-full"]["overall"]
    gain = 100 * pair["delta_vs_baseline"]["g"]
    title = "Digitalização de ECG: avanço automático M6"
    text = f"""<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;color:#21364b;background:#edf1f4;margin:0}}main{{max-width:1120px;margin:30px auto;padding:40px;background:white;border-radius:16px}}h1{{font-size:32px;line-height:1.2}}h2{{font-size:23px;margin-top:36px}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}td,th{{padding:12px;border-bottom:1px solid #d9e0e5;text-align:right}}td:first-child,th:first-child{{text-align:left}}.note{{background:#f1f5f7;padding:18px;border-radius:10px}}a{{color:#086f60}}small{{color:#526475}}@media(max-width:700px){{main{{padding:20px;margin:0}}table{{font-size:12px}}td,th{{padding:7px}}}}</style>
<main><small>26 de outubro de 2026 · Execução em CPU · Dados sintéticos de desenvolvimento</small><h1>{title}</h1>
<p>O ciclo anterior localizou um gargalo na organização da grade. Este ciclo implementou um organizador por consenso e uma escolha de orientação pela leitura dos rótulos do ECG. A referência geométrica entra apenas na avaliação posterior.</p>
<div class="note"><strong>Avanço observado com OCR e o organizador atual.</strong> Na avaliação retrospectiva, G passou de {percent(baseline["g"])} para {percent(recommended["g"])}, com ganho de {gain:.2f} pontos percentuais. O intervalo descritivo de 95% para a diferença foi [{100 * interval["lower"]:.2f}, {100 * interval["upper"]:.2f}] pontos. As coordenadas erradas entre as retornadas passaram de {percent(baseline["r"])} para {percent(recommended["r"])}. A cobertura retornada permaneceu igual.</div>
<h2>O que foi implementado</h2><p>O organizador compara bases com direções e espaçamentos distintos, integra restrições inteiras e pode alinhar componentes observados quando há consenso geométrico suficiente. Não cria interseções ou células nas lacunas. A etapa de OCR compara quatro rotações e exige evidência textual redundante e margem entre alternativas. Uma execução incompleta ou um texto ambíguo mantém a orientação não resolvida.</p>
<p>As duas mudanças são opcionais na CLI. A imagem usada na saída é o RGB original. A máscara geométrica permanece separada da imagem, e o sistema conserva a entrada quando rejeita a transformação completa.</p>
<h2>Comparação com as mesmas detecções</h2><p>A validação usa 24 imagens de seis pacientes. A avaliação retrospectiva usa 32 imagens de oito outros pacientes. Os pesos e as consultas do experimento anterior permanecem fixos. A combinação foi escolhida pela avaliação de células na validação, antes de calcular as coordenadas retrospectivas. Não houve novo treino.</p>
{embedded("partial-coordinates.png")}
<h2>Saída do pipeline completo</h2><p>G mede a fração de consultas com erro até 0,5 mm, W a fração retornada errada, C a fração retornada e R a razão W/C. A origem recebe somente uma translação calculada em âncoras separadas. A orientação não é corrigida pela referência na análise principal. O ajuste D4 permanece apenas como diagnóstico secundário.</p>
<table><thead><tr><th>Método</th><th>G correta</th><th>W errada</th><th>C retornada</th><th>R entre retornadas</th></tr></thead><tbody>{table}</tbody></table>
{embedded("full-pipeline.png")}
<p>O B0 usa o padrão atual da CLI, sem o refinamento opcional de pontos. M2 é a referência histórica com refinamento, medida no experimento anterior. As novas variantes deste ciclo usam os pontos originais. Quando a transformação completa é rejeitada, as coordenadas de células observadas ainda podem existir e o RGB permanece original.</p>
<p>Saídas RGB efetivamente retificadas: {html.escape(availability)}. Essas contagens descrevem disponibilidade, sem certificar precisão.</p>
<p><strong>O organizador por consenso ainda não justifica substituir o atual.</strong> A combinação escolhida na validação atingiu {percent(combined["g"])} de G na avaliação retrospectiva, abaixo do braço com OCR e organizador atual. Seu ganho contra B0 foi de {100 * combo_pair["delta_vs_baseline"]["g"]:.2f} pontos, com intervalo descritivo de [{100 * combo_pair["delta_95_descriptive_interval"]["g"]["lower"]:.2f}, {100 * combo_pair["delta_95_descriptive_interval"]["g"]["upper"]:.2f}], que inclui zero. O consenso isolado também não melhorou o agregado. A recomendação prática é usar o OCR opcional com o organizador atual e manter o consenso experimental. Essa leitura dos braços é retrospectiva e não equivale a uma seleção confirmada em conjunto novo.</p>
<p>O gráfico abaixo mostra as diferenças da combinação experimental com consenso. No braço com organizador atual e OCR, houve melhora de G em {ocr_case_changes["improved"]} casos, empate em {ocr_case_changes["tied"]} e piora em {ocr_case_changes["worse"]} neste painel.</p>
{embedded("case-differences.png")}
<h2>Inspeção das saídas</h2><p>O painel mostra o modo experimental com consenso e OCR. Inclui o exemplo fixo 220288-0, o maior ganho entre saídas retificadas e a maior perda global, escolhidos depois da avaliação para ilustrar acertos e limitações. Os 32 casos permanecem nas tabelas e nas médias.</p>
{embedded("output-examples.jpg")}
<h2>Falhas e limites mantidos</h2><p>O OCR resolveu a orientação em {ocr_summary["statuses"].get("resolved-from-text", 0)} das 32 imagens e deixou {ocr_summary["statuses"].get("unresolved", 0)} sem resolução. Em {len(ocr_summary["incomplete_OCR_cases"])} casos houve pelo menos uma alternativa de OCR incompleta. A mediana de duração do OCR foi {ocr_summary["latency_seconds"]["median"]:.2f} s e o máximo foi {ocr_summary["latency_seconds"]["max"]:.2f} s, incluindo os limites de tempo. Não resolver a orientação não significa confirmar que a imagem já estava correta.</p>
<p>O passo físico ainda assume interseções principais de 5 mm. Grades esparsas, falsas detecções, orientação ilegível e ambiguidades de passo continuam sendo problemas. As imagens já haviam sido inspecionadas em estudos anteriores. Os intervalos são descritivos e usam apenas oito pacientes na avaliação retrospectiva.</p>
<p>Este ciclo demonstra resultados geométricos automáticos em dados sintéticos. Não mede uma nova melhoria da forma de onda, não resolve automaticamente layout e calibração dos sinais e não constitui validação clínica ou comparação nova com sistemas externos.</p>
<h2>Verificação e próxima prioridade</h2><p>Os valores foram recontados a partir das coordenadas salvas, sem importar o avaliador original. A auditoria também confere que todas as células usam nós observados e compara amostras das saídas RGB com uma interpolação independente sobre a malha salva. A referência B0 completa reproduz exatamente as coordenadas e a elegibilidade arquivadas do M3 em todos os 32 casos.</p>
<p>A auditoria detectou um PNG incompleto em uma saída compartilhada. O arquivo foi recuperado da segunda gravação da mesma matriz RGB, sem alterar coordenadas ou métricas. A CLI passou a verificar todos os pixels decodificados antes de publicar cada PNG por substituição atômica. Testes verificam que uma gravação truncada ou com pixels alterados mantém a versão completa anterior.</p>
<p>A prioridade seguinte é melhorar a rejeição de identidades e passos de grade incorretos, reduzir a dependência de texto legível e medir a extração da forma de onda com essa geometria. A promoção de um novo padrão depende de um conjunto novo com pacientes e fotografias reais.</p>
<p><a href="https://github.com/fmenegui/ecg-torch-renderer/pull/25">Código, protocolos, tabelas e validação no PR #25</a>.</p></main></html>"""
    report.write_text(text)
    result = {
        "baseline_B0": baseline,
        "legacy_M2": legacy_mean,
        "baseline_with_text": recommended,
        "consensus_with_text": combined,
        "paired_OCR_vs_B0": pair,
        "paired_consensus_text_vs_B0": combo_pair,
        "resampled_counts": counts,
        "OCR_statuses": ocr_summary["statuses"],
        "OCR_case_changes": ocr_case_changes,
        "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "independent_count_checks": va["coordinate_count_checks"]
        + ta["coordinate_count_checks"],
        "independent_RGB_samples": ta["original_RGB_sample_checks"],
    }
    write(HERE / "completed-results.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("work", "comparison", "bank", "report"):
        parser.add_argument("--" + key, type=Path, required=True)
    build(**vars(parser.parse_args()))
