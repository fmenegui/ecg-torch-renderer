"""Render an external ECG pilot report from completed, independently audited data."""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "npj_native": "Modelo publicado",
    "text_then_npj": "Orientação + publicado",
    "m6_then_npj": "M6 + publicado",
}
COLORS = {"npj_native": "#64748b", "text_then_npj": "#2563eb", "m6_then_npj": "#d97706"}


def embedded(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def fmt(value, places=3):
    return "ausente" if value is None else f"{value:.{places}f}".replace(".", ",")


def save_figure(fig, path):
    fig.savefig(
        path.with_suffix(".png"), dpi=160, facecolor="white", bbox_inches="tight"
    )
    fig.savefig(path.with_suffix(".svg"), facecolor="white", bbox_inches="tight")
    plt.close(fig)


def build(results, manifest, run, output):
    results, manifest, run, output = map(Path, (results, manifest, run, output))
    summary = json.loads((results / "summary.json").read_text())
    data = json.loads(manifest.read_text())
    report = json.loads((run / "report.json").read_text())
    rows = json.loads((results / "cases.json").read_text())
    audit = json.loads((results / "independent-audit.json").read_text())
    source_audit = json.loads((results / "original-source-audit.json").read_text())
    if (summary["patients"], summary["images"], summary["case_method_results"]) != (
        8,
        24,
        72,
    ):
        raise ValueError(
            "This fixed M7 report requires eight patients and all 72 scheduled results"
        )
    if (
        not audit["passed"]
        or not source_audit["passed"]
        or report["status"] != "completed"
    ):
        raise ValueError("Completed audited evidence required")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
        }
    )
    arms = list(LABELS)
    patients = sorted(summary["arms"][arms[0]]["per_patient"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for arm in arms:
        values = summary["arms"][arm]["per_patient"]
        axes[0].plot(
            range(len(patients)),
            [values[p]["coverage"] * 100 for p in patients],
            "o-",
            color=COLORS[arm],
            label=LABELS[arm],
            ms=5,
            lw=1.5,
        )
        axes[1].plot(
            range(len(patients)),
            [values[p]["raw_all_printed_nrmse"] for p in patients],
            "o-",
            color=COLORS[arm],
            label=LABELS[arm],
            ms=5,
            lw=1.5,
        )
    for ax in axes:
        ax.set_xticks(range(len(patients)), [f"P{i + 1}" for i in range(len(patients))])
        ax.set_xlabel("Paciente, média das três capturas")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Saída finita em instantes impressos (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("Cobertura do sinal")
    axes[1].set_ylabel("RMSE / RMS do sinal de referência")
    axes[1].axhline(1.0, ls=":", color="#94a3b8", lw=1)
    axes[1].set_title("Erro em todo o sinal impresso, menor é melhor")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.07),
        frameon=False,
    )
    fig.tight_layout()
    save_figure(fig, results / "patient-comparison")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.1))
    conditions = data["conditions"]
    names = ["Scan", "Celular", "Papel dobrado"]
    x = np.arange(3)
    for i, arm in enumerate(arms):
        groups = summary["arms"][arm]["by_condition"]
        axes[0].bar(
            x + (i - 1) * 0.24,
            [groups[c]["macro"]["coverage"] * 100 for c in conditions],
            width=0.23,
            color=COLORS[arm],
            label=LABELS[arm],
        )
        axes[1].bar(
            x + (i - 1) * 0.24,
            [groups[c]["macro"]["raw_all_printed_nrmse"] for c in conditions],
            width=0.23,
            color=COLORS[arm],
        )
    for ax in axes:
        ax.set_xticks(x, names)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Cobertura média por paciente (%)")
    axes[0].set_ylim(0, 105)
    axes[1].set_ylabel("NRMSE em todo o sinal impresso")
    axes[0].set_title("Condições reais de aquisição")
    axes[1].set_title("Ausências mantidas na pontuação")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.06),
        frameon=False,
    )
    fig.tight_layout()
    save_figure(fig, results / "condition-comparison")
    cases = {r["image_id"]: r for r in data["cases"]}
    by_id = {(r["image_id"], r["arm"]): r for r in rows}
    delta = [
        (
            by_id[(k, "m6_then_npj")]["raw_all_printed_nrmse"]
            - by_id[(k, "npj_native")]["raw_all_printed_nrmse"],
            k,
        )
        for k in cases
    ]
    selected = [min(delta)[1], max(delta)[1]]
    selection = {
        "rule": "Largest improvement and largest worsening in M6 all-printed NRMSE, selected for display after evaluation only",
        "image_ids": selected,
    }
    (results / "example-selection.json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    for case_index, key in enumerate(selected):
        row = cases[key]
        with np.load(manifest.parent / row["reference"]) as z:
            truth = z["values_mv"]
        predictions = {}
        for arm in arms:
            record = next(
                r for r in report["cases"] if r["image_id"] == key and r["arm"] == arm
            )
            if record["status"] == "failed":
                predictions[arm] = np.full_like(truth, np.nan)
            else:
                with np.load(run / record["prediction"]) as z:
                    predictions[arm] = z["values_mv"]
        for lead_index, lead in enumerate([1, 6]):
            ax = axes[case_index * 2 + lead_index]
            time = np.arange(5000) / 500
            ax.plot(
                time, truth[lead], color="#111827", lw=1.5, label="Referência PM-ECG-ID"
            )
            mask = np.isfinite(truth[lead])
            for arm in arms:
                curve = np.where(mask, predictions[arm][lead], np.nan)
                ax.plot(
                    time, curve, color=COLORS[arm], alpha=0.8, lw=1, label=LABELS[arm]
                )
            shown = np.flatnonzero(mask)
            ax.set_xlim(max(0, shown[0] / 500 - 0.05), min(10, shown[-1] / 500 + 0.05))
            ax.set_ylabel(("II" if lead == 1 else "V1") + " (mV)")
            ax.grid(alpha=0.2)
            if lead_index == 0:
                direction = "Maior melhora" if case_index == 0 else "Maior piora"
                ax.set_title(
                    f"{direction} da M6 no NRMSE: {key} · {row['reference_layout_scoring_only']}"
                )
            ax.set_xlabel(
                "Tempo impresso (s), sem ajuste de amplitude, tempo ou linha de base"
            )
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.015),
        frameon=False,
    )
    fig.tight_layout()
    save_figure(fig, results / "signal-examples")
    table = ""
    for arm in arms:
        s = summary["arms"][arm]
        m = s["macro"]
        table += f"<tr><th>{LABELS[arm]}</th><td>{fmt(m['coverage'] * 100, 1)}%</td><td>{fmt(m['raw_all_printed_nrmse'])}</td><td>{fmt(m['raw_observed_rmse_mv'])} mV</td><td>{s['lead_coverage_failures']}/{s['expected_leads']}</td><td>{s['exceptions']}</td></tr>"
    paired = ""
    for comparison, values in summary["paired"].items():
        after, before = comparison.split(" minus ")
        c, n = values["coverage"], values["raw_all_printed_nrmse"]
        ci, ni = (
            c["descriptive_patient_bootstrap_95"],
            n["descriptive_patient_bootstrap_95"],
        )
        paired += f"<tr><th>{LABELS[after]} − {LABELS[before]}</th><td>{fmt(c['delta'] * 100, 2)} pp<br><small>[{fmt(ci[0] * 100, 2)}, {fmt(ci[1] * 100, 2)}]</small></td><td>{fmt(n['delta'])}<br><small>[{fmt(ni[0])}, {fmt(ni[1])}]</small></td></tr>"
    secondary = ""
    for arm in arms:
        s = summary["arms"][arm]
        eligible = sum(r["aligned_leads"] for r in rows if r["arm"] == arm)
        secondary += f"<tr><th>{LABELS[arm]}</th><td>{fmt(s['macro']['mean_eligible_aligned_lead_snr_db'], 2)} dB</td><td>{fmt(s['macro']['mean_eligible_aligned_lead_rmse_mv'])} mV</td><td>{eligible}/{s['expected_leads']}</td></tr>"
    case_table = ""
    for r in rows:
        case_table += f"<tr><td>{html.escape(r['image_id'])}</td><td>{LABELS[r['arm']]}</td><td>{fmt(r['coverage'] * 100, 1)}%</td><td>{fmt(r['raw_all_printed_nrmse'])}</td><td>{fmt(r['raw_observed_rmse_mv'])} mV</td><td>{r['status']}</td></tr>"
    text = f'''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ECG · teste externo M7</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;color:#172033;background:#f3f6fa;max-width:1120px;margin:auto;padding:28px}}h1{{font-size:38px;line-height:1.15;max-width:860px}}h2{{font-size:24px;margin-top:0}}.eyebrow{{text-transform:uppercase;letter-spacing:.13em;font-size:12px;color:#2563eb;font-weight:700}}section{{background:white;padding:26px;border-radius:16px;margin:22px 0;border:1px solid #e2e8f0}}.notice{{border-left:4px solid #d97706;padding-left:18px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.card{{background:#eaf0f8;padding:18px;border-radius:12px}}.number{{font-size:30px;font-weight:750;display:block}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border-bottom:1px solid #e2e8f0;text-align:left;vertical-align:top;padding:11px 8px}}th{{font-weight:650}}.scroll{{overflow-x:auto}}img{{display:block;width:100%;height:auto;margin:14px 0}}small,.muted{{color:#526176}}a,summary{{color:#1d4ed8}}summary{{cursor:pointer;font-weight:650}}code{{background:#edf2f7;padding:2px 5px}}@media(max-width:700px){{body{{padding:14px}}section{{padding:18px}}h1{{font-size:30px}}.cards{{grid-template-columns:1fr}}}}</style></head><body>
<p class="eyebrow">Digitalização de ECG · M7 · 26 de outubro de 2026</p><h1>Da imagem externa ao sinal, com todas as falhas visíveis</h1>
<p>Primeiro piloto completo desta rodada com capturas físicas do PM-ECG-ID. Comparação pareada de três caminhos automáticos usando os mesmos sinais, regras de avaliação e modelo publicado.</p>
<div class="cards"><div class="card"><span class="number">24</span>imagens externas</div><div class="card"><span class="number">8</span>pacientes sem sobreposição conhecida com nosso banco anterior</div><div class="card"><span class="number">72</span>resultados programados e preservados</div></div>
<section><h2>O que mudou</h2><p>Agora a avaliação recebe a imagem original e termina em sinais de doze derivações. O reconhecimento do layout, a geometria e a extração são executados. O sinal verdadeiro, o layout esperado e as janelas impressas entram apenas na pontuação.</p><p>As três opções são o Open-ECG-Digitizer publicado, a orientação por texto seguida desse mesmo modelo e a retificação M6 seguida desse mesmo modelo. A M6 usa o organizador anterior, orientação por texto e RGB original a 4 pixels/mm.</p><p class="notice">Este é um piloto de desenvolvimento com oito pacientes. Não comprova estado da arte, equivalência clínica ou generalização para novos hospitais. Não houve treino ou ajuste dos parâmetros após observar os resultados.</p></section>
<section><h2>Resultado principal</h2><div class="scroll"><table><thead><tr><th>Caminho</th><th>Cobertura</th><th>NRMSE de todo o sinal</th><th>RMSE observado</th><th>Leads com &lt;90% de cobertura</th><th>Exceções</th></tr></thead><tbody>{table}</tbody></table></div>
<p class="muted">Médias por paciente, após calcular a média de suas três capturas. Cobertura significa saída finita nos instantes impressos e inclui valores incorretos. A fidelidade aparece nos erros de sinal. NRMSE é o erro RMS dividido pelo RMS do sinal de referência. Ausências recebem zero somente na pontuação de todo o sinal. Esse zero nunca é apresentado como sinal reconstruído. O RMSE observado é condicionado aos trechos recuperados e deve ser lido junto à cobertura.</p><img src="{embedded(results / "patient-comparison.png")}" alt="Cobertura e erro normalizado de cada paciente nos três caminhos"></section>
<section><h2>Quanto cada etapa altera o resultado</h2><div class="scroll"><table><tr><th>Comparação pareada</th><th>Variação de cobertura</th><th>Variação de NRMSE</th></tr>{paired}</table></div><p class="muted">Intervalos descritivos de 95% com 20.000 reamostragens dos oito pacientes. Cobertura maior e NRMSE menor são favoráveis. As variantes da mesma pessoa permanecem juntas. Os intervalos não transformam este piloto em um teste confirmatório.</p><img src="{embedded(results / "condition-comparison.png")}" alt="Resultados em scans, fotos de celular e papel dobrado"></section>
<section><h2>Inspeção dos sinais recuperados</h2><p>Exemplos escolhidos pela maior melhora e pela maior piora da M6 no erro de todo o sinal. As curvas mostram os valores originais, sem deslocar tempo, ajustar amplitude ou corrigir a linha de base.</p><p>No scan 6x2 do exame 9, a inspeção visual mostrou traços preservados, mas o modelo mudou de 6x2 para um layout só de derivações periféricas após a M6. A cobertura caiu de 95,2% para 45,9%. Uma grade visualmente regular não garante reconhecimento correto das derivações.</p><img src="{embedded(results / "signal-examples.png")}" alt="Referência e sinais digitalizados nos casos de maior melhora e maior piora"></section>
<section><h2>Alinhamento, apresentado separadamente</h2><div class="scroll"><table><tr><th>Caminho</th><th>SNR alinhado</th><th>RMSE alinhado</th><th>Leads elegíveis</th></tr>{secondary}</table></div><p>Este resultado secundário permite até 100 ms de deslocamento e remove as médias separadas. Só pontua uma derivação quando o alinhamento mantém pelo menos 90% de seus instantes impressos originais. Não há ajuste de amplitude. Uma média boa com poucos leads elegíveis não significa sucesso completo.</p><p>Estes valores não são os do leaderboard Kaggle nem reproduzem o protocolo do artigo comercial, que permite outro alinhamento e usa outros denominadores.</p></section>
<section><h2>Confiabilidade dos dados e das medições</h2><p>Os oito pacientes foram conferidos na tabela original do PTB-XL, com o mesmo checksum usado no banco anterior. Dois exames inicialmente sorteados foram excluídos antes da inferência. As capturas incluem quatro layouts de uma página, scans, celular Samsung e papel dobrado.</p><p>As {source_audit["lead_checks"]} derivações foram comparadas às fontes WFDB originais, com validação de valores iniciais e checksums. Após medir apenas o deslocamento DC constante aplicado pela fonte, o maior resíduo foi {source_audit["max_centered_residual_mv"]:.2g} mV. As referências usadas na pontuação não foram modificadas.</p><p>A auditoria separada recontou os arquivos de previsão e verificou {audit["recount_checks"]} medidas, sem importar o avaliador. Nenhum dos 72 resultados programados foi removido. Os membros baixados do arquivo público foram conferidos por CRC e fixados por SHA-256. O checksum do ZIP completo de 33,9 GB não foi verificado.</p></section>
<section><h2>Limites para comparar com a literatura</h2><p>O código e os pesos do Open-ECG-Digitizer foram mantidos no commit publicado fixado. Esta execução usa CPU float32, dois threads e lado máximo de 1500 pixels, com catálogo completo de layouts. A resolução difere do preset publicado de 3000 pixels. As convenções de dez segundos, 500 Hz, 25 mm/s e 10 mm/mV são fixas. A comparação M6 inclui geometria, orientação e reamostragem RGB, portanto não isola a geometria dos efeitos da amostragem.</p><p>Não conhecemos toda a sobreposição dos sinais PTB-XL com o treino do modelo publicado. O conjunto é externo ao nosso gerador de imagens, mas isso não demonstra independência de todos os sinais ou fabricantes usados pelos outros autores. O sistema comercial da Powerful Medical e o ensemble completo do Kaggle não foram executados.</p><p>O marco agora é uma comparação reproduzível de componentes na mesma tarefa completa. A decisão sobre novos padrões deve considerar as perdas de cobertura, os layouts adversos e uma próxima amostra que ainda não tenha sido inspecionada.</p></section>
<section><details><summary>Todos os resultados por imagem e caminho</summary><div class="scroll"><table><tr><th>Imagem</th><th>Caminho</th><th>Cobertura</th><th>NRMSE</th><th>RMSE observado</th><th>Execução</th></tr>{case_table}</table></div></details></section>
<section><h2>Código e fontes</h2><p><a href="https://github.com/fmenegui/ecg-torch-renderer/pull/26">PR 26, código, protocolo e evidências</a></p><p><a href="https://zenodo.org/records/13617673">PM-ECG-ID, Iring e colaboradores</a>, dados sob GPL-3.0-or-later. <a href="https://physionet.org/content/ptb-xl/1.0.3/">PTB-XL 1.0.3</a>, dados sob CC BY 4.0.</p><p><a href="https://doi.org/10.1038/s41746-025-02327-1">Stenhede, Bjørnstad e Ranjbar, npj Digital Medicine, 2026</a>. <a href="https://doi.org/10.1016/j.jelectrocard.2025.153900">Demolder e colaboradores, Journal of Electrocardiology, 2025</a>.</p><p class="muted">Resultados de engenharia e pesquisa. Nenhum diagnóstico clínico é produzido por esta avaliação.</p></section></body></html>'''
    output.write_text(text)
    print(
        json.dumps(
            {
                "report": str(output),
                "bytes": output.stat().st_size,
                "examples": selected,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "manifest", "run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    build(**vars(parser.parse_args()))
