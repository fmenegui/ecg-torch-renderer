"""Summarize completed conditional signals and build the self-contained report."""
import argparse
import base64
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


METHODS = ['E5-reference-warp-bilinear', 'E5-reference-warp-2x2', 'E6-regenerated-canonical']
LABELS = ['Referência + bilinear', 'Referência + 2×2', 'Página canônica']


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def balanced(rows, field):
    values = []
    for patient in sorted({r['group_id'] for r in rows}):
        by_backend = []
        for backend in ('offline', 'torch-cached'):
            v = [r[field] for r in rows if r['group_id'] == patient and r['backend'] == backend and r[field] is not None]
            by_backend.append(float(np.mean(v)) if v else np.nan)
        values.append(float(np.nanmean(by_backend)) if np.isfinite(by_backend).any() else np.nan)
    return np.array(values)


def interval(vector, indices):
    draws = np.nanmean(vector[indices], axis=1)
    finite = draws[np.isfinite(draws)]
    return {'lower': float(np.quantile(finite, .025)), 'upper': float(np.quantile(finite, .975)),
            'defined_draws': len(finite)} if len(finite) else {'lower': None, 'upper': None, 'defined_draws': 0}


def build(work, out, report):
    out.mkdir(parents=True, exist_ok=True)
    signal = work/'m5-signals-final'
    complete = json.loads((signal/'complete.json').read_text())
    assert complete['scheduled'] == 32 and complete['case_method_results'] == 96
    rows = json.loads((signal/'cases.json').read_text())
    keys = sorted({r['key'] for r in rows})
    checks = 0
    for key in keys:
        archives = {m: np.load(signal/'cases'/key/(m+'.npz')) for m in METHODS}
        instance_ids = [k[:-6] for k in archives[METHODS[0]].files if k.endswith('_times')]
        common_sse = {m: 0. for m in METHODS}
        common_n, expected = 0, 0
        for i in instance_ids:
            reference = archives[METHODS[0]][i+'_reference']
            times = archives[METHODS[0]][i+'_times']
            prediction = {m: a[i+'_prediction'] for m, a in archives.items()}
            common = np.logical_and.reduce([np.isfinite(v) for v in prediction.values()])
            common_n += int(common.sum())
            expected += len(reference)
            for m, a in archives.items():
                assert np.array_equal(a[i+'_reference'], reference)
                assert np.array_equal(a[i+'_times'], times)
                common_sse[m] += float(np.sum((prediction[m][common]-reference[common])**2))
                checks += 2
        for m, a in archives.items():
            row = next(r for r in rows if r['key'] == key and r['method'] == m)
            observed, sse = 0, 0.
            for i in instance_ids:
                reference, prediction = a[i+'_reference'], a[i+'_prediction']
                keep = np.isfinite(prediction)
                observed += int(keep.sum())
                sse += float(np.sum((prediction[keep]-reference[keep])**2))
            assert row['expected_samples'] == expected and row['observed_samples'] == observed
            assert abs(row['coverage']-observed/expected) < 1e-15
            if observed:
                assert abs(row['rmse_mv']-np.sqrt(sse/observed)) < 1e-12
            row['common_observed_samples'] = common_n
            row['common_coverage'] = common_n/expected
            row['common_rmse_mv'] = float(np.sqrt(common_sse[m]/common_n)) if common_n else None
            a.close()
            checks += 3
    rng = np.random.default_rng(1973)
    bootstrap = rng.integers(0, 8, (20000, 8))
    by_method = {m: [r for r in rows if r['method'] == m] for m in METHODS}
    summary = {'scope': 'Conditional known-layout/calibration synthetic diagnostics, not native autonomous digitization.',
        'cases': 32, 'patients': 8, 'backends': 2, 'source_fs_hz': 100,
        'aggregation': 'Equal patient and backend means. RMSE conditional on observed printed samples.',
        'bootstrap': '20000 paired patient draws, descriptive, no multiplicity adjustment.',
        'common_support': 'Same printed-instance time samples finite in all three methods.',
        'recomputed_array_checks': checks, 'methods': {}, 'paired': {}}
    for m, r in by_method.items():
        summary['methods'][m] = {'coverage': float(balanced(r, 'coverage').mean()),
            'coverage_95': interval(balanced(r, 'coverage'), bootstrap),
            'rmse_conditional_mv': float(np.nanmean(balanced(r, 'rmse_mv'))),
            'rmse_defined_cases': sum(x['rmse_mv'] is not None for x in r),
            'common_coverage': float(balanced(r, 'common_coverage').mean()),
            'common_rmse_mv': float(np.nanmean(balanced(r, 'common_rmse_mv'))),
            'common_rmse_defined_cases': sum(x['common_rmse_mv'] is not None for x in r),
            'case_coverage_min': min(x['coverage'] for x in r)}
    for a, b in [(METHODS[1], METHODS[0]), (METHODS[0], METHODS[2])]:
        fields = {}
        for field in ('coverage', 'common_rmse_mv'):
            delta = balanced(by_method[a], field)-balanced(by_method[b], field)
            fields[field] = {'delta': float(np.nanmean(delta)), 'interval_95': interval(delta, bootstrap)}
        summary['paired'][a+' minus '+b] = fields
    mesh = []
    for key in keys:
        mesh.append({'key': key, **json.loads((signal/'cases'/key/'reference-mesh-validation.json').read_text())})
    summary['reference_field_approximation'] = {
        'max_case_p95_mm': max(r['p95_mm'] for r in mesh if r['p95_mm'] is not None),
        'min_query_coverage': min(r['query_coverage'] for r in mesh),
        'queries_per_case': 4096,
        'cases_p95_above_0_5_mm': [r['key'] for r in mesh if r['p95_mm'] is not None and r['p95_mm'] > .5]}
    write(out/'signal-summary.json', summary)
    write(out/'signal-mesh-validation.json', mesh)
    fields = ['key', 'group_id', 'backend', 'layout', 'method', 'expected_samples', 'observed_samples',
              'coverage', 'rmse_mv', 'common_observed_samples', 'common_coverage', 'common_rmse_mv']
    with (out/'signal-cases.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r[k] for k in fields} for r in rows)
    geometry = json.loads((out/'geometry-summary.json').read_text())
    means = geometry['domains']['paper']['means']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    pairs = [('E2-matched-inferred-IDs', 'E2-oracle-association-IDs'),
             ('E3-perfect-points-inferred-IDs', 'E4-perfect-points-perfect-IDs')]
    for ax, pair, title in zip(axes, pairs, ['Mesmos pontos detectados e associados', 'Mesmos pontos perfeitos']):
        for y, method in enumerate(pair):
            v = means[method]['overall']
            ax.barh(y, v['g']*100, color='#087e83', label='Correto' if y == 0 else None)
            ax.barh(y, v['w']*100, left=v['g']*100, color='#d98770', label='Errado' if y == 0 else None)
            ax.barh(y, (1-v['c'])*100, left=v['c']*100, color='#e3e9ed', label='Não retornado' if y == 0 else None)
            ax.text(max(v['g']*50, 10), y, f"{v['g']*100:.2f}%", va='center', ha='center', color='white', weight='bold')
        ax.set_yticks([0, 1], ['Organização\natual', 'Correspondências\nconhecidas'])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel('Consultas no papel (%)')
        ax.set_title(title, fontsize=11, loc='left')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(.53, .075), ncol=3, frameon=False)
    fig.suptitle('O gargalo persiste mesmo com pontos perfeitos', x=.03, ha='left', weight='bold', fontsize=15)
    fig.text(.03, .02, '32 capturas sintéticas, 8 pacientes. Intervenções com referência conhecida. Não são ganhos de um modelo automático.', fontsize=9, color='#526774')
    fig.tight_layout(rect=(.01, .14, .99, .92), h_pad=2.)
    fig.savefig(out/'geometry-diagnostics.svg', bbox_inches='tight')
    fig.savefig(out/'geometry-diagnostics.png', dpi=170, bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    coverage = [summary['methods'][m]['coverage']*100 for m in METHODS]
    for ax, values, label in [(axes[0], coverage, 'Tempo impresso reconstruído (%)'),
        (axes[1], [summary['methods'][m]['common_rmse_mv'] for m in METHODS], 'RMSE no suporte comum (mV)')]:
        ax.bar(np.arange(3), values, color=['#3a7d9a', '#76a9b3', '#234856'], width=.6)
        ax.set_xticks(np.arange(3), ['Ref. bilinear', 'Ref. 2×2', 'Canônica'])
        ax.set_ylabel(label)
        for i, value in enumerate(values):
            ax.text(i, value, f'{value:.2f}' if ax is axes[0] else f'{value:.4f}', ha='center', va='bottom')
        ax.set_ylim(0, 100 if ax is axes[0] else max(values)*1.2)
    fig.suptitle('Extração condicional dos sinais', x=.03, ha='left', weight='bold', fontsize=15)
    fig.text(.03, .02, 'Layout, calibração e geometria conhecidos. Referências de 100 Hz. Mesma rede de segmentação e decoder sem preencher lacunas.', fontsize=9, color='#526774')
    fig.tight_layout(rect=(0, .06, 1, .9))
    fig.savefig(out/'signal-diagnostics.svg', bbox_inches='tight')
    fig.savefig(out/'signal-diagnostics.png', dpi=170, bbox_inches='tight')
    plt.close(fig)
    key = '220288-2'
    first = np.load(signal/'cases'/key/(METHODS[2]+'.npz'))
    i = next(k[:-6] for k in first.files if k.endswith('_times'))
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(first[i+'_times'], first[i+'_reference'], color='#152d39', lw=1.5, label='Sinal de referência')
    for method, label, color in zip(METHODS, LABELS, ['#d8844e', '#278d98', '#7f65a3']):
        with np.load(signal/'cases'/key/(method+'.npz')) as a:
            ax.plot(a[i+'_times'], a[i+'_prediction'], lw=1, color=color, label=label, alpha=.8)
    ax.set_xlabel('Tempo (s)')
    ax.set_ylabel('Amplitude (mV)')
    ax.set_title(f'Exemplo fixo: {key}, instância {i}. Lacunas permanecem ausentes.', loc='left')
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out/'waveform-example.svg', bbox_inches='tight')
    fig.savefig(out/'waveform-example.png', dpi=170, bbox_inches='tight')
    plt.close(fig)

    def image_data(name):
        b = base64.b64encode((out/name).read_bytes()).decode()
        return '<img alt="Gráfico do experimento" src="data:image/png;base64,'+b+'">'
    metric_rows = ''
    for method, label in zip(METHODS, LABELS):
        v = summary['methods'][method]
        metric_rows += f"<tr><td>{label}</td><td>{v['coverage']*100:.2f}%</td><td>{v['rmse_conditional_mv']:.5f}</td><td>{v['common_rmse_mv']:.5f}</td></tr>"
    geom_rows = ''
    for method, label in [('E0-native-M2', 'M2 atual'), ('E1-orientation-oracle', 'M2 + orientação conhecida'),
                         ('E2-matched-inferred-IDs', 'Detecções associadas, organização atual'),
                         ('E2-oracle-association-IDs', 'Mesmas detecções, correspondências conhecidas'),
                         ('E3-perfect-points-inferred-IDs', 'Pontos perfeitos, organização atual'),
                         ('E4-perfect-points-perfect-IDs', 'Pontos e correspondências perfeitos')]:
        v = means[method]['overall']
        geom_rows += '<tr><td>'+label+'</td>'+''.join(f'<td>{v[k]*100:.2f}%</td>' for k in ('g', 'w', 'c', 'r'))+'</tr>'
    report.parent.mkdir(parents=True, exist_ok=True)
    body = f'''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Digitalização ECG: execução M0/M1</title>
<style>body{{margin:0;background:#edf2f4;color:#18313e;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1080px;margin:30px auto;padding:44px 52px;background:white;border-radius:14px}}h1{{font-size:37px;line-height:1.15}}h2{{margin-top:38px;border-top:1px solid #dce5e8;padding-top:22px}}.lead{{font-size:21px}}.note{{padding:16px 20px;background:#eaf4f3;border-left:4px solid #087e83}}.small{{font-size:13px;color:#536a77}}img{{display:block;width:100%;height:auto;margin:20px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #dce5e8;text-align:left}}th{{background:#eff5f7}}.table{{overflow-x:auto}}code{{font-size:13px;overflow-wrap:anywhere}}a{{color:#096d7c}}@media(max-width:700px){{main{{margin:0;padding:24px 18px;border-radius:0}}h1{{font-size:29px}}table{{min-width:650px}}}}@media print{{main{{padding:0;margin:0}}body{{background:white;font-size:10pt}}img,tr{{break-inside:avoid}}}}</style></head><body><main>
<p class="small">26 de outubro de 2026 · Execução do primeiro ciclo · 32 capturas sintéticas, 8 pacientes</p>
<h1>O diagnóstico foi executado. A prioridade é corrigir a organização da grade.</h1>
<p class="lead">Pontos perfeitos ainda produzem geometria errada com o organizador atual. O extrator de sinal também apresenta limitações mesmo com layout e calibração conhecidos.</p>
<div class="note"><strong>Resultado principal:</strong> mantendo pontos perfeitos, a cobertura correta passa de 64,39% para 86,58% ao fornecer correspondências e organização de referência. Diferença de 22,18 pontos percentuais, intervalo descritivo de 95% [14,76, 28,98]. Esse experimento usa referência conhecida e não representa um novo método automático.</div>
<h2>Reprodução e integridade</h2>
<p>Os hashes das 112 capturas originais conferiram. O resumo M3 foi reproduzido byte a byte a partir dos 160 resultados agendados. Reexecutei o detector em CPU nas 32 capturas e obtive exatamente as coordenadas e os scores arquivados. Os 14 registros públicos originais tiveram os hashes de cabeçalho e sinal conferidos.</p>
<p>Os controles independentes M4 de geometria, amostragem e exportação foram reexecutados. Uma recontagem independente dos novos resultados geométricos passou 1.280 verificações, sem importar o código de pontuação original. A galeria de 16 cenários foi regenerada e inspecionada.</p>
<h2>Onde a geometria falha</h2>
{image_data('geometry-diagnostics.png')}
<div class="table"><table><tr><th>Intervenção</th><th>G correto</th><th>W errado</th><th>C retornado</th><th>R = W/C</th></tr>{geom_rows}</table></div>
<p>O limiar é 0,5 mm. As médias equilibram pacientes e os dois renderizadores. Os intervalos usam 20 mil reamostragens dos oito pacientes, com interpretação descritiva. Esses casos já haviam sido examinados no desenvolvimento.</p>
<p>Na comparação que conserva as mesmas detecções associadas, G sobe 15,31 pontos percentuais [8,51, 22,07]. Os dois braços retiram detecções sem associação confiável de referência. A construção conhecida também altera os componentes e as células retidos, portanto o efeito não pode ser atribuído somente ao passo em milímetros.</p>
<p>Os braços de componentes retornam coordenadas parciais. Eles não representam sucesso de retificação do RGB. Zero erro nas consultas do braço perfeito vale apenas para esse painel e esse limiar.</p>
<h2>Extração dos sinais com informação conhecida</h2>
{image_data('signal-diagnostics.png')}
<div class="table"><table><tr><th>Entrada do extrator comum</th><th>Cobertura temporal</th><th>RMSE condicional (mV)</th><th>RMSE no suporte comum (mV)</th></tr>{metric_rows}</table></div>
<p>A geometria de referência, o layout, as derivações, o ganho, a velocidade, a linha de base e as janelas impressas foram fornecidos para este diagnóstico. Usei a mesma rede de segmentação npj, com pesos verificados, seguida pelo decoder de regiões que conserva lacunas. Não ajustei tempo, amplitude ou componente DC para melhorar o erro.</p>
<p>O RMSE condicional considera somente amostras observadas. O suporte comum considera os mesmos instantes recuperados nos três braços, com cobertura de {summary['methods'][METHODS[0]]['common_coverage']*100:.2f}% do tempo impresso. Os resultados de cobertura incluem todos os casos. A verificação dos arrays realizou {checks} checagens de referências, tempos, contagens e erros.</p>
<p>O mapa de referência foi aproximado por células adjacentes de 8 pixels. O maior p95 de erro entre os casos foi {summary['reference_field_approximation']['max_case_p95_mm']:.5f} mm. A menor cobertura de consultas foi {summary['reference_field_approximation']['min_query_coverage']*100:.2f}%. Essa aproximação foi medida, sem chamar o warp de geometria analítica exata.</p>
{image_data('waveform-example.png')}
<p><strong>Falhas graves:</strong> no braço bilinear, o caso 215864-6 apresentou RMSE de 6,81044 mV com cobertura de 59,11%. O caso 206852-2 teve RMSE de 4,02439 mV e cobertura de 23,00%. Esses erros permanecem no resultado. Uma verificação posterior com o alvo limpo arquivado apoiou o alinhamento nos dois exemplos, sem ajuste de parâmetros, mas não prova fidelidade de cada onda.</p>
<p>O braço 2×2 alterou a cobertura em -0,228 ponto percentual, intervalo descritivo de 95% [-0,600, +0,100], e o RMSE no suporte comum em -0,00055 mV [-0,00271, +0,00201]. Esses intervalos não sustentam adotar 2×2 como uma melhora consistente neste painel.</p>
<p>O contraste entre a captura retificada e a página canônica inclui dano da captura, visibilidade, reamostragem e transferência do extrator. Ele não isola apenas a interpolação. O braço 2×2 é experimental e não garante preservação universal de traços.</p>
<h2>Decisão após os resultados</h2>
<ol><li><strong>Organização da grade primeiro.</strong> Tratar orientação semântica, hipóteses de passo, identidades e múltiplas ilhas. A melhora do detector sozinha deixa uma parcela importante de geometria errada.</li><li><strong>Manter um teste independente do extrator.</strong> A página canônica fornece um piso mensurado para a recuperação do sinal. Comparar alterações de geometria com o mesmo extrator e preservar a máscara de observação.</li><li><strong>Validar reamostragem e resolução separadamente.</strong> Os controles conhecidos continuam impedindo declarar uma quadratura fixa suficiente para qualquer captura.</li><li><strong>Congelar o próximo candidato e testar em fontes externas.</strong> Estes resultados sintéticos orientam desenvolvimento. Não demonstram superioridade clínica ou comercial.</li></ol>
<h2>Escopo e reprodução</h2>
<p>As referências de sinal têm 100 Hz. Este estudo não valida recuperação de espículas curtas, detalhes acima dessa resolução ou equivalência clínica. O extrator condicional não é o pipeline nativo completo do npj. Não foi executado Dotter/Gridder comercial nem um novo benchmark contra os top 3 do Kaggle.</p>
<p>Código, testes, protocolos, tabelas por caso e resultados compactos estão no <a href="https://github.com/fmenegui/ecg-torch-renderer/pull/24">PR #24</a>. As medições completas incluem coordenadas, imagens e sinais com lacunas explícitas. A validação de integração e seus limites estão registrados em <code>validation.json</code> no mesmo diretório do estudo.</p>
<p class="small">Fontes: M2/M3/M4 arquivados no repositório. Sinais PTB-XL 1.0.3, CC BY 4.0, DOI 10.1038/s41597-020-0495-6. Segmentação: Ahus-AIM/Open-ECG-Digitizer, commit 97a15087d4abcda843da8c58ee74b1d8f47e6f9a, Stenhede et al., npj Digital Medicine (2026), DOI 10.1038/s41746-025-02327-1. Sem novo treinamento.</p>
</main></body></html>'''
    report.write_text(body)
    print(json.dumps({'report': str(report), 'signal_results': summary}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('work', 'out', 'report'):
        parser.add_argument('--'+name, type=Path, required=True)
    build(**vars(parser.parse_args()))
