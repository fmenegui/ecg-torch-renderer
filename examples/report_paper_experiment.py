"""Build a self-contained comparison from completed, checksum-matched experiments."""
import argparse
import base64
import html
import json
from pathlib import Path


def build(inputs, results, out):
    inputs, results, out = map(Path, (inputs, results, out))
    manifest_path = inputs/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    from hashlib import sha256
    digest = sha256(manifest_path.read_bytes()).hexdigest()
    npj = json.loads((results/'npj-profiles/report.json').read_text())
    kaggle = json.loads((results/'kaggle/report.json').read_text())
    environment = json.loads((results/'environment.json').read_text())
    for report in (npj, kaggle):
        if report['status'] != 'completed' or report['input_manifest_sha256'] != digest:
            raise ValueError('Completed reports from exactly the same input manifest required')
    records = {(c['arm'], c['name']): c for c in npj['cases']}
    records.update({('kaggle', c['name']): c for c in kaggle['cases']})
    names = [c['name'] for c in manifest['cases']]
    if len(names) != len(set(names)) or any((a, n) not in records for a in ('published', 'configured', 'kaggle') for n in names):
        raise ValueError('A unique complete three-arm matrix is required')

    def embed(path):
        mime = 'image/svg+xml' if path.suffix == '.svg' else 'image/png'
        return 'data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode()

    def cell(record):
        e = record.get('evaluation', {})
        score = e.get('mean_observed_lead_snr_db')
        metrics = (f'<b>{score:.2f} dB</b><br>{e["printed_time_coverage"]:.1%} cobertura'
                   f'<br>{e["unsupported_output_fraction"]:.1%} saída fora do trecho impresso'
                   if score is not None else 'Sem escore')
        flags = []
        if record['status'] == 'failed':
            flags.append('Rejeitado' if e else 'Falha')
        if 'layout_matches_expected' in record and not record['layout_matches_expected']:
            flags.append('Layout incorreto')
        if score is not None and score < 0:
            flags.append('Erro maior que a energia do sinal')
        return metrics+'<br><span class="warn">'+html.escape(' · '.join(flags))+'</span>'

    page = '''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Experimento · digitalização de ECG</title><style>
body{font:16px/1.55 system-ui;background:#0b1220;color:#e2e8f0;max-width:1300px;margin:auto;padding:28px}h1{line-height:1.15;font-size:36px}h2{line-height:1.2}a,summary{color:#7dd3fc}summary{cursor:pointer;padding:8px 0}article,.note{background:#162235;border-radius:12px;padding:22px;margin:22px 0}.warn{color:#fbbf24}table{border-collapse:collapse;width:100%}th,td{padding:12px;text-align:left;vertical-align:top;border-bottom:1px solid #334155}img{width:100%;height:auto}code{overflow-wrap:anywhere}.scroll{overflow-x:auto}small{color:#94a3b8}</style>
<h1>Até onde os modelos transferem para outros papéis?</h1>
<p>10 imagens do nosso gerador, 3 condições, pesos congelados. Oito capturas online e duas capturas offline por path tracing, incluindo uma dobra.</p>
<div class="note"><b>Leitura principal</b><p>A configuração de papel corrige falhas de catálogo e calibração do Open-ECG-Digitizer. Isso não resolve a deformação não plana nem torna o modelo universal. O Heng/Tang mantém dependências rígidas de referências impressas, grade e quatro linhas.</p>
<p>O modo configurado recebe o ganho nominal do papel. Não recebe vértices verdadeiros, caixas das derivações, baselines ou sinal de referência. A aquisição de 10 segundos e a saída a 500 Hz são convenções fornecidas. Não houve treinamento ou reconhecimento automático de ganho.</p></div>'''
    page += '<p>Dispositivo: '+html.escape(str(environment.get('gpu_name') or environment['device']))
    page += ' · Torch '+html.escape(environment['torch'])+' · float32 · TF32 desativado.</p>'
    page += '''<p><b>Métrica:</b> média do SNR por derivação nos trechos observados, com alinhamento limitado a ±100 ms e remoção de DC. É uma comparação diagnóstica, não a métrica oficial do Kaggle. Cobertura conta todo o tempo impresso. SNR alto pode coexistir com lacunas e saída atribuída a trechos que não foram impressos. Curvas abaixo aparecem sem alinhamento.</p>
<p>Heng Stage0/1 + Tang run15/fold0 é o perfil single. O ensemble completo compartilha a geometria de Stage0/1. A referência npj usa o catálogo completo dos autores. O perfil configurado adiciona Cabrera 3×4, preserva retrato, amplia a busca da grade e recebe o ganho declarado.</p>
<div class="scroll"><table><tr><th>Papel</th><th>Kaggle · single</th><th>npj · referência</th><th>npj · configurado</th></tr>'''
    for case in manifest['cases']:
        name = case['name']
        page += '<tr><td><a href="#'+html.escape(name)+'">'+html.escape(name)+'</a><br>'+html.escape(case['layout'])+'</td>'
        for arm in ('kaggle', 'published', 'configured'):
            page += '<td>'+cell(records[(arm, name)])+'</td>'
        page += '</tr>'
    page += '</table></div><p class="warn">Um único sinal artificial e uma captura por condição. Estes resultados não estimam desempenho clínico ou a frequência de falhas no mundo real.</p>'
    for case in manifest['cases']:
        name = case['name']
        image = inputs/case['image']
        if sha256(image.read_bytes()).hexdigest() != case['image_sha256']:
            raise ValueError('Image checksum mismatch')
        page += '<article id="'+html.escape(name)+'"><h2>'+html.escape(name)+'</h2>'
        page += '<details><summary>Ver imagem de entrada</summary><img alt="ECG gerado" src="'+embed(image)+'"></details>'
        for arm, label in (('kaggle', 'Kaggle'), ('published', 'npj · referência'), ('configured', 'npj · configurado')):
            c = records[(arm, name)]
            directory = results/'kaggle' if arm == 'kaggle' else results/'npj-profiles'/arm
            page += '<h3>'+label+'</h3><p>'+cell(c)+'</p>'
            diagnostic = c.get('error') or c.get('predicted_layout', '')
            if arm == 'kaggle':
                diagnostic += ' · última rede: Stage'+str(c.get('last_network_stage'))
            page += '<p><code>'+html.escape(diagnostic)+'</code></p>'
            wave = directory/(name+'-waves.svg')
            if wave.exists():
                page += '<details><summary>Comparar as 12 curvas · verde: original, laranja: saída</summary><img alt="Curvas em mV sem alinhamento" src="'+embed(wave)+'"></details>'
        page += '</article>'
    page += '''<div class="note"><h2>Para generalizar a abordagem do Kaggle</h2><p>Retificar pela grade sem depender de nomes em posições específicas, detectar um número variável de instâncias de derivações e treinar uma cabeça compartilhada por região. Preservar identidade, polaridade, baseline, janela de tempo e calibração física por instância. O novo decodificador do pacote implementa essa interface, mas ainda precisa dos campos previstos por detectores ou fornecidos por um perfil.</p>
<p>Para papéis desconhecidos, ainda faltam reconhecimento de velocidade/ganho e validação com ECGs reais independentes. Grade visível determina milímetros, mas não determina sozinha as unidades elétricas e temporais.</p>
<p>Fontes: <a href="https://github.com/Ahus-AIM/Open-ECG-Digitizer">código original npj</a> · <a href="https://www.nature.com/articles/s41746-025-02327-1">artigo npj Digital Medicine</a>. Código e pesos originais permanecem externos ao pacote. Relatórios JSON preservam falhas, hashes e saídas ausentes.</p></div></html>'''
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    build(args.inputs, args.results, args.out)
