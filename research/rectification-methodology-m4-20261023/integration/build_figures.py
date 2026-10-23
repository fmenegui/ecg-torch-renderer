"""Scientific figures from saved measurements, with no inference or tuning."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent/'visuals'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                     'axes.spines.top':False, 'axes.spines.right':False,
                     'savefig.facecolor':'white'})

def thin_trace():
    results = json.loads((ROOT/'m3/sampling/results/results.json').read_text())
    case = next(x for x in results['cases']
                if x['arm']=='native-bilinear-point' and x['complete_disappearance'])
    with np.load(ROOT/'m3/sampling/results'/f"{case['map']}.npz") as z:
        target = np.array([case['width_mm'],case['normal_angle_degrees'],case['offset_mm']])
        index = int(np.flatnonzero(np.isclose(z['scene_width_angle_offset'],target).all(1))[0])
        arms = z['arms'].tolist()
        panels = [z['source_gray_u8'][index],
                  np.uint8(255-z['darkness_u8'][index,arms.index('native-bilinear-point')]),
                  np.uint8(255-z['darkness_u8'][index,arms.index('area-bilinear-4')]),
                  255*(1-z['ideal_coverage_f64'][index])]
        # Crop the original around the same physical interval. This is a
        # visual illustration, never a new pixel-error measurement.
        source=panels[0]
        panels[0]=source[30:181,30:181]
    fig,axs=plt.subplots(1,4,figsize=(12,3.6),layout='constrained')
    for ax,panel,title in zip(axs,panels,
            ['Captura original (recorte)', 'Pontual: traço desaparece',
             'Área 4×4: traço retido', 'Referência analítica de área']):
        ax.imshow(panel,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
        ax.set_title(title,fontsize=10,pad=10); ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Geometria exata, traço de 0,1 mm, saída de 4 px/mm',fontsize=14)
    fig.text(.5,-.035,'Exemplo diagnóstico M3, escolhido por falha do braço pontual. '
             'Área retém energia, com menor pico e maior custo. Não é uma garantia geral.',
             ha='center',fontsize=9)
    fig.savefig(OUT/'thin-trace.png',dpi=160,bbox_inches='tight');plt.close(fig)
    (OUT/'thin-trace-provenance.json').write_text(json.dumps({
        'source':'M3 archived predictions, unchanged', 'case':case,
        'selection':'first archived native-point complete disappearance',
        'scope':'illustration only, no new quality score'},indent=2)+'\n')

def geometry():
    result=json.loads((ROOT/'m4/geometry/results-guard-v2/results.json').read_text())
    cases=result['cases']
    labels=[c['scene']['name'] for c in cases]
    methods=['fixed-H-LS','current-direct-M2-cells',
             'old-unregularized-composition-LS','selected-input-only']
    colors=['#8c939b','#bb735a','#a096ba','#167b86']
    fig,axs=plt.subplots(1,2,figsize=(13.5,8.3),sharey=True,layout='constrained',
                         gridspec_kw={'width_ratios':[1.3,1]})
    y=np.arange(len(cases))
    for mi,(method,color) in enumerate(zip(methods,colors)):
        vals=[c['methods'][method]['rmse_mm'] for c in cases]
        good=np.array([v is not None for v in vals])
        values=np.array([max(v,1e-7) if v is not None else np.nan for v in vals])
        axs[0].scatter(values[good],(y+(mi-1.5)*.15)[good],s=25,color=color,label=method)
    axs[0].set_xscale('log');axs[0].set_xlim(5e-8,100)
    axs[0].set_yticks(y,labels,fontsize=8.5);axs[0].invert_yaxis()
    axs[0].set_xlabel('RMSE em mm, condicional à cobertura (log)')
    axs[0].set_title('Todos os 19 controles geométricos')
    axs[0].grid(axis='x',alpha=.15)
    handles,legend_labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,fontsize=9,loc='lower center',
               bbox_to_anchor=(.5,-.09),ncol=2,frameon=False)
    axs[0].text(.01,-.085,'Valores < 10⁻⁷ mm no piso visual. Rejeição não tem RMSE.',
                transform=axs[0].transAxes,fontsize=8)
    for offset,method,color in [(-.16,'current-direct-M2-cells','#bb735a'),
                                 (.16,'selected-input-only','#167b86')]:
        g=np.array([c['methods'][method]['G'] for c in cases])*100
        w=np.array([c['methods'][method]['W'] for c in cases])*100
        axs[1].barh(y+offset,g,height=.28,color=color,alpha=.9)
        axs[1].barh(y+offset,w,left=g,height=.28,color=color,alpha=.3,hatch='///')
    axs[1].set_xlim(0,100);axs[1].set_xlabel('Domínio fixo: G sólido + W hachurado (%)')
    axs[1].set_title('Cobertura correta e errada, limiar 0,5 mm')
    axs[1].grid(axis='x',alpha=.15)
    fig.suptitle('Seleção por observações melhora vários casos e mantém falhas críticas',fontsize=15)
    fig.savefig(OUT/'geometry-all-cases.png',dpi=160,bbox_inches='tight');plt.close(fig)

if __name__=='__main__':
    thin_trace();geometry()
    print('Wrote two figures from archived arrays and scores.')
