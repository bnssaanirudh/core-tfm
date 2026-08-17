from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

regs=['Uniform low noise','Marginals reliable','Conditionals reliable','$J_1$ reliable','$J_2$ reliable','All views noisy']
values={
'Selected order':[0.004827,0.040880,0.034547,0.000000,0.000000,0.062733],
'Arithmetic':[0.000002,0.014434,0.000000,0.027255,0.027504,0.000000],
'Hard CoRe':[0.010954,0.000635,0.142265,0.101272,0.094534,0.125168],
'Adaptive Soft':[0.001443,0.002658,0.003439,0.000540,0.000276,0.003150],
'Soft lambda=10':[0.0065,0.000000,0.113,0.076,0.067,0.106],
}
x=np.arange(len(regs)); width=.15
fig,ax=plt.subplots(figsize=(8.2,3.8))
for i,(name,vals) in enumerate(values.items()):
    ax.bar(x+(i-2)*width, vals, width, label=name)
ax.axhline(0,linewidth=.8)
ax.set_ylabel('Expected NLL regret vs. best method')
ax.set_xticks(x); ax.set_xticklabels(regs,rotation=20,ha='right')
ax.legend(ncol=3,fontsize=8,frameon=False); ax.set_ylim(bottom=0)
fig.tight_layout()
Path('figures').mkdir(exist_ok=True)
fig.savefig('figures/exact_view_regime_regret.pdf',bbox_inches='tight')
fig.savefig('figures/exact_view_regime_regret.png',dpi=220,bbox_inches='tight')
