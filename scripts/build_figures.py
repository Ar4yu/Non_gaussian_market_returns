"""Build reviewer figures from the frozen project evidence."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.reproduce import verify_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'docs/assets')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    returns = verify_data()['QQQ'].dropna()
    summary = pd.read_csv(ROOT / 'outputs/summary_qqq.csv').set_index('model')
    bt = pd.read_csv(ROOT / 'outputs/var_backtest_qqq.csv')
    colors = {'Normal': '#c16b37', 'Student-t': '#087f8c'}
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
        'axes.spines.top':False, 'axes.spines.right':False, 'text.color':'#172b4d',
        'axes.labelcolor':'#172b4d', 'axes.titleweight':'bold'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(top=.72, bottom=.22, wspace=.25)
    fig.suptitle('A better tail model does not win at every threshold', x=.08, y=.98,
                 ha='left', fontsize=20, fontweight='bold')
    fig.text(.08,.88,'QQQ held-out sample: 803 trading days, October 6, 2022 to December 17, 2025',fontsize=11)
    for ax, alpha in zip(axes,[.01,.05]):
        subset = bt[bt.alpha == alpha]
        bars=ax.bar(subset.model, subset.exceed_rate*100, color=[colors[m] for m in subset.model], width=.55)
        ax.axhline(alpha*100, color='#172b4d', ls='--', label=f'Expected: {alpha:.0%}')
        for bar, row in zip(bars,subset.itertuples()):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.12,
                    f'{row.exceed_rate:.2%}\n{row.exceed_count} breaches',ha='center',fontsize=11, bbox=dict(facecolor='white',edgecolor='none',pad=2))
        ax.set_title(f'{alpha:.0%} left-tail probability',loc='left',pad=14)
        ax.set_ylim(0, max(subset.exceed_rate*100)*1.35)
        ax.set_ylabel('Observed exceedance rate (%)')
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.legend(frameon=False)
    fig.text(.08,.055,'Fixed parameters estimated on training data. Student-t uses df = 3.\nAt 1%, coverage does not reject Student-t, but independence rejects at 5% significance.',fontsize=10,linespacing=1.5)
    fig.savefig(args.output_dir/'var-calibration.png',dpi=180,facecolor='white');plt.close(fig)

    fig, axes=plt.subplots(1,2,figsize=(12,5))
    fig.subplots_adjust(top=.72,bottom=.22,wspace=.28)
    fig.suptitle('Heavy tails improve the distributional fit',x=.08,y=.98,ha='left',fontsize=20,fontweight='bold')
    fig.text(.08,.88,'QQQ daily log returns. Training: 3,211 observations through October 5, 2022.',fontsize=11)
    train=returns.loc[:'2022-10-05']
    grid=np.linspace(-.08,.08,900)
    axes[0].hist(train,bins=90,density=True,color='#dce4ec',label='Training returns')
    for name,row in summary.iterrows():
        distribution=stats.norm(loc=row.mu_hat,scale=row.sigma_hat) if name=='Normal' else stats.t(row.df_hat,loc=row.mu_hat,scale=row.sigma_hat)
        axes[0].plot(grid,distribution.pdf(grid),color=colors[name],lw=2,label=name)
        axes[1].plot(grid,distribution.cdf(grid),color=colors[name],lw=2,label=name)
    xs=np.sort(train.to_numpy())
    axes[1].step(xs,np.arange(1,len(xs)+1)/len(xs),color='#172b4d',lw=1,label='Empirical training CDF')
    axes[0].set(xlim=(-.065,.065),xlabel='Daily log return',ylabel='Density',title='Center and fitted densities')
    axes[1].set(xlim=(-.065,-.005),ylim=(0,.15),xlabel='Daily log return',ylabel='Cumulative probability',title='Left-tail distribution')
    for ax in axes:
        ax.xaxis.set_major_formatter(PercentFormatter(1));ax.legend(frameon=False,fontsize=9);ax.grid(alpha=.15)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    fig.text(.08,.055,'Student-t test log-likelihood: 2,385.22 versus 2,332.03 for Normal.\nThe fitted t scale differs from its standard deviation. Model fitting uses training observations only.',fontsize=10,linespacing=1.5)
    fig.savefig(args.output_dir/'distribution-fit.png',dpi=180,facecolor='white');plt.close(fig)
    print('Wrote two figures from frozen results.')


if __name__ == '__main__':
    main()
