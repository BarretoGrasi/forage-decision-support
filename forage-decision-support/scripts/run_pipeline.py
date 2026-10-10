"""Run in an isolated output folder, without Colab globals or magics."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
from figures import generate_figures

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='results/latest')
    parser.add_argument('--full-clustering',action='store_true',help='Run the expensive exploratory method comparison; requires optional dependencies.')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];out=Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Output directory is nonempty. Choose a fresh directory to avoid mixing executions.')
    out.mkdir(parents=True,exist_ok=True)
    for name in ('db_pastagem2_original.csv','correcoes_curadoria.csv'):
        shutil.copy2(root/'data'/name,out/name)
    previous=Path.cwd()
    try:
        os.chdir(out)
        with (out/'execution.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log):
            runpy.run_path(str(root/'scripts/reconciliar_xaraes.py'),init_globals={'DIRETORIO_EXECUCAO':out})
            state=runpy.run_path(str(root/'scripts/recomendador_reconciliado_xaraes_mg5.py'),init_globals={'EXECUTAR_CLUSTERING_COMPLETO':args.full_clustering})
            summary=generate_figures(state,out/'figures')
        summary['versions']={p:importlib.metadata.version(p) for p in ('numpy','pandas','scipy','scikit-learn','gower','matplotlib')}
        summary['input_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'data').glob('*.csv')}
        (out/'run_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(summary,ensure_ascii=False,indent=2));print(f'Results: {out}')
    finally:
        os.chdir(previous)

if __name__=='__main__':
    main()
