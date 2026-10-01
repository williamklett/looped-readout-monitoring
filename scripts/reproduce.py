"""Rebuild figures from the released fixed evidence, without model inference."""
from pathlib import Path
import subprocess,sys,os
root=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(root/'build/matplotlib'))
for folder in ['paper/native-readout-draft/figures','paper/native-readout-chess-revision/figures']:
 (root/folder).mkdir(parents=True,exist_ok=True)
for script in ['scripts/verify_results.py','paper/native-readout-draft/build_stopping_control.py','paper/native-readout-draft/build_figures.py','paper/native-readout-draft/build_wide_figures.py','scripts/build_paper_arithmetic_wide.py','scripts/build_chess_primary_paper.py']:
 subprocess.run([sys.executable,script],cwd=root,check=True)
print('Reproduced all main figures and arithmetic supplementary figures.')
