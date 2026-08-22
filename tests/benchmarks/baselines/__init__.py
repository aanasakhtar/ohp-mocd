"""
baselines package — Authentic Published Baseline Algorithms:
1. SLPA (Xie & Szymanski, IEEE TKDE 2011/2012)
2. MCMOEA (Wen et al., IEEE TEVC 2016)
3. Çetin & Amrahov (Kybernetika, 2022)
4. LPAM (Ponomarenko et al., PLOS ONE, 2021)
5. NOCD (Shchur & Günnemann, KDD / ICLR, 2019)
"""

from tests.benchmarks.baselines.slpa import run_slpa
from tests.benchmarks.baselines.lpam import run_lpam
from tests.benchmarks.baselines.nocd import run_nocd
from tests.benchmarks.baselines.cetin import run_cetin
