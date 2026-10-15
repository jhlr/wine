"""Código do Lab 2 (Deep Learning, CESAR School).

Cada módulo cuida de uma parte do enunciado. O notebook ``lab02.ipynb`` chama
estas funções e explica o raciocínio; a lógica avaliada mora aqui.
"""
from lab02lib.data import MeuDataset
from lab02lib.split import particionar_sem_vazar
from lab02lib.train import treinar_com_parada

__all__ = ["MeuDataset", "particionar_sem_vazar", "treinar_com_parada"]
