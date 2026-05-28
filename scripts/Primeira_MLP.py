import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
 
 
tabela = pd.read_csv("subset.csv")
 
rotulos = tabela["target"].values
 
nome_coluna_aneis = ""
for coluna in tabela.columns:
    if "ringsE" in coluna:
        nome_coluna_aneis = coluna
        break
 
def texto_para_array(s):
    vals = np.fromstring(str(s).strip("[]"), sep=",")
    return vals
 
energia_aneis = np.vstack(tabela[nome_coluna_aneis].apply(texto_para_array).values)
 
corte = int(len(rotulos) * 0.8)
 
aneis_treino,    aneis_teste    = energia_aneis[:corte], energia_aneis[corte:]
rotulos_treino,  rotulos_teste  = rotulos[:corte],       rotulos[corte:]
 
media   = aneis_treino.mean(axis=0)
desvio  = aneis_treino.std(axis=0) + 1e-8
aneis_treino_normalizado = (aneis_treino - media) / desvio
aneis_teste_normalizado  = (aneis_teste  - media) / desvio
 
dataset_treino = TensorDataset(torch.tensor(aneis_treino_normalizado, dtype=torch.float32), torch.tensor(rotulos_treino, dtype=torch.float32).unsqueeze(1))
dataset_teste  = TensorDataset(torch.tensor(aneis_teste_normalizado,  dtype=torch.float32), torch.tensor(rotulos_teste,  dtype=torch.float32).unsqueeze(1))
 
 
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Linear(100, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        return self.rede(x)
 
 
modelo     = MLP()
criterio   = nn.BCELoss()
otimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)
 
batches_treino = DataLoader(dataset_treino, batch_size=256, shuffle=True)
batches_teste  = DataLoader(dataset_teste,  batch_size=256, shuffle=False)
 
melhor_acuracia    = 0
epocas_sem_melhora = 0
melhores_pesos     = None
 
for epoca in range(1, 51):
    modelo.train()
    perdas = []
    for aneis_batch, rotulos_batch in batches_treino:
        otimizador.zero_grad()
        loss = criterio(modelo(aneis_batch), rotulos_batch)
        loss.backward()
        otimizador.step()
        perdas.append(loss.item())
 
    modelo.eval()
    corretos, total = 0, 0
    with torch.no_grad():
        for aneis_batch, rotulos_batch in batches_teste:
            predicao = modelo(aneis_batch)
            corretos += ((predicao >= 0.5).float() == rotulos_batch).sum().item()
            total    += rotulos_batch.size(0)
 
    acuracia = corretos / total
    print(f"Época {epoca:>3} | Loss: {np.mean(perdas):.4f} | Acurácia: {acuracia*100:.2f}%")
 
    if acuracia > melhor_acuracia:
        melhor_acuracia    = acuracia
        epocas_sem_melhora = 0
        melhores_pesos = {nome: peso.clone() for nome, peso in modelo.state_dict().items()}
    else:
        epocas_sem_melhora += 1
 
    if epocas_sem_melhora == 5:
        break
 
modelo.load_state_dict(melhores_pesos)

torch.save(modelo.state_dict(), "mlp_primeira_tentativa.pt")
