import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

tabela = pd.read_csv("subset.csv")
rotulos = tabela["target"].values

nome_coluna_aneis = ""
for coluna in tabela.columns:
    if "ringsE" in coluna:
        nome_coluna_aneis = coluna
        break


def texto_para_array(s):
    return np.fromstring(str(s).strip("[]"), sep=",")


energia_aneis = np.vstack(tabela[nome_coluna_aneis].apply(texto_para_array).values)


def split_estratificado(dados_entrada, rotulos_entrada, fracao_conjunto_b, semente=42):
    gerador_aleatorio = np.random.default_rng(semente)
    indices_conjunto_a, indices_conjunto_b = [], []
    for classe_rotulo in np.unique(rotulos_entrada):
        indices_da_classe = np.where(rotulos_entrada == classe_rotulo)[0]
        gerador_aleatorio.shuffle(indices_da_classe)
        ponto_de_corte = int(len(indices_da_classe) * (1 - fracao_conjunto_b))
        indices_conjunto_a.extend(indices_da_classe[:ponto_de_corte])
        indices_conjunto_b.extend(indices_da_classe[ponto_de_corte:])
    indices_conjunto_a = np.array(indices_conjunto_a)
    indices_conjunto_b = np.array(indices_conjunto_b)
    gerador_aleatorio.shuffle(indices_conjunto_a)
    gerador_aleatorio.shuffle(indices_conjunto_b)
    return (dados_entrada[indices_conjunto_a], dados_entrada[indices_conjunto_b],
            rotulos_entrada[indices_conjunto_a], rotulos_entrada[indices_conjunto_b])


aneis_treino, aneis_temp, rotulos_treino, rotulos_temp = split_estratificado(energia_aneis, rotulos, 0.4)
aneis_val, aneis_teste, rotulos_val, rotulos_teste = split_estratificado(aneis_temp, rotulos_temp, 0.5)

media = aneis_treino.mean(axis=0)
desvio = aneis_treino.std(axis=0) + 1e-8
aneis_treino_normalizado = (aneis_treino - media) / desvio
aneis_val_normalizado = (aneis_val - media) / desvio
aneis_teste_normalizado = (aneis_teste - media) / desvio

dataset_treino = TensorDataset(torch.tensor(aneis_treino_normalizado, dtype=torch.float32),
                                torch.tensor(rotulos_treino, dtype=torch.float32).unsqueeze(1))
dataset_val = TensorDataset(torch.tensor(aneis_val_normalizado, dtype=torch.float32),
                             torch.tensor(rotulos_val, dtype=torch.float32).unsqueeze(1))
dataset_teste = TensorDataset(torch.tensor(aneis_teste_normalizado, dtype=torch.float32),
                               torch.tensor(rotulos_teste, dtype=torch.float32).unsqueeze(1))

batches_treino = DataLoader(dataset_treino, batch_size=256, shuffle=True)
batches_val = DataLoader(dataset_val, batch_size=256, shuffle=False)
batches_teste = DataLoader(dataset_teste, batch_size=256, shuffle=False)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Linear(100, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.rede(x)


modelo = MLP()
criterio = nn.BCELoss()
otimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)

historico_loss_treino = []
historico_loss_val = []
melhor_loss_val = float("inf")
epocas_sem_melhora = 0
melhores_pesos = None
paciencia = 5

for epoca in range(1, 101):
    modelo.train()
    perdas_treino = []
    for aneis_batch, rotulos_batch in batches_treino:
        otimizador.zero_grad()
        loss = criterio(modelo(aneis_batch), rotulos_batch)
        loss.backward()
        otimizador.step()
        perdas_treino.append(loss.item())

    modelo.eval()
    perdas_val = []
    with torch.no_grad():
        for aneis_batch, rotulos_batch in batches_val:
            loss = criterio(modelo(aneis_batch), rotulos_batch)
            perdas_val.append(loss.item())

    loss_treino_medio = np.mean(perdas_treino)
    loss_val_medio = np.mean(perdas_val)
    historico_loss_treino.append(loss_treino_medio)
    historico_loss_val.append(loss_val_medio)

    print(f"Época {epoca:>3} | Loss treino: {loss_treino_medio:.4f} | Loss val: {loss_val_medio:.4f}")

    if loss_val_medio < melhor_loss_val:
        melhor_loss_val = loss_val_medio
        epocas_sem_melhora = 0
        melhores_pesos = {nome: peso.clone() for nome, peso in modelo.state_dict().items()}
    else:
        epocas_sem_melhora += 1

    if epocas_sem_melhora == paciencia:
        break

modelo.load_state_dict(melhores_pesos)
torch.save(modelo.state_dict(), "mlp_modelo1_sem_kfold.pt")

plt.figure(figsize=(7, 5))
plt.plot(historico_loss_treino, label="Treino")
plt.plot(historico_loss_val, label="Validação")
plt.xlabel("Época")
plt.ylabel("Loss (BCE)")
plt.title("Modelo 1 - Curva de Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("modelo1_curva_loss.png", dpi=150)
plt.close()


def curva_roc_manual(rotulos_verdadeiros, probabilidades_preditas):
    limiares_decisao = np.concatenate(([np.inf], np.sort(np.unique(probabilidades_preditas))[::-1]))
    total_positivos = np.sum(rotulos_verdadeiros == 1)
    total_negativos = np.sum(rotulos_verdadeiros == 0)
    taxas_verdadeiro_positivo, taxas_falso_positivo = [], []
    for limiar in limiares_decisao:
        predicao_no_limiar = (probabilidades_preditas >= limiar).astype(int)
        verdadeiros_positivos = np.sum((predicao_no_limiar == 1) & (rotulos_verdadeiros == 1))
        falsos_positivos = np.sum((predicao_no_limiar == 1) & (rotulos_verdadeiros == 0))
        taxas_verdadeiro_positivo.append(verdadeiros_positivos / total_positivos if total_positivos > 0 else 0.0)
        taxas_falso_positivo.append(falsos_positivos / total_negativos if total_negativos > 0 else 0.0)
    return np.array(taxas_falso_positivo), np.array(taxas_verdadeiro_positivo)


def avaliar_modelo(modelo_avaliado, batches_avaliacao):
    modelo_avaliado.eval()
    probabilidades, rotulos_reais = [], []
    with torch.no_grad():
        for aneis_batch, rotulos_batch in batches_avaliacao:
            saida = modelo_avaliado(aneis_batch)
            probabilidades.extend(saida.squeeze(1).tolist())
            rotulos_reais.extend(rotulos_batch.squeeze(1).tolist())
    probabilidades = np.array(probabilidades)
    rotulos_reais = np.array(rotulos_reais)
    predicoes_binarias = (probabilidades >= 0.5).astype(int)

    verdadeiros_positivos = np.sum((predicoes_binarias == 1) & (rotulos_reais == 1))
    verdadeiros_negativos = np.sum((predicoes_binarias == 0) & (rotulos_reais == 0))
    falsos_positivos = np.sum((predicoes_binarias == 1) & (rotulos_reais == 0))
    falsos_negativos = np.sum((predicoes_binarias == 0) & (rotulos_reais == 1))
    matriz_confusao = np.array([[verdadeiros_negativos, falsos_positivos],
                                 [falsos_negativos, verdadeiros_positivos]])

    acuracia = (verdadeiros_positivos + verdadeiros_negativos) / len(rotulos_reais)
    probabilidade_deteccao = verdadeiros_positivos / (verdadeiros_positivos + falsos_negativos) \
        if (verdadeiros_positivos + falsos_negativos) > 0 else 0.0
    falso_alarme = falsos_positivos / (falsos_positivos + verdadeiros_negativos) \
        if (falsos_positivos + verdadeiros_negativos) > 0 else 0.0

    taxa_falsos_positivos, taxa_verdadeiros_positivos = curva_roc_manual(rotulos_reais, probabilidades)
    area_sob_curva = np.trapezoid(taxa_verdadeiros_positivos, taxa_falsos_positivos)

    return (acuracia, probabilidade_deteccao, falso_alarme, area_sob_curva,
            matriz_confusao, taxa_falsos_positivos, taxa_verdadeiros_positivos)


(acuracia_val, probabilidade_deteccao_val, falso_alarme_val, area_sob_curva_val,
 matriz_confusao_val, _, _) = avaliar_modelo(modelo, batches_val)

(acuracia_teste, probabilidade_deteccao_teste, falso_alarme_teste, area_sob_curva_teste,
 matriz_confusao_teste, taxa_falsos_positivos_teste, taxa_verdadeiros_positivos_teste) = avaliar_modelo(modelo, batches_teste)

print("\n=== Resultados ===")
print("\nValidação:")
print(f"  ACURACIA: {acuracia_val*100:.2f}%")
print(f"  PD: {probabilidade_deteccao_val*100:.2f}%")
print(f"  FA: {falso_alarme_val*100:.2f}%")
print(f"  AUC: {area_sob_curva_val:.4f}")

print("\nTeste:")
print(f"  ACURACIA: {acuracia_teste*100:.2f}%")
print(f"  PD: {probabilidade_deteccao_teste*100:.2f}%")
print(f"  FA: {falso_alarme_teste*100:.2f}%")
print(f"  AUC: {area_sob_curva_teste:.4f}")

print("\nMatriz de Confusão (Teste):")
print(matriz_confusao_teste)

plt.figure(figsize=(5, 4))
plt.imshow(matriz_confusao_teste, cmap="Blues")
for linha in range(2):
    for coluna in range(2):
        plt.text(coluna, linha, str(matriz_confusao_teste[linha, coluna]), ha="center", va="center", fontsize=14)
plt.xticks([0, 1], ["Ruído (0)", "Sinal (1)"])
plt.yticks([0, 1], ["Ruído (0)", "Sinal (1)"])
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Modelo 1 - Matriz de Confusão")
plt.colorbar()
plt.tight_layout()
plt.savefig("modelo1_matriz_confusao.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 5))
plt.plot(taxa_falsos_positivos_teste, taxa_verdadeiros_positivos_teste,
         label=f"Modelo 1 (AUC = {area_sob_curva_teste:.3f})", color="tab:orange")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Chance Aleatória")
plt.fill_between(taxa_falsos_positivos_teste, taxa_verdadeiros_positivos_teste, alpha=0.2, color="tab:orange")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Modelo 1 - Curva ROC")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("modelo1_curva_roc.png", dpi=150)
plt.close()
