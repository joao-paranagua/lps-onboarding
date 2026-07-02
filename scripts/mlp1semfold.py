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


def split_estratificado(X, y, frac_teste, seed=42):
    rng = np.random.default_rng(seed)
    idx_a, idx_b = [], []
    for classe in np.unique(y):
        idx = np.where(y == classe)[0]
        rng.shuffle(idx)
        corte = int(len(idx) * (1 - frac_teste))
        idx_a.extend(idx[:corte])
        idx_b.extend(idx[corte:])
    idx_a, idx_b = np.array(idx_a), np.array(idx_b)
    rng.shuffle(idx_a)
    rng.shuffle(idx_b)
    return X[idx_a], X[idx_b], y[idx_a], y[idx_b]


aneis_treino, aneis_temp, rotulos_treino, rotulos_temp = split_estratificado(energia_aneis, rotulos, 0.4)
aneis_val, aneis_teste, rotulos_val, rotulos_teste = split_estratificado(aneis_temp, rotulos_temp, 0.5)

media = aneis_treino.mean(axis=0)
desvio = aneis_treino.std(axis=0) + 1e-8
aneis_treino_norm = (aneis_treino - media) / desvio
aneis_val_norm = (aneis_val - media) / desvio
aneis_teste_norm = (aneis_teste - media) / desvio

dataset_treino = TensorDataset(torch.tensor(aneis_treino_norm, dtype=torch.float32),
                                torch.tensor(rotulos_treino, dtype=torch.float32).unsqueeze(1))
dataset_val = TensorDataset(torch.tensor(aneis_val_norm, dtype=torch.float32),
                             torch.tensor(rotulos_val, dtype=torch.float32).unsqueeze(1))
dataset_teste = TensorDataset(torch.tensor(aneis_teste_norm, dtype=torch.float32),
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


def roc_curve_manual(y_true, y_score):
    limiares = np.concatenate(([np.inf], np.sort(np.unique(y_score))[::-1]))
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    tprs, fprs = [], []
    for t in limiares:
        pred = (y_score >= t).astype(int)
        tp_ = np.sum((pred == 1) & (y_true == 1))
        fp_ = np.sum((pred == 1) & (y_true == 0))
        tprs.append(tp_ / P if P > 0 else 0.0)
        fprs.append(fp_ / N if N > 0 else 0.0)
    return np.array(fprs), np.array(tprs)


def avaliar(modelo, batches):
    modelo.eval()
    probs, reais = [], []
    with torch.no_grad():
        for aneis_batch, rotulos_batch in batches:
            saida = modelo(aneis_batch)
            probs.extend(saida.squeeze(1).tolist())
            reais.extend(rotulos_batch.squeeze(1).tolist())
    probs = np.array(probs)
    reais = np.array(reais)
    preds = (probs >= 0.5).astype(int)
    tp = np.sum((preds == 1) & (reais == 1))
    tn = np.sum((preds == 0) & (reais == 0))
    fp = np.sum((preds == 1) & (reais == 0))
    fn = np.sum((preds == 0) & (reais == 1))
    cm = np.array([[tn, fp], [fn, tp]])
    acc = (tp + tn) / (tp + tn + fp + fn)
    pd_ = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fa_ = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fpr, tpr = roc_curve_manual(reais, probs)
    auc_ = np.trapezoid(tpr, fpr)
    return acc, pd_, fa_, auc_, cm, fpr, tpr, probs, reais


acc_val, pd_val, fa_val, auc_val, cm_val, _, _, _, _ = avaliar(modelo, batches_val)
acc_teste, pd_teste, fa_teste, auc_teste, matriz_confusao, fpr, tpr, probs_teste, rotulos_teste_arr = avaliar(modelo, batches_teste)
area_sob_curva = auc_teste

print("\n=== Resultados ===")
print("\nValidação:")
print(f"  ACURACIA: {acc_val*100:.2f}%")
print(f"  PD: {pd_val*100:.2f}%")
print(f"  FA: {fa_val*100:.2f}%")
print(f"  AUC: {auc_val:.4f}")

print("\nTeste:")
print(f"  ACURACIA: {acc_teste*100:.2f}%")
print(f"  PD: {pd_teste*100:.2f}%")
print(f"  FA: {fa_teste*100:.2f}%")
print(f"  AUC: {auc_teste:.4f}")

print("\nMatriz de Confusão (Teste):")
print(matriz_confusao)

plt.figure(figsize=(5, 4))
plt.imshow(matriz_confusao, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(matriz_confusao[i, j]), ha="center", va="center", fontsize=14)
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
plt.plot(fpr, tpr, label=f"Modelo 1 (AUC = {area_sob_curva:.3f})", color="tab:orange")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Chance Aleatória")
plt.fill_between(fpr, tpr, alpha=0.2, color="tab:orange")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Modelo 1 - Curva ROC")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("modelo1_curva_roc.png", dpi=150)
plt.close()
