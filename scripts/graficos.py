import pandas as pd
import matplotlib.pyplot as plt

DATASET = "subset.csv"

df = pd.read_csv(DATASET)

plt.hist(df["ElectronContainer.et"], bins=128, color="cyan", edgecolor="black")
plt.title(r"$E_T$")
plt.xlabel("Valor")
plt.ylabel("Frequência")
plt.show()

plt.hist(df["ElectronContainer.eta"], bins=128, color="orange", edgecolor="black")
plt.title(r"$\eta$")
plt.xlabel("Valor")
plt.ylabel("Frequência")
plt.show()

plt.hist(df["ElectronContainer.phi"], bins=128, color="limegreen", edgecolor="black")
plt.title(r"$\phi$")
plt.xlabel("Valor")
plt.ylabel("Frequência")
plt.show()

plt.hist(df["EventInfoContainer.avgmu"], bins=128, color="mediumpurple", edgecolor="black")
plt.title(r"$\langle \mu \rangle$")
plt.xlabel("Valor")
plt.ylabel("Frequência")
plt.show()
