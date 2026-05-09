import pandas as pd
from pretraitement import pretraitement

input_csv = input("Entrez le chemin du fichier CSV (ex: IMDB Dataset.csv) : ").strip()
output_arff = input("Entrez le chemin de sortie ARFF (ex: imdb.arff) : ").strip()

df = pd.read_csv(input_csv)

print("Prétraitement...")
df["review"] = df["review"].apply(pretraitement)

with open(output_arff, "w", encoding="utf-8") as f:
    f.write("@relation imdb\n\n")
    f.write("@attribute review string\n")
    f.write("@attribute sentiment {positive,negative}\n\n")
    f.write("@data\n")

    for _, row in df.iterrows():
        text = str(row["review"]).replace('"', '\\"')
        f.write(f'"{text}",{row["sentiment"]}\n')

print("Done:", output_arff)
