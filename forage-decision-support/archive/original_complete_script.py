import unicodedata
import gower
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------
# 1. CARREGAMENTO E FUNÇÕES DE TRATAMENTO
# ---------------------------------------------------------


def normalizar_coluna(col):
    col = str(col).strip().lower()
    col = unicodedata.normalize("NFKD", col)
    col = "".join(c for c in col if not unicodedata.combining(c))
    return col.replace(" ", "_")


def limpar_num(val):
    if pd.isna(val):
        return np.nan
    permitido = "".join(c for c in str(val) if c.isdigit() or c in ",.-")
    try:
        return float(permitido.replace(",", "."))
    except:
        return np.nan


def limpar_texto(val):
    return str(val).strip().lower() if pd.notna(val) else ""


def valor_binario(val):
    return (
        1
        if str(val).strip().lower() in ["1", "1.0", "sim", "yes", "true", "x"]
        else 0
    )


def mapear_fertilidade(txt):
    txt = limpar_texto(txt)
    if txt.startswith("baix"):
        return 1
    elif txt.startswith("med") or txt.startswith("méd"):
        return 2
    elif txt.startswith("alt"):
        return 3
    return np.nan


def mapear_textura(txt):
    txt = limpar_texto(txt)
    if txt in ["", "na", "nan"]:
        return np.nan
    areia = ["aren", "leve"]
    media = ["media", "média", "franco", "medio"]
    argila = ["argil", "pesad"]
    valores = []
    if any(p in txt for p in areia):
        valores.append(1)
    if any(p in txt for p in media):
        valores.append(2)
    if any(p in txt for p in argila):
        valores.append(3)
    return np.mean(valores) if valores else np.nan


def classificar_estrato(altura):
    if pd.isna(altura):
        return np.nan
    if altura < 0.4:
        return 1
    elif altura < 1.2:
        return 2
    else:
        return 3


# Carregamento (Certifique-se de que o db_pastagem2.csv está no Colab)
caminho = "db_pastagem2.csv"
df = pd.read_csv(caminho)

# Normalização de colunas
df.columns = [normalizar_coluna(col) for col in df.columns]
df = df[df["especie"].notna()].copy()

# Reset de índice
df = df.reset_index(drop=True)
df["especie"] = df["especie"].astype(str).str.strip()

# Tratamento sem acento para o campo grupo
df["grupo"] = df["grupo"].apply(normalizar_coluna)

colunas_clima_total = [
    "cerrado",
    "tropical",
    "subtropical",
    "temperado",
    "subtropical_secundario",
    "subtropical_quente",
    "subtropical_de_altitude",
    "subtropical_umido",
    "subtropical_frio",
    "quente",
    "chuvoso",
    "semiarido",
    "subtropical_semiarido",
    "tropical_altitude",
    "estacao_fria",
    "quente_umido",
    "tropical_umido",
    "temperado_frio",
    "temperado_medio",
    "regioes_frias",
]

usos = [
    "pastagem",
    "pastejo",
    "feno",
    "silagem",
    "consorcio",
    "forragem",
    "forragem_ruminantes",
    "pastagem_consorciada",
    "pastejo_direto",
    "adubacao_verde",
]

for col in colunas_clima_total + usos:
    df[col] = df[col].apply(valor_binario) if col in df.columns else 0

df["uso_str"] = df.apply(
    lambda r: ", ".join([u for u in usos if r.get(u, 0) == 1]), axis=1
)

if "exigencia_fertilidade" in df.columns:
    df["fert_num"] = df["exigencia_fertilidade"].apply(mapear_fertilidade)
else:
    df["fert_num"] = np.nan

if "textura_solo" in df.columns:
    df["text_num"] = df["textura_solo"].apply(mapear_textura)
else:
    df["text_num"] = np.nan

cols_num = [
    "altura_min",
    "altura_max",
    "profundidade_raiz_max",
    "ph_min",
    "ph_max",
    "produtividade_minima_ms",
    "produtividade_maxima_ms",
]
for col in cols_num:
    if col in df.columns:
        df[col] = df[col].apply(limpar_num)

df["altura_media"] = (df.get("altura_min", 0) + df.get("altura_max", 0)) / 2

# Demais variáveis
df["produtividade_media"] = (
    df.get("produtividade_minima_ms", 0) + df.get("produtividade_maxima_ms", 0)
) / 2

df["produtividade_media"] = df["produtividade_media"].fillna(
    df["produtividade_media"].median()
)

max_prod = df["produtividade_media"].max()
df["prod_norm"] = df["produtividade_media"] / max_prod if max_prod > 0 else 0

# ============================================================
# DIAGNÓSTICO — PROFUNDIDADE RADICULAR ANTES DA IMPUTAÇÃO
# ============================================================

raiz_original = df["profundidade_raiz_max"].copy()

n_total = len(raiz_original)
n_validos = raiz_original.notna().sum()
n_ausentes = raiz_original.isna().sum()

perc_validos = 100 * n_validos / n_total
perc_ausentes = 100 * n_ausentes / n_total

mediana_raiz = raiz_original.median()

print("=" * 65)
print("DIAGNÓSTICO DA PROFUNDIDADE RADICULAR — ANTES DA IMPUTAÇÃO")
print("=" * 65)

print(f"Total de espécies: {n_total}")
print(f"Valores originais disponíveis: {n_validos} ({perc_validos:.1f}%)")
print(f"Valores ausentes: {n_ausentes} ({perc_ausentes:.1f}%)")
print(f"Mediana usada para imputação: {mediana_raiz:.2f}")

print("\nESTATÍSTICAS DOS VALORES ORIGINAIS:")
print(
    raiz_original.dropna().describe()
)

print("\nNÚMERO DE VALORES ÚNICOS ORIGINAIS:")
print(
    raiz_original.dropna().nunique()
)

print("\nVALORES MAIS FREQUENTES:")
print(
    raiz_original.dropna()
    .value_counts()
    .head(10)
)


# Imputação
features_solo = [
    "profundidade_raiz_max",
    "altura_media",
    "ph_min",
    "ph_max",
    "text_num",
    "fert_num",
]


# ============================================================
# DIAGNÓSTICO PARA ANÁLISE DE INCERTEZA
# NÃO ALTERA O DATASET
# Executar ANTES da imputação (fillna)
# ============================================================

import pandas as pd
import numpy as np

print("=" * 75)
print("1. DIAGNÓSTICO DE VALORES AUSENTES ANTES DA IMPUTAÇÃO")
print("=" * 75)

# Variáveis numéricas utilizadas pelo modelo
variaveis = [
    "profundidade_raiz_max",
    "altura_media",
    "ph_min",
    "ph_max",
    "text_num",
    "fert_num",
    "produtividade_media"
]

resultado_missing = []

for col in variaveis:
    if col in df.columns:
        total = len(df)
        ausentes = df[col].isna().sum()
        presentes = df[col].notna().sum()

        resultado_missing.append({
            "Variável": col,
            "Total": total,
            "Presentes": presentes,
            "Ausentes": ausentes,
            "Ausentes (%)": round(100 * ausentes / total, 2)
        })
    else:
        resultado_missing.append({
            "Variável": col,
            "Total": len(df),
            "Presentes": np.nan,
            "Ausentes": np.nan,
            "Ausentes (%)": np.nan
        })

diagnostico_missing = pd.DataFrame(resultado_missing)

print(diagnostico_missing.to_string(index=False))


# ============================================================
# 2. VERIFICAR COLUNAS RELACIONADAS À PRODUTIVIDADE
# ============================================================

print("\n" + "=" * 75)
print("2. COLUNAS RELACIONADAS À PRODUTIVIDADE")
print("=" * 75)

colunas_prod = [
    col for col in df.columns
    if (
        "prod" in col.lower()
        or "rend" in col.lower()
        or "yield" in col.lower()
    )
]

print("\nColunas encontradas:")

for col in colunas_prod:
    print(" -", col)


# ============================================================
# 3. MOSTRAR OS VALORES DAS COLUNAS DE PRODUTIVIDADE
# ============================================================

if len(colunas_prod) > 0:

    print("\n" + "=" * 75)
    print("3. AMOSTRA DOS DADOS DE PRODUTIVIDADE")
    print("=" * 75)

    cols_exibir = []

    if "especie" in df.columns:
        cols_exibir.append("especie")

    cols_exibir += colunas_prod

    print(
        df[cols_exibir]
        .head(30)
        .to_string(index=False)
    )


# ============================================================
# 4. ESTATÍSTICAS DA PRODUTIVIDADE MÉDIA
# ============================================================

print("\n" + "=" * 75)
print("4. PRODUTIVIDADE MÉDIA")
print("=" * 75)

if "produtividade_media" in df.columns:

    prod = df["produtividade_media"]

    print(f"Total: {len(prod)}")
    print(f"Presentes: {prod.notna().sum()}")
    print(f"Ausentes: {prod.isna().sum()}")
    print(
        f"Ausentes (%): "
        f"{100 * prod.isna().sum() / len(prod):.2f}%"
    )

    print("\nEstatísticas:")
    print(prod.dropna().describe())

    print("\nNúmero de valores únicos:")
    print(prod.dropna().nunique())

else:
    print("Coluna 'produtividade_media' não encontrada.")


# ============================================================
# 5. PROCURAR POSSÍVEIS COLUNAS DE LIMITES
# ============================================================

print("\n" + "=" * 75)
print("5. POSSÍVEIS COLUNAS DE LIMITES/FAIXAS")
print("=" * 75)

palavras = [
    "min",
    "max",
    "faixa",
    "range",
    "inferior",
    "superior"
]

colunas_faixa = [
    col for col in df.columns
    if any(p in col.lower() for p in palavras)
]

for col in colunas_faixa:
    print(" -", col)


# ============================================================
# 6. LISTA COMPLETA DAS COLUNAS
# ============================================================

print("\n" + "=" * 75)
print("6. TODAS AS COLUNAS DO DATASET")
print("=" * 75)

for i, col in enumerate(df.columns, start=1):
    print(f"{i:03d} - {col}")


for col in features_solo:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# ============================================================
# DIAGNÓSTICO — DEPOIS DA IMPUTAÇÃO
# ============================================================

print("\n" + "=" * 65)
print("PROFUNDIDADE RADICULAR — DEPOIS DA IMPUTAÇÃO")
print("=" * 65)

print(
    df["profundidade_raiz_max"].describe()
)

print(
    "\nNúmero de valores únicos após imputação:",
    df["profundidade_raiz_max"].nunique()
)

n_mediana = (
    df["profundidade_raiz_max"] == mediana_raiz
).sum()

print(
    f"Espécies com valor igual à mediana ({mediana_raiz:.2f}): "
    f"{n_mediana} de {len(df)} "
    f"({100*n_mediana/len(df):.1f}%)"
)


# Estrato numérico
df["estrato_num"] = df["altura_media"].apply(classificar_estrato)

# Mapeia a posição original do índice para busca na Matriz Gower
df["pos_matriz"] = df.index

# ---------------------------------------------------------
# 2. AGRUPAMENTO HIERÁRQUICO (HAC + GOWER + SILHOUETTE)
# ---------------------------------------------------------
features_cluster = features_solo + ["prod_norm"] + colunas_clima_total
df_cluster = df[features_cluster].copy()

# Distância Gower
matriz_distancia = gower.gower_matrix(df_cluster)

melhor_k = 2
melhor_sil = -1
melhor_labels = None
historico_silhouette = []

for k in range(2, 7):
    hac = AgglomerativeClustering(
        n_clusters=k, metric="precomputed", linkage="complete"
    )
    labels = hac.fit_predict(matriz_distancia)
    sil = silhouette_score(matriz_distancia, labels, metric="precomputed")
    historico_silhouette.append((k, sil))
    if sil > melhor_sil:
        melhor_k = k
        melhor_sil = sil
        melhor_labels = labels

df["cluster_ia"] = melhor_labels

# ---------------------------------------------------------
# 3. REGRAS, NORMALIZAÇÃO E SCORES (AJUSTADOS PARA O REVISOR)
# ---------------------------------------------------------


def categorizar_ciclo(val):
    txt = limpar_texto(val)
    if any(
        p in txt for p in ["peren", "pluri", "long", "semi-peren", "semiperen"]
    ):
        return "perene"
    elif any(p in txt for p in ["anual", "bienal", "curt", "temp", "estacion"]):
        return "anual_bienal"
    return "desconhecido"


def e_especie_perene_pastejavel(row):
    ciclo_cat = categorizar_ciclo(row.get("ciclo", ""))
    if ciclo_cat != "perene":
        return False
    apto_pastejo = (
        row.get("pastejo", 0) == 1
        or row.get("pastagem", 0) == 1
        or row.get("consorcio", 0) == 1
        or row.get("forragem", 0) == 1
        or row.get("forragem_ruminantes", 0) == 1
        or row.get("pastagem_consorciada", 0) == 1
        or row.get("pastejo_direto", 0) == 1
    )
    return apto_pastejo


def normalize_min_max(val, val_min, val_max):
    """Normaliza um valor bruto para a escala [0, 1]."""
    if val_max == val_min:
        return 0.0
    norm = (val - val_min) / (val_max - val_min)
    return max(0.0, min(1.0, norm))


def calcular_score_normalizado(
    row, ph, clima, text, fert, w_eco=0.6, w_prod=0.4
):
    """Calcula o Score com sub-componentes em [0, 1] e estima a contribuição (%)"""
    # 1. Checa aptidão climática (Binário)
    if clima == "cerrado":
        aptidao_clima = (
            row.get("cerrado", 0) == 1
            or row.get("tropical", 0) == 1
            or row.get("quente", 0) == 1
            or row.get("tropical_umido", 0) == 1
        )
    elif clima in ["tropical", "quente"]:
        aptidao_clima = (
            row.get("tropical", 0) == 1
            or row.get("cerrado", 0) == 1
            or row.get("quente", 0) == 1
        )
    else:
        aptidao_clima = row.get(clima, 0) == 1

    if not aptidao_clima:
        return 0.0, {}

    # Checa tolerância de pH
    #if ph < (row["ph_min"] - 0.2) or ph > (row["ph_max"] + 0.2):
        #return 0.0, {}

    # 2. Scores Brutos

    #centro_ph = (row["ph_min"] + row["ph_max"]) / 2
    #raw_ph = max(25.0 - (abs(ph - centro_ph) * 20.0), -25.0)

    # =========================================================
    # pH — FUNÇÃO TRAPEZOIDAL
    # =========================================================

    ph_min = row["ph_min"]
    ph_max = row["ph_max"]

    delta_ph = 0.2

    # Fora da faixa ampliada -> espécie inviável
    if ph < (ph_min - delta_ph) or ph > (ph_max + delta_ph):
        return 0.0, {}

    # Dentro da faixa nominal -> adequação máxima
    if ph_min <= ph <= ph_max:

        ph_norm = 1.0

    # Margem inferior
    elif (ph_min - delta_ph) <= ph < ph_min:

        ph_norm = (
            ph - (ph_min - delta_ph)
        ) / delta_ph

    # Margem superior
    elif ph_max < ph <= (ph_max + delta_ph):

       ph_norm = (
            (ph_max + delta_ph) - ph
        ) / delta_ph

    else:

        ph_norm = 0.0

    ph_norm = float(
       np.clip(
            ph_norm,
            0.0,
            1.0
          )
    )

    # 2. Scores Brutos


    textura_alvo = mapear_textura(text)
    raw_text = (
        15.0
        if (
            pd.notna(textura_alvo)
            and abs(row["text_num"] - textura_alvo) <= 1
        )
        else -10.0
    )

    fertilidade_alvo = mapear_fertilidade(fert)
    raw_fert = (
        15.0
        if (
            pd.notna(fertilidade_alvo)
            and abs(row["fert_num"] - fertilidade_alvo) <= 1
        )
        else -10.0
    )

    raw_uso = 20.0 if "pastagem" in limpar_texto(row["uso_str"]) else 5.0
    raw_raiz = min(row["profundidade_raiz_max"] / 20.0, 10.0)

    # 3. Normalização [0, 1] dos componentes ecológicos
    norm_scores = {
        "ph": ph_norm,
        "textura": normalize_min_max(raw_text, -10, 15),
        "fertilidade": normalize_min_max(raw_fert, -10, 15),
        "uso": normalize_min_max(raw_uso, 5, 20),
        "raiz": normalize_min_max(raw_raiz, 0, 10),
    }

    # 4. Pesos de S_eco
    eco_weights = {
        "ph": 1.1,
        "textura": 1.0,
        "fertilidade": 1.0,
        "uso": 0.8,
        "raiz": 1.0,
    }
    sum_eco_weights = sum(eco_weights.values())  # 4.9

    # S_eco em escala [0, 1]
    s_eco_norm = (
        sum(eco_weights[k] * norm_scores[k] for k in eco_weights)
        / sum_eco_weights
    )

    # 5. Componente de Produtividade S_prod em [0, 1]
    s_prod_norm = float(
    np.clip(
        row["prod_norm"],
        0.0,
        1.0
    )
)
    # 6. Score Final S_i em [0, 100]
    S_i_norm = (w_eco * s_eco_norm) + (w_prod * s_prod_norm)
    score_final = S_i_norm * 100.0

    # 7. Contribuição Efetiva (%) de cada critério para o Score Final
    contrib = {}
    if S_i_norm > 0:
        for k in eco_weights:
            part = w_eco * (eco_weights[k] * norm_scores[k] / sum_eco_weights)
            contrib[k] = round((part / S_i_norm) * 100.0, 1)
        contrib["produtividade"] = round(
            ((w_prod * s_prod_norm) / S_i_norm) * 100.0, 1
        )
    else:
        contrib = {k: 0.0 for k in list(eco_weights.keys()) + ["produtividade"]}

    return round(score_final, 2), contrib


def score_consorcio_avancado(g, l, d_gower_pair):
    base_score = 0.40 * g["score"] + 0.40 * l["score"]
    diff_raiz = abs(g["profundidade_raiz_max"] - l["profundidade_raiz_max"])
    comp_raiz = min(diff_raiz / 10.0, 10.0)

    diff_estrato = abs(g["estrato_num"] - l["estrato_num"])
    comp_estrato = 10.0 if diff_estrato >= 1 else 3.0

    compat_gower_multivariada = round(10.0 * (1.0 - d_gower_pair), 2)
    return round(
        base_score + comp_raiz + comp_estrato + compat_gower_multivariada, 2
    )


# ---------------------------------------------------------
# 4. RECOMENDAÇÃO COMPLETA COM REPORTAGEM DE CONTRIBUIÇÕES
# ---------------------------------------------------------

################################################################################
def recomendar(ph=5.0, clima="cerrado", text="argiloso", fert="media"):
    # Aplica o cálculo normalizado
    res = df.apply(
        lambda r: calcular_score_normalizado(r, ph, clima, text, fert), axis=1
    )
    df["score"] = [r[0] for r in res]
    #df["contrib_pH"] = [r[1].get("ph", 0.0) for r in res]
    #df["contrib_Prod"] = [r[1].get("produtividade", 0.0) for r in res]
    #df["contrib_Clima"] = [r[1].get("clima", 0.0) for r in res]
    #df["contrib_Textura"] = [r[1].get("textura", 0.0) for r in res]

    df["contrib_pH"] = [r[1].get("ph", 0.0)for r in res]
    df["contrib_Textura"] = [r[1].get("textura", 0.0)for r in res]
    df["contrib_Fertilidade"] = [r[1].get("fertilidade", 0.0)for r in res]
    df["contrib_Uso"] = [r[1].get("uso", 0.0)for r in res]
    df["contrib_Raiz"] = [r[1].get("raiz", 0.0)for r in res]
    df["contrib_Prod"] = [r[1].get("produtividade", 0.0)for r in res]

    df_aptas = df[df.apply(e_especie_perene_pastejavel, axis=1)].copy()

    # Seleção dos TOP 10 com as novas colunas de contribuição efetiva (%) solicitadas
    cols_exibir = [
        "especie",
        "grupo",
        "score",
        "contrib_pH",
        "contrib_Textura",
        "contrib_Fertilidade",
        "contrib_Uso",
        "contrib_Raiz",
        "contrib_Prod"

      ]
    top_10_mono = df_aptas.sort_values("score", ascending=False).head(10)[
        cols_exibir
    ]
    top_10_mono.columns = [
        "Espécie / Cultivar",
    "Grupo",
    "Score",
    "pH (%)",
    "Texture (%)",
    "Fertility (%)",
    "Agronomic use (%)",
    "Root depth (%)",
    "Productivity (%)"

    ]

    gramineas = df_aptas[
        df_aptas["grupo"].str.contains("gramin", na=False)
    ].sort_values("score", ascending=False)

    if len(gramineas) == 0:
        print("Nenhuma gramínea perene atendeu aos critérios ecológicos.")
        return

    max_g_score = gramineas.iloc[0]["score"]
    gramineas_candidatas = gramineas[gramineas["score"] >= 0.95 * max_g_score]

    leguminosas = df_aptas[df_aptas["grupo"].str.contains("legumin", na=False)]

    consorcios = []
    for _, g in gramineas_candidatas.iterrows():
        for _, l in leguminosas.iterrows():
            if l["score"] == 0:
                continue

            pos_g = int(g["pos_matriz"])
            pos_l = int(l["pos_matriz"])
            d_gower_pair = matriz_distancia[pos_g, pos_l]

            consorcios.append(
                {
                    "Gramínea (Principal)": g["especie"],
                    "Leguminosa (Acompanhante)": l["especie"],
                    "Score Consórcio": score_consorcio_avancado(
                        g, l, d_gower_pair
                    ),
                }
            )

    if consorcios:
        df_cons = (
            pd.DataFrame(consorcios)
            .sort_values("Score Consórcio", ascending=False)
            .head(10)
        )
    else:
        df_cons = pd.DataFrame(
            columns=[
                "Gramínea (Principal)",
                "Leguminosa (Acompanhante)",
                "Score Consórcio",
            ]
        )

    print(
        "======================================================================="
    )
    print("SISTEMA HÍBRIDO DE RECOMENDAÇÃO: MONOCULTURA vs. CONSÓRCIOS")
    print(f"Condições: pH={ph} | Clima={clima} | Solo={text} | Fert={fert}")
    print(
        f"Número Ótimo de Clusters (HAC): k={melhor_k} | Coef. Silhouette:"
        f" {melhor_sil:.4f}"
    )
    print(
        "=======================================================================\n"
    )

    print(
        "--- 1. TOP 10 ESPÉCIES INDIVIDUAIS COM CONTRIBUIÇÃO EFETIVA (%) ---"
    )
    print(top_10_mono.to_string(index=False))

    print(
        "\n--- 2. TOP 10 CONSÓRCIOS RECOMENDADOS (Gramíneas Threshold >= 95%:"
        f" {len(gramineas_candidatas)} avaliadas) ---"
    )
    print(df_cons.to_string(index=False))


if __name__ == "__main__":
    #recomendar(ph=5.0, clima="cerrado", text="argiloso", fert="media")
    #recomendar(ph=4.5, clima="tropical", text="arenoso", fert="baixa")
    recomendar(ph=6.0, clima="subtropical", text="arenoso", fert="baixa")


from scipy.stats import spearmanr
# =========================================================
# ANÁLISE DE SENSIBILIDADE:
# PESO ECOLÓGICO vs. PRODUTIVIDADE
# =========================================================

cenarios_pesos = {
    "70/30": (0.70, 0.30),
    "60/40": (0.60, 0.40),  # baseline
    "50/50": (0.50, 0.50)
}


def gerar_ranking_pesos(w_eco, w_prod):

    df_teste = df.copy()

    resultados = df_teste.apply(
        lambda r: calcular_score_normalizado(
            r,
            ph=5.0,
            clima="cerrado",
            text="argiloso",
            fert="media",
            w_eco=w_eco,
            w_prod=w_prod
        )[0],
        axis=1
    )

    df_teste["score_teste"] = resultados

    # Mantém exatamente o mesmo filtro de elegibilidade
    df_teste = df_teste[
        df_teste.apply(
            e_especie_perene_pastejavel,
            axis=1
        )
    ].copy()

    ranking = (
        df_teste[
            ["especie", "grupo", "score_teste"]
        ]
        .sort_values(
            "score_teste",
            ascending=False
        )
        .reset_index(drop=True)
    )

    ranking["rank"] = (
        np.arange(len(ranking)) + 1
    )

    return ranking


# ---------------------------------------------------------
# GERA OS TRÊS RANKINGS
# ---------------------------------------------------------

rankings_pesos = {}

for nome, (w_eco, w_prod) in cenarios_pesos.items():

    rankings_pesos[nome] = gerar_ranking_pesos(
        w_eco,
        w_prod
    )


# ---------------------------------------------------------
# BASELINE = 60/40
# ---------------------------------------------------------

baseline = rankings_pesos["60/40"]


# ---------------------------------------------------------
# COMPARAÇÃO COM O BASELINE
# ---------------------------------------------------------

resultados_sensibilidade = []

for nome, ranking in rankings_pesos.items():

    comp = baseline[
        ["especie", "rank"]
    ].merge(
        ranking[
            ["especie", "rank"]
        ],
        on="especie",
        suffixes=("_base", "_teste")
    )

    rho, p_value = spearmanr(
        comp["rank_base"],
        comp["rank_teste"]
    )

    top10_base = set(
        baseline.head(10)["especie"]
    )

    top10_teste = set(
        ranking.head(10)["especie"]
    )

    overlap = len(
        top10_base & top10_teste
    )

    resultados_sensibilidade.append({
        "Cenário": nome,
        "Peso ecológico": cenarios_pesos[nome][0],
        "Peso produtividade": cenarios_pesos[nome][1],
        "Spearman rho": rho,
        "p-value": p_value,
        "Top-10 overlap": overlap
    })




df_sensibilidade = pd.DataFrame(
    resultados_sensibilidade
)


print(
    "\n============================================================"
)
print(
    "ANÁLISE DE SENSIBILIDADE — ECOLÓGICO vs. PRODUTIVIDADE"
)
print(
    "Baseline: 60% ecológico / 40% produtividade"
)
print(
    "============================================================"
)

print(
    df_sensibilidade.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# MOSTRA TOP 10 DE CADA CENÁRIO
# ---------------------------------------------------------

for nome, ranking in rankings_pesos.items():

    print(
        f"\n--- TOP 10 — CENÁRIO {nome} ---"
    )

    print(
        ranking.head(10)[
            [
                "rank",
                "especie",
                "grupo",
                "score_teste"
            ]
        ].to_string(
            index=False
        )
    )