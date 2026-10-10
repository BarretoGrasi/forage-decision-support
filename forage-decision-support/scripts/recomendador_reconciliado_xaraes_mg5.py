# ============================================================
# SISTEMA DE RECOMENDAÇÃO DE FORRAGEIRAS — v3.8.1 + pendência documental de pH (2026-10-08)
# RESTRIÇÕES OBRIGATÓRIAS + ESTADOS DE INFORMAÇÃO + PARETO
# ============================================================
# Entrada única: db_pastagem2.csv (mesma pasta do script)
#
# PRINCÍPIOS DA v3
#  1. Separar AUSÊNCIA DE INFORMAÇÃO de INAPTIDÃO.
#     - Colunas binárias (usos, clima) funcionam como checklist:
#       "1" = evidência positiva; "0"/vazio = SEM INFORMAÇÃO.
#     - Atributos ordinais vazios ficam NaN (nada de "neutro" silencioso).
#  2. Restrições OBRIGATÓRIAS (não compensatórias) definem o status:
#       apto       -> atende a todas, com dados
#       verificar  -> não viola nenhuma, mas falta dado obrigatório
#       excluido   -> viola ao menos uma (motivo registrado)
#  3. Critérios COMPENSATÓRIOS formam o score (só ordenam aptas/verificar).
#  4. Correção do solo é PARÂMETRO DO CENÁRIO (calagem=True/False).
#  5. Fertilidade só PENALIZA (decisão metodológica, out/2026).
#  6. O Pareto é etapa POSTERIOR de comparação entre pares não excluídos.
#  7. Casos de teste derivados da avaliação externa (shortlist).
#
# MUDANÇAS DA v3.8.1 (só interpretação no relatório; regras e cálculos inalterados)
#  - Par fora do conjunto apto: 'Pareto-eficiente' = 'não avaliado' (não
#    participa da fronteira; isso NÃO significa que seja dominado).
#  - Base do score do par: 'completa' ou 'parcial (condicional)', quando falta
#    componente da complementaridade ou critério individual (cobertura < 1).
#    Score parcial não é comparável diretamente com scores completos.
#  - 'Não dominado por par apto' deixa de sugerir eficiência quando os
#    objetivos são parciais.
#  - Candidatos condicionados: removida a afirmação de que 'entrariam na
#    fronteira'; são apenas não dominados pelos aptos com os dados disponíveis.
#
# MUDANÇAS DA v3.8 (comparação de estratégias de seleção; regras inalteradas)
#  - Seção 11b: compara três estratégias por cenário:
#      (a) referência INDIVIDUAL: melhor gramínea apta + melhor leguminosa
#          apta do arranjo, avaliadas como par pelas MESMAS regras; se o par
#          for excluído ou 'verificar', isso é registrado (sem substituição);
#      (b) ranking DIRETO: todos os pares aptos ordenados pelo score ponderado;
#      (c) PÓS-PARETO: fronteira dos pares aptos ordenada pelo mesmo score.
#    Shortlist de tamanho fixo definido a priori (K_SHORTLIST = 5) para (b) e
#    (c); a referência individual (1 par) é comparada separadamente.
#  - Empates explícitos: pos_score (empate = mesma posição, método 'min') x
#    pos_apresentacao (desempate só para exibição: objetivos, depois nomes).
#    Empate na fronteira do shortlist é sinalizado.
#  - Avaliação do par centralizada em avaliar_par_completo() (usada por
#    montar_pares e pela referência individual).
#
# MUDANÇAS DA v3.7.1 (só relatório; nenhuma regra alterada)
#  - Pendência de nível de pastejo sem ';' interno (contagem de faltantes correta).
#  - Sensibilidade do limite: coluna de exclusões em que R3-nível é o ÚNICO motivo.
#  - DataFrame desfragmentado antes da imputação (remove PerformanceWarning).
#
# MUDANÇAS DA v3.7
#  - R3: nível de tolerância DESCONHECIDO (tolerancia_pastejo='sim' ou só
#    flag de uso pastejo/pastejo_direto) não passa mais pelo filtro sem
#    comparação. Política explícita POLITICA_NIVEL_DESCONHECIDO:
#      'verificar'   (padrão) -> espécie 'verificar' (coerente com mínimo obrigatório)
#      'aceitar_uso'          -> apta com ressalva 'uso documentado; nível desconhecido'
#    Aptidão de USO e INTENSIDADE de tolerância passam a ser distinguidas.
#  - Limite mínimo de tolerância como parâmetro do cenário (tol_min_graminea,
#    tol_min_leguminosa) e ANÁLISE DE SENSIBILIDADE do limite da gramínea
#    ('media' x 'alta' x 'muito alta'), listando as gramíneas cujo status muda.
#    O limite 'alta' é hipótese metodológica: requer justificativa independente.
#  - Casos de teste condicionados à calagem: expectativa de acidez só vale
#    sem calagem; casos que não se aplicam ficam 'N/A'; repetições entre
#    cenários não contam como avaliações independentes.
#  - Cobertura (critérios avaliados) exibida em TODOS os rankings individuais.
#
# MUDANÇAS DA v3.6
#  - R3 (manejo) alterada na REGRA, não só no teste: em todos os cenários
#    (todos são de uso sob pastejo), a GRAMÍNEA principal precisa de
#    tolerância a pastejo >= TOL_PASTEJO_MIN_GRAMINEA ('alta'). Tolerância
#    'média' passa a EXCLUIR a gramínea (antes: apta com ressalva).
#    Para leguminosas o mínimo é TOL_PASTEJO_MIN_LEGUMINOSA ('media'),
#    mantido como parâmetro explícito.
#    Não depende da curadoria: usa o dado já presente na base.
#  - Comentário da seção 2b corrigido (suspensão ocorre ANTES da imputação).
#
# MUDANÇAS DA v3.5
#  - Suspensão de intervalos invertidos ANTES da imputação e da Gower:
#    os valores inválidos não entram no KNN nem em nenhuma representação.
#    Na Gower, a célula suspensa recebe estimativa KNN feita só com vizinhos
#    válidos; no modelo, ela volta a NaN (a imputação não apaga a pendência).
#    Valores originais preservados em <coluna>_original_invertido (auditoria).
#  - Critério compensatório sem dado = 'NÃO AVALIADO': sai da média ponderada
#    (pesos renormalizados), em vez de receber 0,5. Colunas novas:
#    criterios_nao_avaliados e cobertura_score (fração do peso avaliada).
#    O mesmo vale para a complementaridade do par (média dos componentes
#    disponíveis; par 'verificar').
#    Escolha metodológica explícita: aptidão individual NÃO implica dados
#    suficientes para avaliar a complementaridade do consórcio.
#  - Escala da raiz calculada só com valores válidos e impressa na saída
#    (corrigir a base altera a normalização de TODAS as espécies).
#  - Correções de curadoria via correcoes_curadoria.csv: aplicadas só com
#    valor novo E fonte preenchidos; log em log_correcoes_curadoria.csv.
#  - Casos de teste exigem o MOTIVO esperado: PASSA / PARCIAL / FALHA.
#  - Capacidade de suporte (UA/ha) desligada por padrão (LER 1,52 não
#    validado para os pares avaliados).
#
# MUDANÇAS DA v3.4
#  - Verificação de INTERVALOS INVERTIDOS (mínimo > máximo) em todos os pares
#    min/max da base. Os valores NÃO são invertidos automaticamente: o
#    atributo afetado é SUSPENSO (tratado como dado inconsistente) até a
#    fonte ser conferida e o CSV corrigido.
#      raiz      -> profundidade_raiz_max suspensa; critério não avaliado
#                   (v3.5) e par 'verificar'
#      altura    -> altura/estrato suspensos; porte 'indefinido'; par 'verificar'
#      pH        -> faixa suspensa; R4 sem dado -> 'verificar'
#      precip.   -> R7 sem dado -> 'verificar'
#      produtiv. -> sinalizada (média não depende da ordem)
#      demais    -> apenas relatadas (não usadas no modelo)
#    Relatório exportado em intervalos_invertidos.csv.
#  - Correção: estrato ausente não é mais comparado como se fosse inferior.
#
# MUDANÇAS DA v3.3
#  - R7 (obrigatória): precipitação anual do cenário >= precipitacao_min da
#    espécie. Sem precipitacao_min -> 'verificar'. Cenário sem precipitação
#    definida (None) -> R7 desativada e sinalizada no relatório.
#    Cerrado/Campos Belos: 1.249 mm/ano (Climatempo, normal climatológica).
#  - Porte: hábitos trepador/volúvel/prostrado/rasteiro/estolonífero/decumbente
#    são sempre herbáceos (a altura de trepadeira depende do suporte);
#    limiar de altura passa a ser estrito (> ALTURA_ARBUSTIVA).
#
# MUDANÇAS DA v3.2
#  - ARRANJO do par como parâmetro do cenário:
#      'consorcio_pastejo' -> só leguminosas herbáceas (rasteiras, estoloníferas,
#                             eretas/semieretas, subarbustos);
#      'banco_proteina'    -> só leguminosas arbustivas/arbóreas (banco de
#                             proteína / silvipastoril), em cenário próprio.
#    Motivo: a complementaridade funcional cresce com o contraste de raiz e
#    estrato, o que favorecia sistematicamente leucena e guandu.
#  - Leguminosa sem hábito e sem altura -> par 'verificar'.
#  - UA/ha potencial calculada só para consórcio em pastejo (o LER de
#    referência vem de consórcio Cynodon + A. pintoi).
#
# MUDANÇAS DA v3.1
#  - Consórcio: TODAS as gramíneas aptas são candidatas (sem limiar de 95%).
#  - Rankings, líderes e fronteira de Pareto: SOMENTE espécies/pares 'apto'.
#  - Espécies/pares 'verificar' listados à parte (candidatos condicionados).
#  - Relatório de possíveis linhas duplicadas da mesma espécie.
#  - UA/ha rotulada como POTENCIAL e opcional (CALCULAR_UA).
#  - Sensibilidade ao conjunto de gramíneas (top-k) no lugar do threshold.
#  - Pareto vetorizado (numpy).
#
# RESTRIÇÕES OBRIGATÓRIAS
#  R0  Entrada genérica não concorre quando há cultivares cadastradas
#  R1  Ciclo perene
#  R2  Aptidão climática ao cenário
#  R3  Aptidão a pastejo (tolerancia_pastejo; capineira sem aptidão = exclui)
#  R4  pH do cenário (após calagem, se houver) dentro da faixa ± DELTA_PH
#  R5  Sem calagem e pH <= PH_LIMIAR_AL: tolerância ao Al >= moderada
#  R6  (par) Espécie do estrato inferior deve tolerar sombreamento
#  R7  Precipitação anual do cenário >= precipitacao_min da espécie
#
# CRITÉRIOS COMPENSATÓRIOS (S_eco) + produtividade (S_prod)
#  pH (trapezoidal), Al (graduado, só quando R5 está ativa), textura,
#  fertilidade (penalização assimétrica), tolerância a pastejo (graduada),
#  profundidade radicular.
# ============================================================

import io
import re
import unicodedata
from contextlib import redirect_stdout
from pathlib import Path

import gower
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import KNNImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# 0. CONFIGURAÇÃO
# ---------------------------------------------------------
CAMINHO_CSV = Path("db_pastagem2.csv")

PH_LIMIAR_AL = 5.5      # sem calagem e pH <= limiar -> restrição de Al ativa
PH_POS_CALAGEM = 5.8    # pH assumido após calagem
DELTA_PH = 0.2          # margem da faixa de pH
# R3 — tolerância mínima a pastejo direto (níveis: baixa < media < alta < muito alta)
TOL_PASTEJO_MIN_GRAMINEA = "alta"      # gramínea principal do pasto
TOL_PASTEJO_MIN_LEGUMINOSA = "media"   # leguminosa acompanhante
POLITICA_NIVEL_DESCONHECIDO = "verificar"  # "verificar" | "aceitar_uso" (ver cabeçalho)
LIMITES_SENSIBILIDADE_PASTEJO = ("media", "alta", "muito alta")
CALCULAR_UA = False                     # UA/ha potencial: LER 1,52 não validado para os pares
CAMINHO_CORRECOES = Path("correcoes_curadoria.csv")  # correções com fonte (opcional)
TOP_K_SENSIBILIDADE = (3, 5, 10, None)  # None = todas as gramíneas aptas
K_SHORTLIST = 5                         # tamanho do shortlist, fixado ANTES de ver os resultados
ALTURA_ARBUSTIVA = 2.0                  # m; altura_max > limiar -> arbustiva/arbórea
PRECIP_CAMPOS_BELOS = 1249              # mm/ano (Climatempo; seca de maio a setembro)
RODAR_COMPARACAO_CLUSTERING = bool(globals().get("EXECUTAR_CLUSTERING_COMPLETO", False))

# Sinônimos usados apenas para AGRUPAR cultivares da mesma espécie
# (não alteram o nome exibido; corrija a nomenclatura no CSV).
SINONIMOS = {
    "panicum maximum": "megathyrsus maximus",
    "pennisetum purpureum": "cenchrus purpureus",
}

CENARIOS = {
    "Baseline (Cerrado, pH 5.0, sem calagem)": dict(
        ph=5.0, clima="cerrado", text="argiloso", fert="media", calagem=False,
        arranjo="consorcio_pastejo", precipitacao=PRECIP_CAMPOS_BELOS,
    ),
    "Baseline (Cerrado, pH 5.0, com calagem)": dict(
        ph=5.0, clima="cerrado", text="argiloso", fert="media", calagem=True,
        arranjo="consorcio_pastejo", precipitacao=PRECIP_CAMPOS_BELOS,
    ),
    "Tropical ácido (pH 4.5, sem calagem)": dict(
        ph=4.5, clima="tropical", text="arenoso", fert="baixa", calagem=False,
        arranjo="consorcio_pastejo", precipitacao=None,  # definir com dado local
    ),
    "Subtropical (pH 6.0)": dict(
        ph=6.0, clima="subtropical", text="arenoso", fert="baixa", calagem=False,
        arranjo="consorcio_pastejo", precipitacao=None,  # definir com dado local
    ),
    "Banco de proteína/silvipastoril (Cerrado, pH 5.0, sem calagem)": dict(
        ph=5.0, clima="cerrado", text="argiloso", fert="media", calagem=False,
        arranjo="banco_proteina", precipitacao=PRECIP_CAMPOS_BELOS,
    ),
}
ARRANJOS_VALIDOS = {"consorcio_pastejo", "banco_proteina"}
CENARIO_REF = "Baseline (Cerrado, pH 5.0, sem calagem)"

# ---------------------------------------------------------
# 1. FUNÇÕES DE TRATAMENTO
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
    except Exception:
        return np.nan


def limpar_texto(val):
    return str(val).strip().lower() if pd.notna(val) else ""


def evidencia_positiva(val):
    """1 quando a célula afirma o atributo; 0 = sem evidência (NÃO significa 'não')."""
    return 1 if str(val).strip().lower() in ["1", "1.0", "sim", "yes", "true", "x"] else 0


def mapear_fertilidade(txt):
    t = limpar_texto(txt)
    if t.startswith(("muito baix", "baix")):
        return 1
    if t.startswith(("med", "méd")):
        return 2
    if t.startswith(("muito alt", "alt")):
        return 3
    return np.nan


def mapear_textura(txt):
    t = limpar_texto(txt)
    if t in ["", "na", "nan"]:
        return np.nan
    valores = []
    if any(p in t for p in ["aren", "leve"]):
        valores.append(1)
    if any(p in t for p in ["media", "média", "franco", "medio"]):
        valores.append(2)
    if any(p in t for p in ["argil", "pesad"]):
        valores.append(3)
    return np.mean(valores) if valores else np.nan


def mapear_nivel(txt):
    """Tolerâncias ordinais -> 1 (baixa), 1.5 (baixa-moderada), 2 (moderada), 3 (alta).
    Vazio ou ambíguo ('N', 'na') -> NaN (sem informação)."""
    t = limpar_texto(txt)
    if t in ["", "nan", "n", "na", "-"]:
        return np.nan
    if t.startswith(("baixa-moder", "baixa a moder", "baixa/moder")):
        return 1.5
    if t.startswith(("baix", "muito baix", "nao", "não", "intoler", "sensiv", "sensív")):
        return 1.0
    if t.startswith(("med", "méd", "moder", "razoav", "razoáv", "regul")):
        return 2.0
    if t.startswith(("muito alt", "alt", "toler", "boa")):
        return 3.0
    return np.nan


def categorizar_ciclo(val):
    t = limpar_texto(val)
    if any(p in t for p in ["peren", "pluri", "long", "semi-peren", "semiperen"]):
        return "perene"
    if any(p in t for p in ["anual", "bienal", "curt", "temp", "estacion"]):
        return "anual_bienal"
    return "desconhecido"


def classificar_estrato(altura):
    if pd.isna(altura):
        return np.nan
    if altura < 0.4:
        return 1
    if altura < 1.2:
        return 2
    return 3


def chave_binomio(nome):
    partes = [p for p in re.split(r"[\s(]+", limpar_texto(nome)) if p]
    binomio = " ".join(partes[:2])
    return SINONIMOS.get(binomio, binomio)


def e_cultivar(nome):
    return bool(re.search(r"\bcv\b|cv\.|\(|\bbrs\b|tifton", limpar_texto(nome)))


def e_generico_genero(nome):
    return bool(re.search(r"\bspp?\b", limpar_texto(nome))) and not e_cultivar(nome)


ORDEM_STATUS = {"apto": 0, "verificar": 1, "excluido": 2}


def pior_status(*status):
    return max(status, key=lambda s: ORDEM_STATUS[s])


# Pendência documental explicitamente identificada na curadoria.
# Não significa intolerância biológica. Retirar somente após documentar fonte,
# interpretação do limite e método de medição do pH.
def pendencia_ph_documental(row):
    nome = limpar_texto(row.get("especie", ""))
    if chave_binomio(nome) == "megathyrsus maximus" and ("tanzania" in nome or "tanzânia" in nome):
        return "limite inferior de pH de Tanzânia não confirmado documentalmente"
    return ""

# ---------------------------------------------------------
# 2. CARREGAMENTO
# ---------------------------------------------------------
if not CAMINHO_CSV.exists():
    raise FileNotFoundError(
        "O arquivo 'db_pastagem2.csv' não foi encontrado. Coloque o CSV na mesma "
        "pasta do script ou faça upload dele para o Colab antes de executar."
    )

df = pd.read_csv(CAMINHO_CSV)
df.columns = [normalizar_coluna(c) for c in df.columns]
df = df[df["especie"].notna()].copy().reset_index(drop=True)
df["especie"] = df["especie"].astype(str).str.strip()
df["grupo"] = df["grupo"].apply(normalizar_coluna)

# ---------------------------------------------------------
# 2a. CORREÇÕES DE CURADORIA (com fonte obrigatória)
# ---------------------------------------------------------
# Arquivo opcional correcoes_curadoria.csv com colunas:
#   especie, coluna, valor_novo, fonte, justificativa
# Só é aplicada a linha com valor_novo E fonte preenchidos. As demais ficam
# como pendentes no relatório. Nada é corrigido sem referência.
_log_cur = []
if CAMINHO_CORRECOES.exists():
    _cur = pd.read_csv(CAMINHO_CORRECOES, dtype=str).fillna("")
    _cur.columns = [normalizar_coluna(c) for c in _cur.columns]
    for _, c in _cur.iterrows():
        esp, col = c["especie"].strip(), normalizar_coluna(c["coluna"])
        val, fonte = c["valor_novo"].strip(), c["fonte"].strip()
        mask = df["especie"].eq(esp)
        if not mask.any():
            situacao, anterior = "espécie não encontrada", ""
        elif col not in df.columns:
            situacao, anterior = "coluna não encontrada", ""
        elif not val or not fonte:
            situacao = "PENDENTE (falta " + " e ".join(
                x for x, ok in [("valor_novo", val), ("fonte", fonte)] if not ok) + ")"
            anterior = str(df.loc[mask, col].iloc[0])
        else:
            anterior = str(df.loc[mask, col].iloc[0])
            df[col] = df[col].astype(object)
            df.loc[mask, col] = val
            situacao = "APLICADA"
        _log_cur.append({"especie": esp, "coluna": col, "valor_anterior": anterior,
                         "valor_novo": val, "fonte": fonte,
                         "justificativa": c.get("justificativa", ""), "situacao": situacao})
    print("=" * 90)
    print(f"CORREÇÕES DE CURADORIA ({CAMINHO_CORRECOES})")
    print("=" * 90)
    _df_log = pd.DataFrame(_log_cur)
    print(_df_log[["especie", "coluna", "valor_anterior", "valor_novo", "situacao"]].to_string(index=False))
    _df_log.to_csv("log_correcoes_curadoria.csv", index=False, encoding="utf-8-sig")
else:
    print(f"(Sem arquivo de curadoria: {CAMINHO_CORRECOES} — base usada como está.)")


colunas_clima_total = [
    "cerrado", "tropical", "subtropical", "temperado", "subtropical_secundario",
    "subtropical_quente", "subtropical_de_altitude", "subtropical_umido",
    "subtropical_frio", "quente", "chuvoso", "semiarido", "subtropical_semiarido",
    "tropical_altitude", "estacao_fria", "quente_umido", "tropical_umido",
    "temperado_frio", "temperado_medio", "regioes_frias",
]
colunas_uso_binarias = [
    "pastagem", "pastejo", "feno", "silagem", "consorcio", "forragem",
    "forragem_ruminantes", "pastagem_consorciada", "pastejo_direto",
    "adubacao_verde", "capineiras", "corte_verde",
]

# Evidência positiva (0 = sem evidência)
for col in colunas_clima_total + colunas_uso_binarias:
    df[col] = df[col].apply(evidencia_positiva) if col in df.columns else 0

df["tem_info_clima"] = df[colunas_clima_total].sum(axis=1) > 0

# Atributos ordinais — NaN preservado
for col in ["tolerancia_pastejo", "tolerancia_sombra", "tolerancia_aluminio", "exigencia_fertilidade"]:
    if col not in df.columns:
        df[col] = np.nan

df["al_num"] = df["tolerancia_aluminio"].apply(mapear_nivel)
df["sombra_num"] = df["tolerancia_sombra"].apply(mapear_nivel)
df["fert_num"] = df["exigencia_fertilidade"].apply(mapear_fertilidade)
df["text_num"] = df["textura_solo"].apply(mapear_textura) if "textura_solo" in df.columns else np.nan

cols_num = [
    "altura_min", "altura_max", "profundidade_raiz_max", "ph_min", "ph_max",
    "produtividade_minima_ms", "produtividade_maxima_ms", "precipitacao_min", "precipitacao_max",
    "profundidade_raiz_min", "temp_min", "temp_max", "proteina_min", "proteina_max",
    "biomassa_min", "biomassa_max",
]
for col in cols_num:
    df[col] = df[col].apply(limpar_num) if col in df.columns else np.nan

# Valores preservados antes da imputação; não apagar a evidência original.
df["ph_min_antes_imputacao"] = df["ph_min"].copy()
df["ph_max_antes_imputacao"] = df["ph_max"].copy()
df["pendencia_ph_documental"] = df.apply(pendencia_ph_documental, axis=1)
_mask_ph_documental = df["pendencia_ph_documental"].ne("")
if _mask_ph_documental.any():
    print("PENDÊNCIA DOCUMENTAL DE pH — valor preservado; elegibilidade condicionada")
    print(df.loc[_mask_ph_documental,
                 ["id", "especie", "ph_min_antes_imputacao",
                  "ph_max_antes_imputacao", "pendencia_ph_documental"]].to_string(index=False))

df["altura_media"] = df[["altura_min", "altura_max"]].mean(axis=1)

df["produtividade_media"] = df[["produtividade_minima_ms", "produtividade_maxima_ms"]].mean(axis=1)
df["prod_imputada"] = df["produtividade_media"].isna()
df["produtividade_media"] = df["produtividade_media"].fillna(df["produtividade_media"].median())
max_prod = df["produtividade_media"].max()
df["prod_norm"] = df["produtividade_media"] / max_prod if max_prod > 0 else 0

# Identidade taxonômica (para genéricos e coerência entre cultivares)
df["binomio"] = df["especie"].apply(chave_binomio)
df["genero"] = df["binomio"].str.split().str[0]
df["eh_cultivar"] = df["especie"].apply(e_cultivar)
df["eh_generico_genero"] = df["especie"].apply(e_generico_genero)

_cult = df[df["eh_cultivar"]]
_binomios_com_cultivar = set(_cult["binomio"])
_generos_com_cultivar = set(_cult["genero"])


def _generico_substituivel(row):
    if row["eh_cultivar"]:
        return False
    if row["eh_generico_genero"]:
        return row["genero"] in _generos_com_cultivar
    return row["binomio"] in _binomios_com_cultivar


df["generico_substituivel"] = df.apply(_generico_substituivel, axis=1)


# ---------------------------------------------------------
# 2b. VERIFICAÇÃO DE INTERVALOS INVERTIDOS (mínimo > máximo)
# ---------------------------------------------------------
# Não inverte valores: registra e SUSPENDE os atributos que o modelo usa
# ANTES da imputação e da construção das representações (Gower), preservando
# o valor original e o indicador de erro para auditoria. Corrigir no CSV
# (ou na curadoria) após conferir a fonte.
PARES_INTERVALO = [
    # (coluna mín., coluna máx., rótulo, colunas suspensas, consequência no modelo)
    ("profundidade_raiz_min", "profundidade_raiz_max", "raiz",
     ["profundidade_raiz_max"], "raiz suspensa: critério não avaliado no score; par 'verificar'"),
    ("altura_min", "altura_max", "altura",
     ["altura_min", "altura_max", "altura_media"], "altura/estrato suspensos: porte 'indefinido'; componente de estrato não avaliado; par 'verificar'"),
    ("ph_min", "ph_max", "ph",
     ["ph_min", "ph_max"], "faixa de pH suspensa: R4 sem dado -> 'verificar'"),
    ("precipitacao_min", "precipitacao_max", "precipitacao",
     ["precipitacao_min", "precipitacao_max"], "R7 sem dado -> 'verificar'"),
    ("produtividade_minima_ms", "produtividade_maxima_ms", "produtividade",
     [], "sinalizada (a média não depende da ordem, mas um dos valores pode estar errado)"),
    ("temp_min", "temp_max", "temperatura", [], "não usada no modelo (só relato)"),
    ("proteina_min", "proteina_max", "proteina", [], "não usada no modelo (só relato)"),
    ("biomassa_min", "biomassa_max", "biomassa", [], "não usada no modelo (só relato)"),
]

_linhas_inv = []
for c_min, c_max, rot, _susp_cols, consequencia in PARES_INTERVALO:
    col_flag = f"invertido_{rot}"
    if c_min in df.columns and c_max in df.columns:
        df[col_flag] = df[c_min].notna() & df[c_max].notna() & (df[c_min] > df[c_max])
    else:
        df[col_flag] = False
    for _, r in df[df[col_flag]].iterrows():
        _linhas_inv.append({
            "especie": r["especie"], "grupo": r["grupo"], "atributo": rot,
            "minimo": r[c_min], "maximo": r[c_max], "consequencia_no_modelo": consequencia,
        })

df["intervalos_invertidos"] = df.apply(
    lambda r: "; ".join(rot for _, _, rot, _, _ in PARES_INTERVALO if r[f"invertido_{rot}"]), axis=1
)

print("=" * 90)
print("INTERVALOS INVERTIDOS (mínimo > máximo) — CONFERIR A FONTE ANTES DE CORRIGIR")
print("=" * 90)
df_inv = pd.DataFrame(_linhas_inv)
if df_inv.empty:
    print("Nenhum intervalo invertido.")
else:
    print(df_inv.to_string(index=False))
    print(f"\nTotal: {len(df_inv)} intervalo(s) em {df_inv['especie'].nunique()} espécie(s).")
    print("Os valores NÃO foram invertidos; os atributos usados pelo modelo ficam suspensos.")
df_inv.to_csv("intervalos_invertidos.csv", index=False, encoding="utf-8-sig")

# Suspensão ANTES da imputação e das representações (Gower).
# Original preservado para auditoria; marcador <coluna>_suspensa impede que a
# imputação seja tratada como dado válido.
COLUNAS_SUSPENSAS = []
for _c_min, _c_max, _rot, _susp_cols, _ in PARES_INTERVALO:
    _mask = df[f"invertido_{_rot}"]
    for _c in _susp_cols:
        if _c not in df.columns:
            continue
        if f"{_c}_suspensa" not in df.columns:
            df[f"{_c}_suspensa"] = False
            COLUNAS_SUSPENSAS.append(_c)
        df[f"{_c}_original_invertido"] = df[_c].where(_mask)
        df.loc[_mask, f"{_c}_suspensa"] = True
        df.loc[_mask, _c] = np.nan

# ---------------------------------------------------------
# 3. IMPUTAÇÃO (somente variáveis da Gower; registrada)
# ---------------------------------------------------------
# ID 109 canônico; campos não reconciliados permanecem suspensos.
# O ID 58 foi removido da base operacional ANTES de qualquer imputação.
_ids = pd.to_numeric(df["id"], errors="coerce")
assert not _ids.eq(58).any(), "Execute reconciliar_xaraes.py antes desta versão."
_mask_canonico = _ids.eq(109)
assert _mask_canonico.sum() == 1, "É necessário um único registro canônico ID 109."
for _campo in ["profundidade_raiz_max", "ph_min", "ph_max"]:
    if f"{_campo}_suspensa" not in df.columns:
        df[f"{_campo}_suspensa"] = False
    df.loc[_mask_canonico, f"{_campo}_suspensa"] = True
    df.loc[_mask_canonico, _campo] = np.nan
    if _campo not in COLUNAS_SUSPENSAS:
        COLUNAS_SUSPENSAS.append(_campo)

features_solo = ["profundidade_raiz_max", "altura_media", "ph_min", "ph_max", "text_num", "fert_num"]

for col in features_solo:
    _susp = df[f"{col}_suspensa"] if f"{col}_suspensa" in df.columns else pd.Series(False, index=df.index)
    df[f"{col}_imputada"] = df[col].isna() & ~_susp   # suspenso não conta como "imputado"


def imputar_knn_preservando(df_in, colunas, n_neighbors=5):
    """KNN apenas nas células ausentes; valores observados ficam intactos
    (evita ruído numérico de padronização em valores originais)."""
    scaler = StandardScaler()
    padronizado = scaler.fit_transform(df_in[colunas])
    imputado = scaler.inverse_transform(KNNImputer(n_neighbors=n_neighbors).fit_transform(padronizado))
    imputado = pd.DataFrame(imputado, columns=colunas, index=df_in.index)
    saida = df_in.copy()
    for c in colunas:
        saida[c] = df_in[c].where(df_in[c].notna(), imputado[c])
    return saida


df = imputar_knn_preservando(df.copy(), features_solo)  # copy(): desfragmenta o DataFrame

# Representação para a Gower: as células suspensas recebem a estimativa KNN
# (feita só com valores válidos). Os valores invertidos não entram em nada.
_df_gower_base = df.copy()

# No modelo, a pendência é mantida: a imputação NÃO substitui célula suspensa.
for _c in COLUNAS_SUSPENSAS:
    df.loc[df[f"{_c}_suspensa"], _c] = np.nan

RAIZ_MIN = float(df["profundidade_raiz_max"].min())
RAIZ_MAX = float(df["profundidade_raiz_max"].max())
RAIZ_RANGE = (RAIZ_MAX - RAIZ_MIN) if RAIZ_MAX > RAIZ_MIN else 1.0
print(f"\nEscala da profundidade radicular (só valores válidos): {RAIZ_MIN:g}–{RAIZ_MAX:g} m.")
print("Obs.: corrigir/suspender raízes altera esta escala e, portanto, o score de raiz de TODAS as espécies.")

df["estrato_num"] = df["altura_media"].apply(classificar_estrato)
df["pos_matriz"] = df.index

# ---------------------------------------------------------
# 4. RELATÓRIOS DE QUALIDADE DA BASE (independentes de cenário)
# ---------------------------------------------------------
print("=" * 90)
print("COMPLETUDE DOS ATRIBUTOS USADOS NAS RESTRIÇÕES")
print("=" * 90)

campos_completude = {
    "tolerancia_pastejo": df["tolerancia_pastejo"].apply(limpar_texto).ne(""),
    "tolerancia_aluminio": df["al_num"].notna(),
    "tolerancia_sombra": df["sombra_num"].notna(),
    "exigencia_fertilidade": df["fert_num_imputada"].eq(False),
    "textura_solo": df["text_num_imputada"].eq(False),
    "ciclo": df["ciclo"].apply(categorizar_ciclo).ne("desconhecido") if "ciclo" in df.columns else False,
    "clima (alguma flag)": df["tem_info_clima"],
    "precipitacao_min": df["precipitacao_min"].notna(),
    "faixa de pH": df["ph_min"].notna() & df["ph_max"].notna() & ~df["ph_min_imputada"] & ~df["ph_max_imputada"],
}
tab_completude = pd.DataFrame(campos_completude)
tab_completude["grupo"] = df["grupo"]
print((tab_completude.groupby("grupo").mean(numeric_only=True) * 100).round(1).T.to_string())
df["completude"] = tab_completude.drop(columns="grupo").mean(axis=1).round(3)

print("\nEntradas genéricas que NÃO concorrem (há cultivares cadastradas):")
for e in df.loc[df["generico_substituivel"], "especie"]:
    print(" -", e)

print("\n" + "=" * 90)
print("ALERTAS DE COERÊNCIA ENTRE CULTIVARES DA MESMA ESPÉCIE")
print("=" * 90)
alertas = []
for binomio, grupo_b in df.groupby("binomio"):
    if len(grupo_b) < 2:
        continue
    for col, rotulo_attr in [("fert_num", "exigência em fertilidade"), ("al_num", "tolerância ao Al")]:
        obs = grupo_b[grupo_b[col].notna()]
        if col == "fert_num":
            obs = obs[~obs["fert_num_imputada"]]
        if len(obs) >= 2 and obs[col].max() - obs[col].min() >= 1:
            alertas.append({
                "espécie (chave)": binomio,
                "atributo": rotulo_attr,
                "valores": "; ".join(f"{e} = {v:g}" for e, v in zip(obs["especie"], obs[col])),
            })
df_alertas = pd.DataFrame(alertas)
print(df_alertas.to_string(index=False) if not df_alertas.empty else "Nenhum alerta.")
df_alertas.to_csv("alertas_coerencia_cultivares.csv", index=False, encoding="utf-8-sig")

print("\n" + "=" * 90)
print("POSSÍVEIS LINHAS DUPLICADAS (mesma espécie, sem marcador de cultivar: cv./BRS/Tifton)")
print("=" * 90)
_forte = df["especie"].apply(lambda n: bool(re.search(r"\bcv\b|cv\.|\bbrs\b|tifton", limpar_texto(n))))
_cand_dup = df[~_forte & ~df["eh_generico_genero"]]
duplicatas = [
    {"espécie (chave)": b, "linhas": " | ".join(g["especie"])}
    for b, g in _cand_dup.groupby("binomio") if len(g) >= 2
]
df_dup = pd.DataFrame(duplicatas)
print(df_dup.to_string(index=False) if not df_dup.empty else "Nenhuma.")
print("Obs.: variedades legítimas (ex.: guandu anão/arbóreo) também aparecem; revisar caso a caso.")
df_dup.to_csv("possiveis_duplicatas.csv", index=False, encoding="utf-8-sig")



def classificar_arranjo(row):
    """Porte da leguminosa: 'herbacea', 'arbustiva_arborea' ou 'indefinido'.
    Hábito tem precedência; sem hábito conclusivo, usa altura_max (m)."""
    h = limpar_texto(row.get("habito", ""))
    if "subarbust" in h:
        return "herbacea"
    # Trepadeiras e plantas de crescimento horizontal: herbáceas, qualquer altura
    if any(p in h for p in ["trepad", "volub", "volúv", "prostr", "rastei", "estolon", "decumb", "escandent"]):
        return "herbacea"
    if any(p in h for p in ["arbor", "arbór", "arvor", "árvor", "arbust"]):
        return "arbustiva_arborea"
    if row.get("invertido_altura", False):
        return "indefinido"   # altura inconsistente e hábito não conclusivo
    amax = row.get("altura_max", np.nan)
    if pd.notna(amax):
        return "arbustiva_arborea" if amax > ALTURA_ARBUSTIVA else "herbacea"
    return "herbacea" if h else "indefinido"


df["arranjo_leg"] = df.apply(classificar_arranjo, axis=1)

print("\n" + "=" * 90)
print(f"PORTE DAS LEGUMINOSAS (arranjo) — limiar de altura: {ALTURA_ARBUSTIVA} m")
print("=" * 90)
_leg_all = df[df["grupo"].str.contains("legumin", na=False)]
print(_leg_all["arranjo_leg"].value_counts().to_string())
for _cls in ["arbustiva_arborea", "indefinido"]:
    _sub = _leg_all[_leg_all["arranjo_leg"] == _cls]
    if not _sub.empty:
        print(f"\n{_cls}:")
        print(_sub[["especie", "habito", "altura_max"]].to_string(index=False))
print("Conferir: a classificação define em qual cenário (pastejo ou banco de proteína) a leguminosa entra.")

print("\n" + "=" * 90)
print("PRECIPITAÇÃO MÍNIMA DA BASE (R7) — conferir unidade (mm/ano)")
print("=" * 90)
print(df["precipitacao_min"].describe().to_string())
_susp = df[df["precipitacao_min"].notna() & (df["precipitacao_min"] < 200)]
if not _susp.empty:
    print("\nATENÇÃO: valores < 200 (possível mm/mês ou erro de unidade):")
    print(_susp[["especie", "precipitacao_min"]].to_string(index=False))
for _nome, _cen in CENARIOS.items():
    _p = _cen.get("precipitacao")
    if _p is None:
        print(f"\n[{_nome}] precipitação não definida: R7 DESATIVADA neste cenário.")
    else:
        _n = int((df["precipitacao_min"] > _p).sum())
        print(f"\n[{_nome}] {_p} mm/ano: {_n} espécies com precipitação mínima acima do cenário.")


# ---------------------------------------------------------
# 5. GOWER + HAC (exploratório — NÃO entra nas restrições nem no score)
# ---------------------------------------------------------
# Observação: as flags climáticas entram na Gower com 0 = "sem evidência";
# isso deve ser declarado como limitação da representação.
features_cluster = features_solo + ["prod_norm"] + colunas_clima_total
matriz_distancia = gower.gower_matrix(_df_gower_base[features_cluster].copy())

features_gower_ambiental = ["ph_min", "ph_max", "text_num", "fert_num"] + colunas_clima_total
matriz_distancia_ambiental = gower.gower_matrix(_df_gower_base[features_gower_ambiental].copy())

melhor_k, melhor_sil, melhor_labels = 2, -1, None
print("\n" + "=" * 90)
print("GOWER + HAC (exploratório) — SILHOUETTE k=2 a 6")
print("=" * 90)
for k in range(2, 7):
    labels = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="complete").fit_predict(matriz_distancia)
    sil = silhouette_score(matriz_distancia, labels, metric="precomputed")
    print(f"k = {k}: Silhouette = {sil:.6f}")
    if sil > melhor_sil:
        melhor_k, melhor_sil, melhor_labels = k, sil, labels
df["cluster_ia"] = melhor_labels
print(f"k ótimo = {melhor_k} | Silhouette = {melhor_sil:.6f}")

# ---------------------------------------------------------
# 6. AVALIAÇÃO POR ESPÉCIE: RESTRIÇÕES + SCORE
# ---------------------------------------------------------
CLIMA_ACEITO = {
    "cerrado": ["cerrado", "tropical", "quente", "tropical_umido"],
    "tropical": ["tropical", "cerrado", "quente"],
    "quente": ["tropical", "cerrado", "quente"],
}


def avaliar_clima(row, clima):
    if any(row.get(c, 0) == 1 for c in CLIMA_ACEITO.get(clima, [clima])):
        return "apto"
    if row["tem_info_clima"]:
        return "inapto"      # há flags climáticas, mas nenhuma compatível
    return "sem_info"


def aptidao_pastejo(row):
    """(estado, nota em [0,1], texto bruto) a partir de tolerancia_pastejo;
    a flag 'pastejo'/'pastejo_direto' = 1 vale só como evidência positiva."""
    bruto = limpar_texto(row.get("tolerancia_pastejo", ""))
    if bruto.startswith("muito alt"):
        return "apto", 1.0, bruto
    if bruto.startswith("alt"):
        return "apto", 0.85, bruto
    if bruto == "sim":
        return "apto", 0.7, bruto
    if bruto.startswith(("med", "méd", "moder")):
        return "ressalva", 0.5, bruto
    if bruto.startswith(("baix", "nao_aplic", "nao aplic", "não aplic", "toxic", "tóxic")):
        return "inapto", 0.0, bruto
    if row.get("pastejo", 0) == 1 or row.get("pastejo_direto", 0) == 1:
        return "apto", 0.7, "flag pastejo=1"
    return "sem_info", np.nan, bruto


NIVEL_PASTEJO = {"baixa": 1, "media": 2, "alta": 3, "muito alta": 4}


def nivel_tolerancia_pastejo(row):
    """Nível ordinal de tolerancia_pastejo (NaN se não informado ou não ordinal)."""
    b = limpar_texto(row.get("tolerancia_pastejo", ""))
    if b.startswith("muito alt"):
        return 4
    if b.startswith("alt"):
        return 3
    if b.startswith(("med", "méd", "moder")):
        return 2
    if b.startswith("baix"):
        return 1
    return np.nan


COMPONENTES = ["ph", "aluminio", "textura", "fertilidade", "pastejo", "raiz"]


def pendencia_identidade_pueraria(row):
    """Controle documental provisório; não implica inaptidão agronômica.

    Os IDs 28 e 68 não têm material distinto identificado e contêm atributos
    divergentes. Preservar ambos para auditoria, sem escolher um pelo ranking.
    Remover este controle somente após reconciliação documentada da base.
    """
    nome = " ".join(str(row.get("especie", "")).strip().lower().split())
    nomes_pendentes = {
        "pueraria phaseoloides (puerária)",
        "pueraria phaseoloides (kudzu tropical)",
    }
    if nome in nomes_pendentes:
        return ("identidade e descritores de puerária pendentes de reconciliação "
                "(IDs 28/68; material distinto não documentado)")
    return ""


def restricao_uso_documental(row, cen):
    """Elegibilidade de uso separada da tolerância à intensidade de pastejo.

    Capiaçu: recomendação documentada para silagem/picado verde no cocho.
    Fonte: Retore et al. (2021), Embrapa, Comunicado Técnico 263.
    https://www.infoteca.cnptia.embrapa.br/handle/doc/1134720
    O veto aplica-se ao papel de gramínea sob pastejo direto nos arranjos
    atuais. Não é afirmação de inviabilidade para todos os usos forrageiros.
    """
    nome = " ".join(str(row.get("especie", "")).strip().lower().split())
    grupo = str(row.get("grupo", "")).lower()
    arranjo = cen.get("arranjo", "consorcio_pastejo")
    if ("gramin" in grupo and ("capiaçu" in nome or "capiacu" in nome)
            and arranjo in {"consorcio_pastejo", "banco_proteina"}):
        return ("manejo/uso: BRS Capiaçu destinada a corte para silagem ou "
                "picado verde; sem aptidão documental para o papel de "
                "gramínea sob pastejo direto neste workflow")
    return ""


def pendencia_solo_leucena(row, ph_efetivo):
    """Pendência específica de curadoria para o registro genérico de leucena.

    Tropical Forages: pH(H2O)>5.5, ou >5.0 com saturação de Al muito baixa.
    Os cenários atuais não informam saturação de Al nem método de pH.
    Assim, não confirmar automaticamente aptidão em pH efetivo <=5.5.
    Não atribuir um limite percentual de Al que a fonte não quantificou.
    Fonte: https://tropicalforages.info/text/entities/leucaena_leucocephala.htm
    """
    nome = " ".join(str(row.get("especie", "")).strip().lower().split())
    if nome in {"leucaena leucocephala", "leucaena leucocephala (lam.) de wit"}:
        if float(ph_efetivo) <= 5.5:
            return ("leucena em solo ácido: confirmar pH em água, saturação "
                    "de alumínio e condição de correção do solo; aptidão "
                    "não confirmada para o registro genérico")
    return ""


def avaliar_especie(row, cen, w_eco=0.6, w_prod=0.4):
    ph = float(cen["ph"])
    calagem = bool(cen.get("calagem", False))
    excl, falt_obr, falt_comp, ress = [], [], [], []

    # Controle de curadoria: incerteza documental não equivale a veto biológico.
    # O status final continua sendo excluido se houver qualquer veto real.
    pendencia_pueraria = pendencia_identidade_pueraria(row)
    if pendencia_pueraria:
        falt_obr.append(pendencia_pueraria)

    # R0 — genérico
    if row["generico_substituivel"]:
        excl.append("entrada genérica (há cultivares cadastradas)")

    # R1 — ciclo
    ciclo = categorizar_ciclo(row.get("ciclo", ""))
    if ciclo == "anual_bienal":
        excl.append("ciclo não perene")
    elif ciclo == "desconhecido":
        falt_obr.append("ciclo")

    # R2 — clima
    est_clima = avaliar_clima(row, cen["clima"])
    if est_clima == "inapto":
        excl.append(f"sem aptidão ao clima '{cen['clima']}'")
    elif est_clima == "sem_info":
        falt_obr.append("aptidão climática")

    # R3a — aptidão de uso: independente de calagem e do mínimo de tolerância.
    restricao_uso = restricao_uso_documental(row, cen)
    if restricao_uso:
        excl.append(restricao_uso)

    # R3 — pastejo (intensidade; não substitui a aptidão de uso)
    est_past, nota_past, txt_past = aptidao_pastejo(row)
    _eh_gram = "gramin" in str(row["grupo"])
    _papel_past = "gramínea principal" if _eh_gram else "leguminosa"
    _min_past = (cen.get("tol_min_graminea", TOL_PASTEJO_MIN_GRAMINEA) if _eh_gram
                 else cen.get("tol_min_leguminosa", TOL_PASTEJO_MIN_LEGUMINOSA))
    _nivel_past = nivel_tolerancia_pastejo(row)
    if est_past == "inapto":
        excl.append(f"inapto a pastejo (tolerancia_pastejo='{txt_past}')")
    elif row.get("capineiras", 0) == 1 and est_past != "apto":
        excl.append("cultivar de corte (capineira) sem aptidão comprovada a pastejo")
    elif est_past == "sem_info":
        falt_obr.append("aptidão a pastejo")
    elif pd.notna(_nivel_past) and _nivel_past < NIVEL_PASTEJO[_min_past]:
        excl.append(f"tolerância a pastejo '{txt_past}' abaixo do mínimo para {_papel_past} "
                    f"sob pastejo direto ('{_min_past}')")
    elif pd.isna(_nivel_past):
        # Uso em pastejo documentado ('sim' ou flag), mas intensidade desconhecida
        if POLITICA_NIVEL_DESCONHECIDO == "verificar":
            falt_obr.append(f"nível de tolerância a pastejo (só indicação de uso: {txt_past}; "
                            f"mínimo exigido '{_min_past}')".replace("; ", ", "))
        else:
            ress.append("uso em pastejo documentado; nível de tolerância desconhecido")
    elif est_past == "ressalva":
        ress.append("tolerância a pastejo média")
    past_norm = None if pd.isna(nota_past) else nota_past

    # R4 — pH (após calagem, se houver)
    ph_ef = max(ph, PH_POS_CALAGEM) if calagem else ph
    pendencia_leucena = pendencia_solo_leucena(row, ph_ef)
    if pendencia_leucena:
        falt_obr.append(pendencia_leucena)
    pmin, pmax = row["ph_min"], row["ph_max"]
    pendencia_ph = pendencia_ph_documental(row)
    if pendencia_ph:
        falt_obr.append(pendencia_ph)
        # Sem faixa documental confirmada, não avaliar adequação de pH.
        # Outros vetos continuam a prevalecer: verificar não substitui excluido.
        ph_norm = None
    elif pd.isna(pmin) or pd.isna(pmax):
        falt_obr.append("faixa de pH (intervalo invertido)" if row["invertido_ph"] else "faixa de pH")
        ph_norm = None
    elif ph_ef < pmin - DELTA_PH or ph_ef > pmax + DELTA_PH:
        excl.append(f"pH {ph_ef:g} fora da faixa {pmin:g}–{pmax:g}")
        ph_norm = 0.0
    elif pmin <= ph_ef <= pmax:
        ph_norm = 1.0
    elif ph_ef < pmin:
        ph_norm = (ph_ef - (pmin - DELTA_PH)) / DELTA_PH
        ress.append("pH na margem inferior da faixa")
    else:
        ph_norm = ((pmax + DELTA_PH) - ph_ef) / DELTA_PH
        ress.append("pH na margem superior da faixa")
    ph_norm = None if ph_norm is None else float(np.clip(ph_norm, 0.0, 1.0))
    if row["ph_min_imputada"] or row["ph_max_imputada"]:
        falt_comp.append("faixa de pH (imputada)")

    # R5 — alumínio (só sem calagem e em pH ácido)
    al_ativo = (not calagem) and ph_ef <= PH_LIMIAR_AL
    al = row["al_num"]
    if not al_ativo:
        al_norm = 0.0
    elif pd.isna(al):
        falt_obr.append("tolerância ao Al")
        al_norm = None
    elif al < 2:
        excl.append(f"baixa tolerância ao Al (pH {ph:g}, sem calagem)")
        al_norm = 0.0
    else:
        al_norm = (al - 1.0) / 2.0      # moderada = 0.5 ; alta = 1.0

    # R7 — precipitação mínima (obrigatória)
    precip_cen = cen.get("precipitacao", None)
    if precip_cen is not None:
        pmin_chuva = row.get("precipitacao_min", np.nan)
        if pd.isna(pmin_chuva):
            falt_obr.append("precipitação mínima (intervalo invertido)"
                            if row["invertido_precipitacao"] else "precipitação mínima")
        elif precip_cen < pmin_chuva:
            excl.append(f"precipitação {precip_cen:g} mm < mínima da espécie ({pmin_chuva:g} mm)")

    # Compensatório — fertilidade (só penaliza)
    fert_alvo = mapear_fertilidade(cen["fert"])
    if row["fert_num_imputada"] or pd.isna(row["fert_num"]) or pd.isna(fert_alvo):
        falt_comp.append("exigência em fertilidade")
        fert_norm = None
    else:
        gap = round(row["fert_num"]) - fert_alvo
        fert_norm = 1.0 if gap <= 0 else (0.5 if gap == 1 else 0.0)
        if gap >= 1:
            ress.append(f"exigência em fertilidade acima da disponível (+{gap:g} nível)")

    # Compensatório — textura
    text_alvo = mapear_textura(cen["text"])
    if pd.isna(text_alvo) or pd.isna(row["text_num"]):
        text_norm = None
    else:
        text_norm = 1.0 if abs(row["text_num"] - text_alvo) <= 1 else 0.0
    if row["text_num_imputada"]:
        falt_comp.append("textura (imputada)")

    # Compensatório — raiz
    if pd.isna(row["profundidade_raiz_max"]):
        raiz_norm = None
        falt_comp.append("profundidade radicular (intervalo invertido)"
                         if row["invertido_raiz"] else "profundidade radicular")
    else:
        raiz_norm = (row["profundidade_raiz_max"] - RAIZ_MIN) / RAIZ_RANGE
    if row["invertido_produtividade"]:
        falt_comp.append("produtividade (intervalo invertido)")

    if row["prod_imputada"]:
        falt_comp.append("produtividade (mediana)")

    comps = {
        "ph": ph_norm, "aluminio": al_norm, "textura": text_norm,
        "fertilidade": fert_norm, "pastejo": past_norm,
        "raiz": None if raiz_norm is None else float(np.clip(raiz_norm, 0, 1)),
    }
    pesos_base = {
        "ph": 1.1, "aluminio": 1.0 if al_ativo else 0.0, "textura": 1.0,
        "fertilidade": 1.0, "pastejo": 0.8, "raiz": 1.0,
    }
    # Critério sem dado: NÃO AVALIADO (sai da média; pesos renormalizados).
    nao_avaliados = [k for k in COMPONENTES if pesos_base[k] > 0 and comps[k] is None]
    pesos = {k: (0.0 if comps[k] is None else pesos_base[k]) for k in COMPONENTES}
    comps = {k: (0.0 if v is None else v) for k, v in comps.items()}
    soma_p = sum(pesos.values())
    cobertura = soma_p / sum(pesos_base.values())
    if soma_p == 0:
        soma_p = 1.0   # nenhum critério avaliado (s_eco = 0)
    s_eco = sum(pesos[k] * comps[k] for k in COMPONENTES) / soma_p
    s_prod = float(np.clip(row["prod_norm"], 0.0, 1.0))
    s_tot = w_eco * s_eco + w_prod * s_prod
    score_bruto = round(100.0 * s_tot, 2)

    contrib = {}
    for k in COMPONENTES:
        parte = w_eco * pesos[k] * comps[k] / soma_p
        contrib[f"contrib_{k}"] = round(100.0 * parte / s_tot, 1) if s_tot > 0 else 0.0
    contrib["contrib_produtividade"] = round(100.0 * w_prod * s_prod / s_tot, 1) if s_tot > 0 else 0.0

    status = "excluido" if excl else ("verificar" if falt_obr else "apto")

    return {
        "status": status,
        "score": score_bruto if status != "excluido" else 0.0,
        "score_bruto": score_bruto,
        "s_eco": round(s_eco, 4),
        "estado_pastejo": est_past,
        "motivos_exclusao": "; ".join(excl),
        "faltantes_obrig": "; ".join(falt_obr),
        "faltantes_comp": "; ".join(falt_comp),
        "ressalvas": "; ".join(ress),
        "criterios_nao_avaliados": ", ".join(nao_avaliados),
        "cobertura_score": round(cobertura, 3),
        **contrib,
    }


def avaliar_cenario(cen, w_eco=0.6, w_prod=0.4):
    res = df.apply(lambda r: avaliar_especie(r, cen, w_eco, w_prod), axis=1, result_type="expand")
    return pd.concat([df, res], axis=1)


def relatorio_status(av, rotulo, top=12, exportar=True):
    print("\n" + "=" * 100)
    print(f"STATUS DAS ESPÉCIES — {rotulo}")
    print("=" * 100)
    print(pd.crosstab(av["grupo"], av["status"]).to_string())

    motivos = (
        av.loc[av["status"] == "excluido", "motivos_exclusao"]
        .str.split("; ").explode()
        .str.replace(r"\(.*\)", "(...)", regex=True)
        .value_counts()
    )
    print("\nMotivos de exclusão (uma espécie pode ter mais de um):")
    print(motivos.to_string())

    faltas = av.loc[av["status"] == "verificar", "faltantes_obrig"].str.split("; ").explode().value_counts()
    if not faltas.empty:
        print("\nDado obrigatório faltante nas espécies 'verificar':")
        print(faltas.to_string())

    print("\nNota: aptidão individual NÃO implica dados suficientes para avaliar a complementaridade")
    print("do consórcio. 'nao_avaliados' = critérios sem dado, fora da média (cobertura < 1).")
    cols = ["especie", "score", "cobertura_score", "criterios_nao_avaliados", "ressalvas"]
    for grp in ["gramin", "legumin"]:
        nome_grp = "gramíneas" if grp == "gramin" else "leguminosas"
        sub = av[av["grupo"].str.contains(grp, na=False)]
        aptas = sub[sub["status"] == "apto"].sort_values("score", ascending=False)
        print(f"\nTop-{top} {nome_grp} APTAS:")
        cols_g = cols if grp == "gramin" else ["especie", "arranjo_leg"] + cols[1:]
        print(aptas.head(top)[cols_g].to_string(index=False) if not aptas.empty else "Nenhuma.")
        cond = sub[sub["status"] == "verificar"].sort_values("score", ascending=False)
        if not cond.empty:
            print(f"\n{nome_grp.capitalize()} CONDICIONADAS (verificar) — fora do ranking até o dado ser confirmado:")
            print(cond[["especie", "score", "faltantes_obrig"]].to_string(index=False))

    if exportar:
        slug = _slug(rotulo)
        av[[
            "especie", "grupo", "status", "score", "score_bruto", "s_eco", "prod_norm",
            "motivos_exclusao", "faltantes_obrig", "faltantes_comp", "ressalvas", "completude",
            "intervalos_invertidos", "criterios_nao_avaliados", "cobertura_score",
        ] + [c for c in av.columns if c.startswith("contrib_")]].to_csv(
            f"status_especies_{slug}.csv", index=False, encoding="utf-8-sig"
        )


def _slug(txt):
    t = unicodedata.normalize("NFKD", txt.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "_", t).strip("_")


# ---------------------------------------------------------
# 7. AVALIAÇÃO DE PARES (R6 + complementaridade)
# ---------------------------------------------------------
COMP_ESTRATO_IGUAL = 0.3
# Componente sem dado: NÃO AVALIADO (complementaridade = média dos disponíveis;
# o par fica 'verificar').


def avaliar_par(g, l):
    excl, falt = [], []
    if pd.isna(g["estrato_num"]) or pd.isna(l["estrato_num"]):
        sem = [r["especie"] for r in (g, l) if pd.isna(r["estrato_num"])]
        falt.append(f"altura/estrato inconsistente ou ausente ({', '.join(sem)})")
        comp_estrato = None
    elif g["estrato_num"] == l["estrato_num"]:
        comp_estrato = COMP_ESTRATO_IGUAL
    else:
        inferior = l if g["estrato_num"] > l["estrato_num"] else g
        papel = "leguminosa" if inferior is l else "gramínea"
        if pd.isna(inferior["sombra_num"]):
            falt.append(f"tolerância à sombra da {papel} ({inferior['especie']})")
            comp_estrato = None
        elif inferior["sombra_num"] < 2:
            excl.append(f"{papel} do estrato inferior não tolera sombreamento")
            comp_estrato = 0.0
        else:
            comp_estrato = 1.0

    if pd.isna(g["profundidade_raiz_max"]) or pd.isna(l["profundidade_raiz_max"]):
        sem = [r["especie"] for r in (g, l) if pd.isna(r["profundidade_raiz_max"])]
        falt.append(f"profundidade radicular inconsistente ou ausente ({', '.join(sem)})")
        comp_raiz = None
    else:
        diff_raiz = abs(float(g["profundidade_raiz_max"]) - float(l["profundidade_raiz_max"]))
        comp_raiz = float(np.clip(diff_raiz / RAIZ_RANGE, 0.0, 1.0))

    status_par = "excluido" if excl else ("verificar" if falt else "apto")
    _disp = [v for v in (comp_raiz, comp_estrato) if v is not None]
    return {
        "status_par": status_par,
        "motivos_par": "; ".join(excl),
        "faltantes_par": "; ".join(falt),
        "comp_raiz_norm": np.nan if comp_raiz is None else round(comp_raiz, 6),
        "comp_estrato_norm": np.nan if comp_estrato is None else round(comp_estrato, 6),
        "complementaridade_funcional": (
            round(float(np.clip(np.mean(_disp), 0, 1)), 6) if _disp else np.nan
        ),
        "componentes_nao_avaliados": ", ".join(
            n for n, v in [("raiz", comp_raiz), ("estrato/sombra", comp_estrato)] if v is None
        ),
    }


def _pendencias(g, l, par):
    itens = []
    for r in (g, l):
        if r["faltantes_obrig"]:
            itens.append(f"{r['especie']}: {r['faltantes_obrig']}")
    if par["faltantes_par"]:
        itens.append(par["faltantes_par"])
    return " | ".join(itens)


def avaliar_par_completo(g, l, arranjo):
    """Avalia um par gramínea-leguminosa pelas regras do modelo.
    Retorna a linha do par (com status combinado, motivos, pendências e
    objetivos), MESMO se o par for excluído — quem chama decide o que fazer."""
    par = avaliar_par(g, l)
    status = pior_status(g["status"], l["status"], par["status_par"])
    pend = _pendencias(g, l, par)
    if l["arranjo_leg"] == "indefinido" and status != "excluido":
        status = pior_status(status, "verificar")
        pend = " | ".join(p for p in [pend, f"porte/hábito da leguminosa ({l['especie']})"] if p)
    motivos = " | ".join(
        x for x in [
            f"{g['especie']}: {g['motivos_exclusao']}" if g["motivos_exclusao"] else "",
            f"{l['especie']}: {l['motivos_exclusao']}" if l["motivos_exclusao"] else "",
            par["motivos_par"],
        ] if x
    )
    d_env = matriz_distancia_ambiental[int(g["pos_matriz"]), int(l["pos_matriz"])]
    return {
        "Gramínea (Principal)": g["especie"],
        "Leguminosa (Acompanhante)": l["especie"],
        "arranjo": arranjo,
        "status_par": status,
        "motivos_exclusao_par": motivos,
        "pendencias": pend,
        "score_graminea": round(float(g["score"]), 4),
        "score_leguminosa": round(float(l["score"]), 4),
        "produtividade_graminea": float(g["produtividade_media"]),
        "suitability_media": round(float(np.clip((g["score"] + l["score"]) / 200.0, 0, 1)), 6),
        "compatibilidade_ambiental": round(float(np.clip(1.0 - d_env, 0, 1)), 6),
        "comp_raiz_norm": par["comp_raiz_norm"],
        "comp_estrato_norm": par["comp_estrato_norm"],
        "complementaridade_funcional": par["complementaridade_funcional"],
        "base_do_score": _base_do_score(g, l, par),
    }


def _base_do_score(g, l, par):
    """'completa' ou 'parcial (condicional): ...' — score com critérios/componentes não avaliados."""
    faltas = []
    for r in (g, l):
        if r.get("criterios_nao_avaliados", ""):
            faltas.append(f"{r['especie']}: {r['criterios_nao_avaliados']}")
    if par.get("componentes_nao_avaliados", ""):
        faltas.append(f"complementaridade sem {par['componentes_nao_avaliados']}")
    return "completa" if not faltas else "parcial (condicional): " + " | ".join(faltas)


def montar_pares(av, top_k=None, arranjo="consorcio_pastejo"):
    """Forma os pares não excluídos.

    Gramíneas candidatas (v3.1): TODAS as gramíneas 'apto' (ou as top_k aptas
    por score, só para análise de sensibilidade). Gramíneas 'verificar'
    também formam pares, que só podem ser 'verificar' (candidatos condicionados).
    O limiar relativo de 95% foi removido: com as restrições obrigatórias,
    ele reduzia o consórcio a uma única gramínea.
    """
    nao_excl = av[av["status"] != "excluido"]
    gram = nao_excl[nao_excl["grupo"].str.contains("gramin", na=False)].sort_values("score", ascending=False)
    leg = nao_excl[nao_excl["grupo"].str.contains("legumin", na=False) & (nao_excl["score"] > 0)]

    # Arranjo (v3.2): pastejo -> herbáceas (+ indefinidas, como 'verificar');
    # banco de proteína -> arbustivas/arbóreas
    if arranjo not in ARRANJOS_VALIDOS:
        raise ValueError(f"arranjo inválido: {arranjo}")
    n_leg_antes = len(leg)
    if arranjo == "consorcio_pastejo":
        leg = leg[leg["arranjo_leg"].isin(["herbacea", "indefinido"])]
    else:
        leg = leg[leg["arranjo_leg"] == "arbustiva_arborea"]
    n_leg_fora = n_leg_antes - len(leg)

    gram_apto = gram[gram["status"] == "apto"]
    if top_k is not None:
        gram_apto = gram_apto.head(int(top_k))
    gram_verif = gram[gram["status"] == "verificar"]
    gram_c = pd.concat([gram_apto, gram_verif])

    info = {
        "n_gramineas_aptas": int(len(gram_apto)),
        "n_gramineas_verificar": int(len(gram_verif)),
        "n_leguminosas_aptas": int((leg["status"] == "apto").sum()),
        "n_leguminosas_verificar": int((leg["status"] == "verificar").sum()),
        "n_leguminosas_fora_do_arranjo": int(n_leg_fora),
        "n_pares_excluidos_R6": 0,
    }
    if gram_c.empty or leg.empty:
        return pd.DataFrame(), info

    linhas = []
    for _, g in gram_c.iterrows():
        for _, l in leg.iterrows():
            linha = avaliar_par_completo(g, l, arranjo)
            if linha["status_par"] == "excluido":
                info["n_pares_excluidos_R6"] += 1
                continue
            linhas.append(linha)
    return pd.DataFrame(linhas), info


# ---------------------------------------------------------
# 8. CASOS DE TESTE — AVALIAÇÃO EXTERNA DA SHORTLIST (08/09/2026)
# ---------------------------------------------------------
# Critério: o modelo NÃO pode classificar como 'apto' o que o avaliador
# rejeitou, e o MOTIVO registrado deve ser comparado ao do avaliador.
# 'verificar' é a resposta coerente quando o avaliador indicou que
# levaria à avaliação local / pediu informação adicional.
CASOS_TESTE = [
    dict(
        nome="Par A — Tanzânia + A. glabrata", tipo="par",
        graminea=r"tanz", leguminosa=r"glabrata",
        esperado={"verificar", "excluido"},
        motivos=[("sombreamento", r"[Ss]ombr"), ("acidez", r"\bAl\b|pH|[Aa]cidez", "sem_calagem")],
        avaliador="Adaptação: Não (baixa tolerância a solo ácido); manejo: Incerto; "
                  "levaria a avaliação local; faltam estabelecimento e tolerância ao sombreamento",
    ),
    dict(
        nome="Par B — BRS Capiaçu + A. pintoi Belmonte", tipo="par",
        graminea=r"capia", leguminosa=r"belmonte",
        esperado={"excluido"},
        motivos=[("manejo/uso", r"[Pp]astejo|[Cc]apineira|[Cc]orte")],
        avaliador="Não em tudo: Capiaçu não é interessante para pastejo direto",
    ),
    dict(
        nome="Par C — Tanzânia + A. pintoi (sem cultivar)", tipo="par",
        graminea=r"tanz", leguminosa=r"^arachis pintoi\s*$",
        esperado={"verificar", "excluido"},
        motivos=[("cultivar", r"gen[ée]rica|[Cc]ultivar")],
        avaliador="Adaptação: Não; levaria a avaliação local; necessário conhecer a cultivar",
    ),
    dict(
        nome="Tanzânia isolada em pH 5,0 sem calagem", tipo="especie",
        padrao=r"tanz", aplica_se="sem_calagem",
        esperado={"verificar", "excluido"},
        motivos=[("acidez", r"\bAl\b|pH|[Aa]cidez")],
        avaliador="Baixa tolerância a solos ácidos",
    ),
    dict(
        nome="BRS Capiaçu isolada (uso sob pastejo)", tipo="especie",
        padrao=r"capia",
        esperado={"excluido"},
        motivos=[("manejo/uso", r"[Pp]astejo|[Cc]apineira|[Cc]orte")],
        avaliador="Não indicada para pastejo direto",
    ),
]


def _buscar(av, padrao):
    return av[av["especie"].str.lower().str.strip().str.contains(padrao, regex=True, na=False)]


def _resumo(r):
    if r["motivos_exclusao"]:
        return f"excluída: {r['motivos_exclusao']}"
    partes = []
    if r["faltantes_obrig"]:
        partes.append(f"falta: {r['faltantes_obrig']}")
    if r["ressalvas"]:
        partes.append(f"ressalvas: {r['ressalvas']}")
    return "; ".join(partes) if partes else "atende a todas as restrições"


def executar_casos_teste(cen, rotulo):
    av = avaliar_cenario(cen)
    resultados = []
    print("\n" + "=" * 100)
    print(f"CASOS DE TESTE — AVALIAÇÃO EXTERNA | {rotulo}")
    print("=" * 100)
    calagem_cen = bool(cen.get("calagem", False))
    for caso in CASOS_TESTE:
        if caso.get("aplica_se") == "sem_calagem" and calagem_cen:
            resultados.append({"caso": caso["nome"], "obtido": "-", "esperado": "-",
                               "motivos_esperados": "", "motivos_atendidos": "",
                               "motivos_ausentes": "", "resultado": "N/A"})
            print(f"\n[N/A] {caso['nome']} — não se aplica com calagem "
                  "(o avaliador indicou que a resposta muda com a correção do solo)")
            continue
        motivos_caso = [m for m in caso["motivos"]
                        if len(m) < 3 or (m[2] == "sem_calagem" and not calagem_cen)]
        if caso["tipo"] == "especie":
            m = _buscar(av, caso["padrao"])
            if m.empty:
                obtido, detalhe = "não encontrado", "-"
            else:
                r = m.iloc[0]
                obtido, detalhe = r["status"], f"{r['especie']} — {_resumo(r)}"
                texto_motivo = f"{r['motivos_exclusao']} {r['faltantes_obrig']}"
        else:
            mg, ml = _buscar(av, caso["graminea"]), _buscar(av, caso["leguminosa"])
            if mg.empty or ml.empty:
                obtido, detalhe = "não encontrado", "-"
            else:
                g, l = mg.iloc[0], ml.iloc[0]
                par = avaliar_par(g, l)
                obtido = pior_status(g["status"], l["status"], par["status_par"])
                det_par = par["motivos_par"] or par["faltantes_par"] or "sem restrição do par"
                texto_motivo = " ".join([
                    g["motivos_exclusao"], g["faltantes_obrig"], l["motivos_exclusao"],
                    l["faltantes_obrig"], par["motivos_par"], par["faltantes_par"],
                ])
                detalhe = (
                    f"G [{g['especie']}]: {_resumo(g)}\n      "
                    f"L [{l['especie']}]: {_resumo(l)}\n      "
                    f"Par: {det_par}"
                )
        motivos_ok, motivos_falta = [], []
        if obtido == "não encontrado":
            veredito = "N/A"
        elif obtido not in caso["esperado"]:
            veredito = "FALHA"
        else:
            for rot, rx, *_ in motivos_caso:
                (motivos_ok if re.search(rx, texto_motivo)
                 else motivos_falta).append(rot)
            veredito = "PASSA" if not motivos_falta else "PARCIAL"
        resultados.append({
            "caso": caso["nome"], "obtido": obtido, "esperado": "/".join(sorted(caso["esperado"])),
            "motivos_esperados": ", ".join(m[0] for m in motivos_caso),
            "motivos_atendidos": ", ".join(motivos_ok), "motivos_ausentes": ", ".join(motivos_falta),
            "resultado": veredito,
        })
        print(f"\n[{veredito}] {caso['nome']}")
        print(f"   esperado: {'/'.join(sorted(caso['esperado']))} | obtido: {obtido}")
        print(f"   motivo esperado: {', '.join(m[0] for m in motivos_caso)}"
              + (f" | ausente: {', '.join(motivos_falta)}" if motivos_falta else ""))
        print(f"   avaliador: {caso['avaliador']}")
        print(f"   modelo:    {detalhe}")
    df_res = pd.DataFrame(resultados)
    print("\nRESUMO:", df_res["resultado"].value_counts().to_dict())
    if calagem_cen:
        print("Obs.: casos repetidos do cenário de referência NÃO são avaliações independentes.")
    print("PASSA = status e motivo corretos | PARCIAL = status certo, motivo diferente do avaliador")
    print("FALHA = status errado. Falhas por dado devem ser resolvidas na curadoria (com fonte).")
    df_res.to_csv(f"casos_teste_{_slug(rotulo)}.csv", index=False, encoding="utf-8-sig")
    return df_res


# ---------------------------------------------------------
# 9. EXECUÇÃO PRINCIPAL — STATUS POR CENÁRIO + CASOS DE TESTE
# ---------------------------------------------------------
avaliacoes = {}
_ja_relatados = {}
for nome_cen, cen in CENARIOS.items():
    avaliacoes[nome_cen] = avaliar_cenario(cen)
    chave = tuple((k, v) for k, v in sorted(cen.items()) if k != "arranjo")
    if chave in _ja_relatados:
        print(f"\n[{nome_cen}] mesmas condições de solo/clima de '{_ja_relatados[chave]}': "
              "status por espécie idêntico (muda só o arranjo dos pares).")
        continue
    _ja_relatados[chave] = nome_cen
    relatorio_status(avaliacoes[nome_cen], nome_cen)

casos_ref = executar_casos_teste(CENARIOS[CENARIO_REF], CENARIO_REF)
casos_calagem = executar_casos_teste(
    CENARIOS["Baseline (Cerrado, pH 5.0, com calagem)"], "Baseline (Cerrado, pH 5.0, com calagem)"
)

# ---------------------------------------------------------
# 10. SENSIBILIDADE ECOLÓGICO vs. PRODUTIVIDADE (espécies não excluídas)
# ---------------------------------------------------------
cenarios_pesos = {"70/30": (0.70, 0.30), "60/40": (0.60, 0.40), "50/50": (0.50, 0.50)}


def ranking_pesos(cen, w_eco, w_prod):
    av = avaliar_cenario(cen, w_eco, w_prod)
    av = av[av["status"] == "apto"]
    rk = av[["especie", "grupo", "status", "score", "cobertura_score", "criterios_nao_avaliados"]].sort_values(
        "score", ascending=False).reset_index(drop=True)
    rk["rank"] = np.arange(1, len(rk) + 1)
    return rk


rankings_pesos = {n: ranking_pesos(CENARIOS[CENARIO_REF], *p) for n, p in cenarios_pesos.items()}
base_rk = rankings_pesos["60/40"]
linhas_sens = []
for nome, rk in rankings_pesos.items():
    comp = base_rk[["especie", "rank"]].merge(rk[["especie", "rank"]], on="especie", suffixes=("_base", "_teste"))
    rho, p = spearmanr(comp["rank_base"], comp["rank_teste"]) if len(comp) > 2 else (np.nan, np.nan)
    linhas_sens.append({
        "Cenário": nome, "Peso ecológico": cenarios_pesos[nome][0],
        "Peso produtividade": cenarios_pesos[nome][1], "Spearman rho": rho, "p-value": p,
        "Top-10 overlap": len(set(base_rk.head(10)["especie"]) & set(rk.head(10)["especie"])),
    })
print("\n" + "=" * 90)
print(f"SENSIBILIDADE — ECOLÓGICO vs. PRODUTIVIDADE | {CENARIO_REF}")
print("(somente espécies APTAS)")
print("=" * 90)
print(pd.DataFrame(linhas_sens).to_string(index=False))
for nome, rk in rankings_pesos.items():
    print(f"\n--- TOP 10 — {nome} ---")
    print(rk.head(10)[["rank", "especie", "grupo", "score", "cobertura_score",
                       "criterios_nao_avaliados"]].to_string(index=False))
print("\nAtenção: score com cobertura < 1 usa MENOS critérios (pesos renormalizados);")
print("não comparar posições como se a informação fosse igualmente completa.")

# ---------------------------------------------------------
# 11. PARETO — comparação entre pares APTOS
#     (pares 'verificar' listados à parte como candidatos condicionados)
# ---------------------------------------------------------


def _normalizar_pesos(*pesos):
    pesos = np.asarray(pesos, dtype=float)
    if np.any(pesos < 0) or pesos.sum() <= 0:
        raise ValueError("Pesos devem ser não negativos com soma > 0.")
    return pesos / pesos.sum()


PESO_SUITABILITY_BASE, PESO_ENV_BASE, PESO_FUNC_BASE = _normalizar_pesos(80.0, 10.0, 20.0)
PESOS_BASE = (PESO_SUITABILITY_BASE, PESO_ENV_BASE, PESO_FUNC_BASE)
OBJETIVOS = ["suitability_media", "compatibilidade_ambiental", "complementaridade_funcional"]
ORDEM_COLS = OBJETIVOS + ["Gramínea (Principal)", "Leguminosa (Acompanhante)"]


def mascara_nao_dominados(M, atol=1e-12):
    """Vetorizado: True para as linhas de M (a maximizar) não dominadas."""
    M = np.asarray(M, dtype=float)
    nd = np.ones(len(M), dtype=bool)
    for i in range(len(M)):
        domina = np.all(M >= M[i] - atol, axis=1) & np.any(M > M[i] + atol, axis=1)
        domina[i] = False
        nd[i] = not domina.any()
    return nd


def dominados_por(M_ref, M_teste, atol=1e-12):
    """True para cada linha de M_teste dominada por alguma linha de M_ref."""
    M_ref = np.asarray(M_ref, dtype=float)
    M_teste = np.asarray(M_teste, dtype=float)
    if len(M_ref) == 0:
        return np.zeros(len(M_teste), dtype=bool)
    return np.array([
        bool(np.any(np.all(M_ref >= m - atol, axis=1) & np.any(M_ref > m + atol, axis=1)))
        for m in M_teste
    ], dtype=bool)


def calcular_fronteira_pareto(cand, objetivos):
    if cand.empty:
        return cand.copy()
    return cand.loc[mascara_nao_dominados(cand[objetivos].to_numpy(float))].copy()


def score_pesos(df_pares, w, objetivos=OBJETIVOS, coluna="score_preferencia"):
    s = df_pares.copy()
    if s.empty:
        s[coluna] = pd.Series(dtype=float)
        return s
    w = _normalizar_pesos(*w)
    s[coluna] = (100.0 * sum(wi * s[o].astype(float) for wi, o in zip(w, objetivos))).round(4)
    return s.sort_values(
        [coluna] + ORDEM_COLS, ascending=[False] * (1 + len(OBJETIVOS)) + [True, True]
    ).reset_index(drop=True)


def recomendar_consorcios_pareto(cen, rotulo, top_k=None, w=PESOS_BASE, imprimir=True):
    """Retorna (pares_aptos, fronteira_rankeada, weighted_aptos, condicionados, diagnostico)."""
    av = avaliar_cenario(cen)
    todos, info = montar_pares(av, top_k, cen.get("arranjo", "consorcio_pastejo"))
    vazio = pd.DataFrame()
    diag = {"cenario": rotulo, **info}

    if todos.empty:
        if imprimir:
            print(f"[{rotulo}] Nenhum par não excluído ({info}).")
        return vazio, vazio, vazio, vazio, {**diag, "n_pares_apto": 0, "n_pareto": 0}

    aptos = todos[todos["status_par"] == "apto"].reset_index(drop=True)
    cond = todos[todos["status_par"] == "verificar"].reset_index(drop=True)

    if aptos.empty:
        weighted, rank_p = vazio, vazio
    else:
        weighted = score_pesos(aptos, w)
        weighted.insert(0, "rank_weighted_all", np.arange(1, len(weighted) + 1))
        fronteira = calcular_fronteira_pareto(aptos, OBJETIVOS)
        pares_p = set(zip(fronteira["Gramínea (Principal)"], fronteira["Leguminosa (Acompanhante)"]))
        weighted["pareto_eficiente"] = [
            (g, l) in pares_p
            for g, l in zip(weighted["Gramínea (Principal)"], weighted["Leguminosa (Acompanhante)"])
        ]
        rank_p = score_pesos(fronteira, w)
        rank_p.insert(0, "rank_pareto", np.arange(1, len(rank_p) + 1))

    # Condicionados: pares 'verificar' NÃO dominados por nenhum par apto, com os
    # objetivos calculáveis hoje. Isso NÃO garante que entrariam na fronteira:
    # confirmado o dado faltante, os próprios objetivos podem mudar.
    if cond.empty:
        condicionados = vazio
    else:
        M_ref = aptos[OBJETIVOS].to_numpy(float) if not aptos.empty else np.empty((0, len(OBJETIVOS)))
        dom = dominados_por(M_ref, cond[OBJETIVOS].to_numpy(float))
        condicionados = score_pesos(cond[~dom], w)

    diag.update({
        "n_pares_apto": int(len(aptos)),
        "n_pares_verificar": int(len(cond)),
        "n_pareto": int(len(rank_p)),
        "percentual_pareto": round(100.0 * len(rank_p) / len(aptos), 2) if len(aptos) else np.nan,
        "dominados_no_top10_weighted": int((~weighted.head(10)["pareto_eficiente"]).sum()) if not weighted.empty else 0,
        "n_condicionados_nao_dominados": int(len(condicionados)),
        "top1_pareto": (
            f"{rank_p.iloc[0]['Gramínea (Principal)']} + {rank_p.iloc[0]['Leguminosa (Acompanhante)']}"
            if not rank_p.empty else None
        ),
    })

    if imprimir:
        print("=" * 100)
        print(f"[{rotulo}] PARETO — PARES APTOS")
        print("=" * 100)
        print(f"arranjo: {cen.get('arranjo', 'consorcio_pastejo')}")
        for k in ["n_gramineas_aptas", "n_gramineas_verificar", "n_leguminosas_aptas", "n_leguminosas_fora_do_arranjo",
                  "n_leguminosas_verificar", "n_pares_excluidos_R6", "n_pares_apto",
                  "n_pares_verificar", "n_pareto", "percentual_pareto"]:
            print(f"{k}: {diag[k]}")
        cols = ["rank_pareto", "Gramínea (Principal)", "Leguminosa (Acompanhante)", *OBJETIVOS, "score_preferencia"]
        print("\n--- FRONTEIRA DE PARETO (APTOS) + RANKING DE PREFERÊNCIA ---")
        print(rank_p[cols].to_string(index=False) if not rank_p.empty else "Nenhum par apto.")
        print("\n--- CANDIDATOS CONDICIONADOS (status 'verificar', não dominados pelos aptos com os dados disponíveis) ---")
        print("Obs.: não é previsão de entrada na fronteira; scores 'parciais' não são comparáveis aos completos.")
        if condicionados.empty:
            print("Nenhum.")
        else:
            print(condicionados[[
                "Gramínea (Principal)", "Leguminosa (Acompanhante)", *OBJETIVOS,
                "score_preferencia", "base_do_score", "pendencias",
            ]].to_string(index=False))
    return aptos, rank_p, weighted, condicionados, diag


resultados_pareto, diag_pareto = {}, []
for nome_cen, cen in CENARIOS.items():
    print("\n" + "#" * 100)
    print(f"CENÁRIO: {nome_cen}")
    print("#" * 100)
    aptos_c, rank_c, weighted_c, cond_c, diag_c = recomendar_consorcios_pareto(cen, nome_cen)
    resultados_pareto[nome_cen] = (aptos_c, rank_c, weighted_c, cond_c)
    diag_pareto.append(diag_c)
    slug = _slug(nome_cen)
    if not weighted_c.empty:
        weighted_c.to_csv(f"pareto_weighted_aptos_{slug}.csv", index=False, encoding="utf-8-sig")
        rank_c.to_csv(f"pareto_front_aptos_{slug}.csv", index=False, encoding="utf-8-sig")
    if not cond_c.empty:
        cond_c.to_csv(f"pareto_condicionados_{slug}.csv", index=False, encoding="utf-8-sig")

print("\n" + "=" * 100)
print("RESUMO — PARETO NOS CENÁRIOS (somente pares aptos)")
print("=" * 100)
print(pd.DataFrame(diag_pareto).to_string(index=False))


# ---------------------------------------------------------
# 11b. COMPARAÇÃO DE ESTRATÉGIAS DE SELEÇÃO
#      individual  x  ranking direto  x  pós-Pareto
# ---------------------------------------------------------
# Convenções:
#  - Ranking direto = todos os pares APTOS ordenados por score_preferencia
#    (mesmos pesos do pós-Pareto), sem filtro de dominância.
#  - pos_score: posição pelo score; pares com score igual (4 casas decimais)
#    recebem a MESMA posição (método 'min'). pos_apresentacao: ordem de
#    exibição; desempate apenas por convenção (objetivos, depois nomes).
#  - Shortlist = os K_SHORTLIST primeiros na ordem de apresentação; se houver
#    empate de score cruzando a posição K, isso é sinalizado.

_CHAVE = ["Gramínea (Principal)", "Leguminosa (Acompanhante)"]


def _chave(r):
    return (str(r["Gramínea (Principal)"]), str(r["Leguminosa (Acompanhante)"]))


def _posicoes_com_empate(df_ord, col="score_preferencia"):
    d = df_ord.reset_index(drop=True).copy()
    d["pos_apresentacao"] = np.arange(1, len(d) + 1)
    d["pos_score"] = d[col].rank(method="min", ascending=False).astype(int)
    d["n_empatados_no_score"] = d.groupby(col)[col].transform("size").astype(int)
    return d


def _empate_na_fronteira(d, k, col="score_preferencia"):
    """Pares FORA do shortlist com o mesmo score do K-ésimo."""
    if len(d) <= k:
        return []
    s_k = d.iloc[k - 1][col]
    fora = d.iloc[k:]
    return [f"{a} + {b}" for a, b in zip(fora.loc[fora[col] == s_k, _CHAVE[0]], fora.loc[fora[col] == s_k, _CHAVE[1]])]


def _topo_com_empate(d):
    """Espécie de maior score; empates registrados (escolha por ordem alfabética só p/ reprodutibilidade)."""
    s_max = d["score"].max()
    emp = d[d["score"] == s_max].sort_values("especie")
    return emp.iloc[0], list(emp["especie"])


def selecao_individual(av, arranjo):
    apt = av[av["status"] == "apto"]
    gram = apt[apt["grupo"].str.contains("gramin", na=False)]
    leg = apt[apt["grupo"].str.contains("legumin", na=False) & (apt["score"] > 0)]
    leg = leg[leg["arranjo_leg"].isin(["herbacea", "indefinido"])] if arranjo == "consorcio_pastejo" \
        else leg[leg["arranjo_leg"] == "arbustiva_arborea"]
    if gram.empty or leg.empty:
        return None, [], []
    g, emp_g = _topo_com_empate(gram)
    l, emp_l = _topo_com_empate(leg)
    return avaliar_par_completo(g, l, arranjo), emp_g, emp_l


def comparar_estrategias(nome_cen, k=K_SHORTLIST, w=PESOS_BASE):
    aptos, rank_p, weighted, _ = resultados_pareto[nome_cen]
    av = avaliacoes[nome_cen]
    arranjo = CENARIOS[nome_cen].get("arranjo", "consorcio_pastejo")
    slug = _slug(nome_cen)

    print("\n" + "=" * 110)
    print(f"COMPARAÇÃO DE ESTRATÉGIAS — {nome_cen} | shortlist K = {k} (fixado a priori)")
    print("=" * 110)
    resumo = {"Cenário": nome_cen, "K": k}

    # ---------------- (b) e (c): ranking direto x pós-Pareto
    if weighted.empty:
        print("Sem pares aptos: comparação direta/pós-Pareto não se aplica.")
        d = p = pd.DataFrame()
    else:
        d = _posicoes_com_empate(weighted)          # já ordenado por score_pesos
        p = _posicoes_com_empate(rank_p)
        pos_d = {_chave(r): (int(r["pos_score"]), int(r["pos_apresentacao"])) for _, r in d.iterrows()}
        sl_d, sl_p = d.head(k), p.head(k)
        set_d = {_chave(r) for _, r in sl_d.iterrows()}
        set_p = {_chave(r) for _, r in sl_p.iterrows()}

        # Tabela 1 — posição dos pares eficientes no ranking direto
        tab_front = p[_CHAVE + OBJETIVOS + ["score_preferencia", "pos_score", "pos_apresentacao",
                                            "n_empatados_no_score"]].rename(columns={
            "pos_score": "pos_pareto_score", "pos_apresentacao": "pos_pareto_apresentacao"})
        tab_front["pos_direto_score"] = [pos_d[_chave(r)][0] for _, r in p.iterrows()]
        tab_front["pos_direto_apresentacao"] = [pos_d[_chave(r)][1] for _, r in p.iterrows()]
        tab_front["no_shortlist_direto"] = [_chave(r) in set_d for _, r in p.iterrows()]
        tab_front["no_shortlist_pos_pareto"] = [_chave(r) in set_p for _, r in p.iterrows()]
        print(f"\n[1] Pares Pareto-eficientes ({len(p)}) e sua posição no ranking direto ({len(d)} pares aptos):")
        print(tab_front[_CHAVE + ["score_preferencia", "pos_pareto_score", "pos_direto_score",
                                  "pos_direto_apresentacao", "n_empatados_no_score",
                                  "no_shortlist_direto", "no_shortlist_pos_pareto"]].to_string(index=False))
        tab_front.to_csv(f"comparacao_fronteira_no_ranking_direto_{slug}.csv", index=False, encoding="utf-8-sig")

        # Tabela 2 — shortlists
        inter, uniao = set_d & set_p, set_d | set_p
        dom_sl = sl_d[~sl_d["pareto_eficiente"]]
        emp_d, emp_p = _empate_na_fronteira(d, k), _empate_na_fronteira(p, k)
        tab_sl = pd.DataFrame({
            "pos": np.arange(1, k + 1),
            "Direto: par": [f"{a} + {b}" for a, b in zip(sl_d[_CHAVE[0]], sl_d[_CHAVE[1]])] + [""] * (k - len(sl_d)),
            "Direto: score": list(sl_d["score_preferencia"]) + [np.nan] * (k - len(sl_d)),
            "Direto: pos_score": list(sl_d["pos_score"]) + [np.nan] * (k - len(sl_d)),
            "Direto: dominado?": list(~sl_d["pareto_eficiente"]) + [np.nan] * (k - len(sl_d)),
            "Pós-Pareto: par": [f"{a} + {b}" for a, b in zip(sl_p[_CHAVE[0]], sl_p[_CHAVE[1]])] + [""] * (k - len(sl_p)),
            "Pós-Pareto: score": list(sl_p["score_preferencia"]) + [np.nan] * (k - len(sl_p)),
            "Pós-Pareto: pos_score": list(sl_p["pos_score"]) + [np.nan] * (k - len(sl_p)),
        })
        print(f"\n[2] Shortlists (top-{k}):")
        print(tab_sl.to_string(index=False))
        print(f"Sobreposição: {len(inter)}/{k} | Jaccard = {len(inter) / len(uniao):.3f}" if uniao else "Sobreposição: -")
        print(f"Pares DOMINADOS no shortlist direto: {len(dom_sl)}"
              + (" -> " + "; ".join(f"{a} + {b}" for a, b in zip(dom_sl[_CHAVE[0]], dom_sl[_CHAVE[1]])) if len(dom_sl) else ""))
        if emp_d:
            print(f"ATENÇÃO: empate de score na posição {k} do ranking direto com par(es) fora do shortlist: {emp_d}")
        if emp_p:
            print(f"ATENÇÃO: empate de score na posição {k} do pós-Pareto com par(es) fora do shortlist: {emp_p}")
        tab_sl.to_csv(f"comparacao_shortlists_{slug}.csv", index=False, encoding="utf-8-sig")

        resumo.update({
            "Pares aptos": len(d), "N Pareto": len(p),
            "Sobreposição shortlists (n)": len(inter),
            "Jaccard shortlists": round(len(inter) / len(uniao), 3) if uniao else np.nan,
            "Dominados no shortlist direto": len(dom_sl),
            "Empate na posição K (direto/pós-Pareto)": f"{'sim' if emp_d else 'não'}/{'sim' if emp_p else 'não'}",
            "Líder direto": f"{d.iloc[0][_CHAVE[0]]} + {d.iloc[0][_CHAVE[1]]}",
            "Líder pós-Pareto": f"{p.iloc[0][_CHAVE[0]]} + {p.iloc[0][_CHAVE[1]]}",
        })

    # ---------------- (a) referência individual
    linha, emp_g, emp_l = selecao_individual(av, arranjo)
    print("\n[3] Referência INDIVIDUAL (melhor gramínea apta + melhor leguminosa apta do arranjo):")
    if linha is None:
        print("Não há gramínea e leguminosa aptas no arranjo: referência individual não se aplica.")
        resumo.update({"Individual: par": None, "Individual: status": "não aplicável"})
    else:
        ind = score_pesos(pd.DataFrame([linha]), w)
        r = ind.iloc[0]
        chave_i = _chave(r)
        parcial = r["base_do_score"] != "completa"
        if not d.empty and chave_i in pos_d:
            pos_s, pos_a = pos_d[chave_i]
            efic = bool(d.loc[d.apply(_chave, axis=1) == chave_i, "pareto_eficiente"].iloc[0])
            efic_txt = "sim" if efic else "não (dominado por par apto)"
            dom_txt = "não" if efic else "sim"
        else:
            pos_s = pos_a = np.nan
            efic = None
            efic_txt = f"não avaliado (par '{r['status_par']}' não participa da fronteira dos aptos)"
            M_ref = aptos[OBJETIVOS].to_numpy(float) if not aptos.empty else np.empty((0, len(OBJETIVOS)))
            vals = r[OBJETIVOS].to_numpy(float).reshape(1, -1)
            if np.isnan(vals).any():
                dom_txt = "não determinável (objetivo sem valor)"
            else:
                dom = bool(dominados_por(M_ref, vals)[0])
                if dom:
                    dom_txt = ("sim (condicional: objetivos calculados com dados incompletos)"
                               if parcial else "sim")
                else:
                    dom_txt = ("não dominado com os dados disponíveis — NÃO demonstra eficiência"
                               + (" (objetivos parciais)" if parcial else ""))
        tab_ind = pd.DataFrame([{
            "Par": f"{chave_i[0]} + {chave_i[1]}",
            "Status do par": r["status_par"],
            "Motivo (se excluído)": r["motivos_exclusao_par"],
            "Pendências (se verificar)": r["pendencias"],
            **{o: r[o] for o in OBJETIVOS},
            "score_preferencia": r["score_preferencia"],
            "Base do score": r["base_do_score"],
            "Pos. no ranking direto (score)": pos_s if pd.notna(pos_s) else f"fora do ranking (par '{r['status_par']}')",
            "Pareto-eficiente": efic_txt,
            "Dominado por par apto": dom_txt,
            "No shortlist direto": (not d.empty) and chave_i in {_chave(x) for _, x in d.head(k).iterrows()},
            "No shortlist pós-Pareto": (not p.empty) and chave_i in {_chave(x) for _, x in p.head(k).iterrows()},
            "Empate no topo (gramínea)": "; ".join(emp_g) if len(emp_g) > 1 else "",
            "Empate no topo (leguminosa)": "; ".join(emp_l) if len(emp_l) > 1 else "",
        }])
        print(tab_ind.T.to_string(header=False))
        if r["status_par"] != "apto":
            print(f"OBS.: o par individual ficou '{r['status_par']}'. Registrado como está; NÃO foi substituído.")
        if parcial:
            print("OBS.: score PARCIAL e condicional — calculado sem todos os critérios/componentes;")
            print("      não compará-lo diretamente com scores de pares com base completa.")
        tab_ind.to_csv(f"comparacao_referencia_individual_{slug}.csv", index=False, encoding="utf-8-sig")
        resumo.update({
            "Individual: par": f"{chave_i[0]} + {chave_i[1]}",
            "Individual: status": r["status_par"],
            "Individual: pos. direto": pos_s,
            "Individual: Pareto-eficiente": efic_txt,
            "Individual: base do score": "completa" if not parcial else "parcial (condicional)",
            "Individual: empate no topo": "sim" if (len(emp_g) > 1 or len(emp_l) > 1) else "não",
        })
    return resumo


resumo_comparacao = [comparar_estrategias(nome_cen) for nome_cen in CENARIOS]
print("\n" + "=" * 110)
print(f"RESUMO — COMPARAÇÃO DE ESTRATÉGIAS (K = {K_SHORTLIST})")
print("=" * 110)
_df_rc = pd.DataFrame(resumo_comparacao)
for _c in ["Pares aptos", "N Pareto", "Sobreposição shortlists (n)", "Dominados no shortlist direto",
           "Individual: pos. direto"]:
    if _c in _df_rc.columns:
        _df_rc[_c] = pd.to_numeric(_df_rc[_c], errors="coerce").astype("Int64")
print(_df_rc.to_string(index=False))
_df_rc.to_csv("resumo_comparacao_estrategias.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------
# 12. NÍVEL DE SISTEMA — CAPACIDADE DE SUPORTE POTENCIAL (opcional)
# ---------------------------------------------------------
# A produtividade da base é POTENCIAL (em geral, ensaios adubados) e não
# depende do cenário (calagem, fertilidade). Por isso a UA/ha aqui é
# "potencial sob manejo intensivo", não a esperada no cenário avaliado.


def estimar_capacidade_suporte(prod, consumo_pct_pv=0.025, peso_ua_kg=450, eficiencia_pastejo=0.6):
    """MS (t/ha/ano) -> UA/ha. Confirmar unidade da produtividade na base."""
    if pd.isna(prod) or prod <= 0:
        return np.nan
    return round(prod * 1000 * eficiencia_pastejo / (peso_ua_kg * consumo_pct_pv * 365), 2)


FATOR_LER = 1.52  # EMBRAPA (Acre): Cynodon cv. Lua + A. pintoi Belmonte; extrapolação

if CALCULAR_UA:
    resumo_sistema = []
    for nome_cen, av in avaliacoes.items():
        if CENARIOS[nome_cen].get("arranjo", "consorcio_pastejo") != "consorcio_pastejo":
            continue  # LER de referência (Cynodon + A. pintoi) não se aplica a banco de proteína
        ind = av[av["status"] == "apto"].sort_values("score", ascending=False)
        if not ind.empty:
            r = ind.iloc[0]
            resumo_sistema.append({
                "Cenário": nome_cen, "Nível": "Espécie (apta)", "Líder": r["especie"], "Score": r["score"],
                "UA/ha potencial": estimar_capacidade_suporte(r["produtividade_media"]),
            })
        rank_c = resultados_pareto[nome_cen][1]
        if not rank_c.empty:
            p = rank_c.iloc[0]
            resumo_sistema.append({
                "Cenário": nome_cen, "Nível": "Consórcio (apto)",
                "Líder": f"{p['Gramínea (Principal)']} + {p['Leguminosa (Acompanhante)']}",
                "Score": p["score_preferencia"],
                "UA/ha potencial": estimar_capacidade_suporte(p["produtividade_graminea"] * FATOR_LER),
            })
    print("\n" + "=" * 100)
    print("CAPACIDADE DE SUPORTE POTENCIAL (manejo intensivo) — NÃO é a esperada no cenário")
    print("Limitações: produtividade não condicionada ao cenário; LER fixo, independente da leguminosa.")
    print("=" * 100)
    print(pd.DataFrame(resumo_sistema).to_string(index=False))

# ---------------------------------------------------------
# 13. ABLAÇÃO, SENSIBILIDADE PÓS-PARETO E SENSIBILIDADE AO CONJUNTO
#     DE GRAMÍNEAS (top-k) — cenário de referência, pares aptos
# ---------------------------------------------------------
cand_ref, rank_ref, _, _ = resultados_pareto[CENARIO_REF]


def _id(r):
    return (str(r["Gramínea (Principal)"]), str(r["Leguminosa (Acompanhante)"]))


def _nome(p):
    return None if p is None else f"{p[0]} + {p[1]}"


if rank_ref.empty or len(cand_ref) < 2:
    print("\nAblação/sensibilidades não executadas: poucos pares aptos no cenário de referência.")
else:
    # --- Ablação ---
    mapa_pesos = dict(zip(OBJETIVOS, PESOS_BASE))
    configs = {
        "Complete model": OBJETIVOS,
        "Without environmental compatibility": ["suitability_media", "complementaridade_funcional"],
        "Without functional complementarity": ["suitability_media", "compatibilidade_ambiental"],
        "Suitability only": ["suitability_media"],
    }
    front_base = set(_id(r) for _, r in rank_ref.iterrows())
    lider_base = _id(rank_ref.iloc[0])
    linhas_abl = []
    for nome_cfg, objs in configs.items():
        fr = calcular_fronteira_pareto(cand_ref, objs).copy()
        w_cfg = _normalizar_pesos(*[mapa_pesos[o] for o in objs])
        fr["score_ablacao"] = (100.0 * sum(wi * fr[o] for wi, o in zip(w_cfg, objs))).round(4)
        fr = fr.sort_values(["score_ablacao"] + objs, ascending=False).reset_index(drop=True)
        pares = set(_id(r) for _, r in fr.iterrows())
        inter, uniao = front_base & pares, front_base | pares
        lider = _id(fr.iloc[0]) if not fr.empty else None
        linhas_abl.append({
            "Configuration": nome_cfg, "N Pareto": len(fr),
            "Pareto (%)": round(100.0 * len(fr) / len(cand_ref), 2),
            "Overlap with full (n)": len(inter), "Jaccard": round(len(inter) / len(uniao), 4) if uniao else np.nan,
            "Baseline leader remains Pareto": lider_base in pares, "Top pair": _nome(lider),
            "Baseline leader retained as top": lider == lider_base,
        })
    print("\n" + "=" * 100)
    print(f"ABLAÇÃO MULTIOBJETIVO — {CENARIO_REF} (pares aptos: {len(cand_ref)})")
    print("=" * 100)
    df_abl = pd.DataFrame(linhas_abl)
    print(df_abl.to_string(index=False))
    df_abl.to_csv("ablation_multiobjective_summary.csv", index=False, encoding="utf-8-sig")

    # --- Sensibilidade pós-Pareto ---
    front = rank_ref[["Gramínea (Principal)", "Leguminosa (Acompanhante)", *OBJETIVOS]].copy()
    configs_w = {
        "Baseline": PESOS_BASE, "More suitability": (0.80, 0.07, 0.13),
        "More environmental compatibility": (0.60, 0.25, 0.15),
        "More functional complementarity": (0.60, 0.10, 0.30),
        "Balanced preferences": (1 / 3, 1 / 3, 1 / 3),
    }
    linhas_w = []
    for nome_cfg, w_cfg in configs_w.items():
        rk = score_pesos(front, w_cfg, coluna="score_sens")
        linhas_w.append({"Configuration": nome_cfg, "Top pair": _nome(_id(rk.iloc[0])),
                         "Top score": rk.iloc[0]["score_sens"]})
    print("\n" + "=" * 100)
    print("SENSIBILIDADE PÓS-PARETO — CONFIGURAÇÕES DISCRETAS")
    print("=" * 100)
    df_w = pd.DataFrame(linhas_w)
    print(df_w.to_string(index=False))
    df_w.to_csv("post_pareto_weight_sensitivity_discrete.csv", index=False, encoding="utf-8-sig")

    rng = np.random.default_rng(20260903)
    pesos_mc = rng.dirichlet(np.ones(3), size=10000)
    scores_mc = 100.0 * pesos_mc @ front[OBJETIVOS].astype(float).to_numpy().T
    ordem = np.argsort(-scores_mc, axis=1)
    ranks = np.empty_like(ordem)
    ranks[np.arange(ordem.shape[0])[:, None], ordem] = np.arange(1, ordem.shape[1] + 1)
    mc = pd.DataFrame({
        "Grass": front["Gramínea (Principal)"].values, "Legume": front["Leguminosa (Acompanhante)"].values,
        "Top-1 frequency (%)": [round(100.0 * np.mean(ordem[:, 0] == j), 2) for j in range(len(front))],
        "Mean rank": ranks.mean(axis=0).round(4),
    }).sort_values(["Top-1 frequency (%)", "Mean rank"], ascending=[False, True])
    print("\nMONTE CARLO — 10000 vetores de pesos (Dirichlet(1,1,1), seed=20260903)")
    print(mc.to_string(index=False))
    mc.to_csv("post_pareto_weight_sensitivity_monte_carlo_summary.csv", index=False, encoding="utf-8-sig")

    # --- Sensibilidade ao conjunto de gramíneas (top-k aptas) ---
    linhas_k = []
    for k in TOP_K_SENSIBILIDADE:
        with redirect_stdout(io.StringIO()):
            _, rank_k, _, _, diag_k = recomendar_consorcios_pareto(
                CENARIOS[CENARIO_REF], f"top-k={k}", top_k=k
            )
        pares_k = set(_id(r) for _, r in rank_k.iterrows()) if not rank_k.empty else set()
        inter, uniao = front_base & pares_k, front_base | pares_k
        linhas_k.append({
            "Gramíneas aptas (k)": "todas" if k is None else k,
            "Pares aptos": diag_k.get("n_pares_apto", 0), "N Pareto": diag_k.get("n_pareto", 0),
            "Jaccard vs todas": round(len(inter) / len(uniao), 4) if uniao else np.nan,
            "Top pair": _nome(_id(rank_k.iloc[0])) if not rank_k.empty else None,
            "Leader retained": (not rank_k.empty) and _id(rank_k.iloc[0]) == lider_base,
        })
    print("\n" + "=" * 100)
    print("SENSIBILIDADE AO CONJUNTO DE GRAMÍNEAS CANDIDATAS (top-k aptas por score)")
    print("=" * 100)
    df_k = pd.DataFrame(linhas_k)
    print(df_k.to_string(index=False))
    df_k.to_csv("topk_sensitivity_pareto_summary.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------
# 13b. SENSIBILIDADE AO LIMITE MÍNIMO DE TOLERÂNCIA A PASTEJO (gramínea)
# ---------------------------------------------------------
# O limite 'alta' é hipótese metodológica. Esta análise mostra o que muda
# com 'media' e 'muito alta' e quais gramíneas mudam de status.


def sensibilidade_limite_pastejo(nome_cen, limites=LIMITES_SENSIBILIDADE_PASTEJO):
    base = CENARIOS[nome_cen]
    linhas, status_por_limite, fronteiras = [], {}, {}
    for lim in limites:
        cen_l = dict(base, tol_min_graminea=lim)
        av_l = avaliar_cenario(cen_l)
        gram_l = av_l[av_l["grupo"].str.contains("gramin", na=False)]
        status_por_limite[lim] = gram_l.set_index("especie")["status"]
        with redirect_stdout(io.StringIO()):
            _, rank_l, _, _, diag_l = recomendar_consorcios_pareto(cen_l, f"lim={lim}", imprimir=False)
        fronteiras[lim] = set(_id(r) for _, r in rank_l.iterrows()) if not rank_l.empty else set()
        linhas.append({
            "Limite gramínea": lim,
            "Gramíneas aptas": int((gram_l["status"] == "apto").sum()),
            "Gramíneas excluídas por R3-nível (qualquer)": int(gram_l["motivos_exclusao"].str.contains(
                "abaixo do mínimo", na=False).sum()),
            "… só por R3-nível (decisivo)": int((gram_l["motivos_exclusao"].str.contains("abaixo do mínimo", na=False)
                & ~gram_l["motivos_exclusao"].str.contains("; ", na=False)).sum()),
            "Pares aptos": diag_l.get("n_pares_apto", 0),
            "N Pareto": diag_l.get("n_pareto", 0),
            "Líder (pesos base)": _nome(_id(rank_l.iloc[0])) if not rank_l.empty else None,
        })
    ref = fronteiras.get(TOL_PASTEJO_MIN_GRAMINEA, set())
    for ln in linhas:
        f = fronteiras[ln["Limite gramínea"]]
        ln["Jaccard vs limite adotado"] = round(len(f & ref) / len(f | ref), 4) if (f | ref) else np.nan

    print("\n" + "=" * 100)
    print(f"SENSIBILIDADE AO LIMITE DE TOLERÂNCIA A PASTEJO DA GRAMÍNEA — {nome_cen}")
    print(f"(limite adotado: '{TOL_PASTEJO_MIN_GRAMINEA}'; política p/ nível desconhecido: "
          f"'{POLITICA_NIVEL_DESCONHECIDO}')")
    print("=" * 100)
    df_l = pd.DataFrame(linhas)
    print(df_l.to_string(index=False))

    tab = pd.DataFrame(status_por_limite)
    muda = tab[tab.nunique(axis=1) > 1]
    print("\nGramíneas cujo status muda com o limite (justificar caso a caso):")
    print(muda.to_string() if not muda.empty else "Nenhuma.")
    df_l.to_csv(f"sensibilidade_limite_pastejo_{_slug(nome_cen)}.csv", index=False, encoding="utf-8-sig")
    muda.to_csv(f"gramineas_sensiveis_limite_pastejo_{_slug(nome_cen)}.csv", encoding="utf-8-sig")
    return df_l, muda


for _cen_sens in ["Baseline (Cerrado, pH 5.0, sem calagem)", "Baseline (Cerrado, pH 5.0, com calagem)"]:
    sensibilidade_limite_pastejo(_cen_sens)

# ---------------------------------------------------------
# 14. COMPARAÇÃO DE CLUSTERING (exploratória, opcional)
# ---------------------------------------------------------
if RODAR_COMPARACAO_CLUSTERING:
    from sklearn.cluster import DBSCAN, SpectralClustering
    from sklearn.manifold import MDS
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture

    try:
        from sklearn_extra.cluster import KMedoids
    except ImportError:
        KMedoids = None
        print("\nscikit-learn-extra ausente: PAM ignorado (!pip install scikit-learn-extra).")

    D = np.asarray(matriz_distancia, dtype=float)
    n = D.shape[0]
    labels_hac = melhor_labels
    resultados_clust = [{"Method": "Gower-HAC", "Best configuration": "complete linkage", "k": melhor_k,
                         "n": n, "Silhouette": melhor_sil, "Noise": 0, "ARI vs HAC": 1.0}]

    def _melhor(candidatos):
        return max(candidatos, key=lambda r: r["Silhouette"]) if candidatos else None

    if KMedoids is not None:
        cands = []
        for k in range(2, 7):
            lb = KMedoids(n_clusters=k, metric="precomputed", method="pam", init="k-medoids++",
                          random_state=20260908).fit_predict(D)
            cands.append({"Method": "PAM + Gower", "Best configuration": "PAM", "k": k, "n": n,
                          "Silhouette": silhouette_score(D, lb, metric="precomputed"), "Noise": 0,
                          "ARI vs HAC": adjusted_rand_score(labels_hac, lb)})
        resultados_clust.append(_melhor(cands))

    cands = []
    for gamma in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]:
        A = np.exp(-gamma * np.square(D))
        np.fill_diagonal(A, 1.0)
        for k in range(2, 7):
            lb = SpectralClustering(n_clusters=k, affinity="precomputed", assign_labels="kmeans",
                                    random_state=20260908, n_init=50).fit_predict(A)
            cands.append({"Method": "Spectral + Gower", "Best configuration": f"gamma={gamma}", "k": k, "n": n,
                          "Silhouette": silhouette_score(D, lb, metric="precomputed"), "Noise": 0,
                          "ARI vs HAC": adjusted_rand_score(labels_hac, lb)})
    resultados_clust.append(_melhor(cands))

    cands = []
    for eps in np.arange(0.03, 0.301, 0.01):
        for ms in [3, 4, 5, 6, 8, 10]:
            lb = DBSCAN(eps=float(eps), min_samples=ms, metric="precomputed").fit_predict(D)
            mask = lb != -1
            cl = np.unique(lb[mask])
            if len(cl) < 2 or len(cl) >= mask.sum():
                continue
            Ds = D[np.ix_(mask, mask)]
            lh = AgglomerativeClustering(n_clusters=len(cl), metric="precomputed", linkage="complete").fit_predict(Ds)
            cands.append({"Method": "DBSCAN + Gower", "Best configuration": f"eps={eps:.2f}, minPts={ms}",
                          "k": len(cl), "n": int(mask.sum()),
                          "Silhouette": silhouette_score(Ds, lb[mask], metric="precomputed"),
                          "Noise": int((~mask).sum()), "ARI vs HAC": adjusted_rand_score(lh, lb[mask])})
    if cands:
        resultados_clust.append(_melhor(cands))

    cands = []
    for ndim in [2, 3, 4, 5, 6]:
        X = MDS(n_components=ndim, dissimilarity="precomputed", random_state=20260908,
                n_init=10, max_iter=1000, normalized_stress="auto").fit_transform(D)
        for cov in ["full", "tied", "diag", "spherical"]:
            for k in range(2, 7):
                lb = GaussianMixture(n_components=k, covariance_type=cov, random_state=20260908,
                                     n_init=20).fit_predict(X)
                if len(np.unique(lb)) < 2:
                    continue
                cands.append({"Method": "GMM + Gower-MDS", "Best configuration": f"MDS={ndim}, {cov}",
                              "k": k, "n": n, "Silhouette": silhouette_score(D, lb, metric="precomputed"),
                              "Noise": 0, "ARI vs HAC": adjusted_rand_score(labels_hac, lb)})
    resultados_clust.append(_melhor(cands))

    print("\n" + "=" * 90)
    print("COMPARAÇÃO DE CLUSTERING (exploratória)")
    print("=" * 90)
    print(pd.DataFrame([r for r in resultados_clust if r]).to_string(
        index=False, float_format=lambda x: f"{x:.6f}"))
    print("Obs.: DBSCAN com muito ruído avalia só um subconjunto; não comparar silhouette diretamente.")