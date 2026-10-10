from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(globals().get("DIRETORIO_EXECUCAO", Path(__file__).resolve().parent))
df = pd.read_csv(BASE / 'db_pastagem2_original.csv')
assert len(df) == 112 and df.id.is_unique
orig = df[df.id.isin([58, 109])].copy()
assert len(orig) == 2
orig.to_csv(BASE / 'registros_58_109_originais.csv', index=False, encoding='utf-8-sig')
doc = 'https://www.infoteca.cnptia.embrapa.br/infoteca/bitstream/doc/511279/1/doc112.pdf'
syn = 'https://www.infoteca.cnptia.embrapa.br/handle/doc/1182956'
tf = 'https://tropicalforages.info/text/entities/urochloa_brizantha.htm'
root = 'https://doi.org/10.4025/actasciagron.v32i2.1020'
al = 'https://doi.org/10.11606/D.64.2014.tde-12122014-110257'
log = []
def edit(field, value, source, locator, justification, scope='cultivar'):
    old58 = orig.loc[orig.id.eq(58), field].iloc[0]
    old109 = orig.loc[orig.id.eq(109), field].iloc[0]
    df.loc[df.id.eq(109), field] = value
    log.append(dict(id_canonico=109, id_alias=58, atributo=field,
                    valor_id58=old58, valor_id109=old109, valor_adotado=value,
                    escopo_fonte=scope, fonte=source, localizador=locator,
                    justificativa=justification))

log.append(dict(id_canonico=109, id_alias=58, atributo='identidade',
                valor_id58='Xaraés', valor_id109='MG-5 Vitória',
                valor_adotado='ID 109: Xaraés = MG-5 Vitória = Toledo',
                escopo_fonte='cultivar', fonte=syn+' ; '+tf,
                localizador='Capim-xaraés; Cultivars: CIAT 26110 / BRA 004308',
                justificativa='ID 109 preservado somente para continuidade; seus atributos não têm prioridade automática. ID 58 removido antes de imputação, Gower e triagem.'))
edit('altura_min', np.nan, doc, 'p. 12', 'Fonte informa até 1,5 m; não estabelece mínimo morfológico. Não usar altura manejada como mínimo.')
edit('altura_max', 1.5, doc, 'p. 12', 'Limite superior morfológico documentado; altura representativa será o único limite disponível.')
edit('profundidade_raiz_min', np.nan, root, 'Resumo e métodos', 'Profundidade efetiva/distribuição amostrada não define mínimo do intervalo de profundidade máxima.')
edit('profundidade_raiz_max', np.nan, root, 'Resumo: 80% da massa de raízes', 'Não confirma 1,5 ou 2,5 m como profundidade máxima. Suspenso para decisão; KNN só na representação Gower.')
for field in ['ph_min', 'ph_max']:
    edit(field, np.nan, tf+' ; '+doc, 'Soil requirements; Clima e solo', 'Faixa 4–8 refere-se à espécie, não valida os intervalos conflitantes da cultivar. Limite não reconciliado; suspenso para decisão.', 'espécie; não confirma intervalo da cultivar')
edit('tolerancia_aluminio', np.nan, al, 'Resumo; ensaio em solução nutritiva', 'Maior tolerância relativa de Xaraés não estabelece uma classe absoluta alta/moderada comparável à codificação. Evidência preservada; classe ausente.')
edit('tolerancia_acidez', np.nan, tf, 'Soil requirements', 'Descrição de espécie não confirma classificação alta específica deste registro.', 'espécie')
for field in ['precipitacao_min', 'precipitacao_max']:
    edit(field, np.nan, tf, 'Cultivars: CIAT 26110', 'Seleção sob chuva >1600 mm não é limiar fisiológico mínimo nem limite máximo; 800–1600 da base sem proveniência confirmada.')
for field in ['temp_min', 'temp_max']:
    edit(field, np.nan, doc, 'Clima e solo', 'Não foi confirmado intervalo numérico específico da cultivar nas fontes verificadas.')
for field in ['produtividade_minima_ms', 'produtividade_maxima_ms']:
    edit(field, np.nan, doc, 'p. 16', 'Fonte informa produção acima de 18 t MS/ha/ano no Acre e outra avaliação de 21 t/ha. Não converter valor censurado e ensaios diferentes em intervalo 12–26; produtividade numérica pendente.')
edit('tolerancia_pastejo', np.nan, doc, 'p. 26: Manejo do pastejo', 'Uso contínuo/rotacionado comprovado, porém sem classe ordinal alta/muito alta; não atribuir intensidade a partir do uso.')
edit('pastejo', 1, doc, 'p. 26', 'Uso sob pastejo documentado; não substitui requisito de nível ordinal.')
edit('pastagem', 1, doc, 'pp. 26–28', 'Uso em pastagens documentado.')
edit('exigencia_fertilidade', 'media', doc, 'p. 13', 'Exigência média expressamente documentada.')
edit('habito', 'cespitoso', doc, 'p. 12', 'Hábito morfológico documentado.')
edit('ciclo', 'perene', tf, 'Morphological description', 'Ciclo perene documentado para a espécie.', 'espécie')
for field in ['tropical', 'cerrado', 'tropical_umido']:
    edit(field, 1, doc, 'p. 12: Clima e solo', 'Indicação explícita para tropical de Cerrados e tropical úmido.')
edit('origem', 'Burundi, África', doc, 'p. 11', 'Origem da cultivar documentada.')
edit('tolerancia_encharcamento', 'moderada', doc, 'p. 13', 'Fonte classifica tolerância moderada ao alagamento em estudos controlados; não interpretar como recomendação irrestrita para solo alagado.')
edit('tolerancia_sombra', 'moderada', tf, 'Light', 'Classe documentada para a espécie; escopo não cultivar-específico.', 'espécie')
edit('textura_solo', 'arenoso-medio-argiloso', tf, 'Soil requirements', 'Amplitude de texturas para a espécie; média dos três códigos é regra operacional do modelo.', 'espécie')

# Os campos não usados nestas decisões mantêm valores legados, sem certificação nova.
df = df.loc[~df.id.eq(58)].reset_index(drop=True)
assert len(df) == 111 and df.id.is_unique and 58 not in set(df.id)
df.to_csv(BASE / 'db_pastagem2.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(log).to_csv(BASE / 'auditoria_reconciliacao_xaraes_mg5.csv', index=False, encoding='utf-8-sig')
print('Base original: 112 registros; base operacional: 111; ID 58 alias do ID 109.')
