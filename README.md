# analise-safatle-colonialidade

## Assunções
- A fonte primária do corpus são imagens (slides) publicadas no Instagram e baixadas do Drive.
- Atributos sensíveis (ex.: raça/grupo étnico, orientação sexual) **não devem ser inferidos** por nome, foto ou “dedução”; só entram quando existirem em fonte pública verificável.

## O que foi feito (até agora)
- **Download** das imagens do Drive para `data/images/` (15 arquivos).
- **OCR em lote (PT)** com pré-processamento e recorte do conteúdo do slide:
  - textos por imagem em `data/ocr/*.txt`
  - recortes auditáveis em `data/derived/crops/*.png`
- **Extração e deduplicação de nomes citados** (principalmente autores de referências):
  - `data/derived/people.csv` e `data/derived/people.json`
- **Enriquecimento via Wikidata/Wikipedia** (com links de fonte no dataset) para:
  - `genero` (P21)
  - `raca` = **grupo étnico** (P172) — *campo nomeado como “raça” por pedido do projeto, mas o dado é de “ethnic group”*
  - `orientacao_sexual` (P91) quando existir
  - `classe_ocupacional` (bucket baseado em `ocupacoes_wikidata` P106; transparente e revisável)
  - saídas: `data/derived/people_enriched.csv|json` e atualização de `web/data/people.json`
- **Página web com gráficos**:
  - `web/index.html` + `web/app.js`

## O que falta / pontos de atenção
- **Classe (no sentido socioeconômico)**: não existe de forma consistente/estruturada na Wikipedia/Wikidata. O projeto usa `classe_ocupacional` como proxy (ocupação), e `classe` segue `desconhecido` para preenchimento manual se você quiser uma tipologia própria.
- **Desambiguação**: alguns nomes muito genéricos podem cair em pessoa errada; por isso o dataset inclui:
  - `wikidata_url`, `wikipedia_pt/en` e `wikidata_query` para auditoria.
  - Recomenda-se revisar entradas com `wikipedia_pt` vazio e/ou descrições estranhas.

## Como rodar
### 1) Gerar OCR + nomes (base)
```bash
python3 /workspace/scripts/build_dataset.py
```

### 2) Enriquecer com Wikidata/Wikipedia (gera `people_enriched.*` e atualiza a web)
```bash
python3 /workspace/scripts/enrich_people_wikidata.py
```

### 3) Abrir o dashboard
```bash
cd /workspace/web
python3 -m http.server 8000
```
Abra `http://localhost:8000`.

## Plano de rollback (simples)
- Se o OCR piorar, rode novamente sem reprocessar (ou reforce OCR): `python3 /workspace/scripts/build_dataset.py --force-ocr`.
- Se o enriquecimento errar uma pessoa, edite manualmente a linha em `data/derived/people_enriched.csv` e regenere `web/data/people.json`.
