# Projeto Final MCEQ – reprodutibilidade

## Objetivo
Este pacote reproduz a análise da base METR e preserva o desfecho inicialmente indicado no Marco 2.

- **Análise principal (protocolo do Marco 2):** `initial_implementation_time` (tempo até o pull request inicial), em minutos.
- **Análise secundária:** `total_implementation_time` = implementação inicial + pós-revisão, com imputação dos valores pós-revisão ausentes pela média do grupo, conforme o script oficial da METR.

A versão CSV acessada em 23/11/2026 contém 246 tarefas e 16 desenvolvedores (136 com IA permitida; 110 sem IA). A variável `ai_treatment` representa permissão de uso de IA; não identifica exclusivamente o GitHub Copilot.

## Como executar no computador
1. Instale Python 3.10 ou superior.
2. Abra o terminal nesta pasta.
3. Instale as dependências:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Execute:

   ```bash
   python analise.py
   ```

O script baixa `data_complete.csv` do repositório público da METR caso o arquivo não esteja presente. A base bruta é salva em `data/brutos/` e não é editada manualmente.

## Como executar no Google Colab
Faça upload de `analise.py` ou copie o seu conteúdo para uma célula. Instale as dependências com `!pip install -r requirements.txt` (ou `!pip install pandas numpy statsmodels scipy matplotlib`) e execute `!python analise.py`. É necessário que o Colab tenha acesso à internet para baixar o CSV.

## Arquivos produzidos
A pasta `resultados/` contém:
- `descritivas_desfecho_primario.csv`: média, desvio-padrão e mediana do tempo inicial e previsão sem IA por grupo;
- `descritivas_tempo_total_secundario.csv`: estatísticas do tempo total, com imputação publicada;
- `resultados_modelos.csv`: estimativas, IC95%, p bilateral aproximado normal e p unilateral para a H1 original (redução);
- `figura_efeitos_HC3.png`: figura com os efeitos primário (tempo inicial) e secundário (tempo total), ambos com IC95% HC3.
- `resultados/`: tabelas CSV geradas na execução que acompanha esta entrega.

## Modelo
Para cada desfecho positivo, o script ajusta por OLS:

`log(tempo) ~ ai_treatment + log(predicted_time_no_ai)`

O resultado é apresentado como `100 * (exp(beta_AI) - 1)`. São calculados erros-padrão HC3, agrupados por `dev_id` (sensibilidade) e não robustos. O teste direcional original é `H1: beta_AI < 0`; portanto, um coeficiente positivo não apoia a hipótese de redução.

## Procedimentos e limites
- Unidade de análise: tarefa/issue; os mesmos desenvolvedores contribuem com tarefas repetidas.
- Não se deve interpretar “IA permitida” como “IA efetivamente utilizada em todas as tarefas”.
- Não atribuir os resultados ao GitHub Copilot: o estudo utilizou principalmente Cursor Pro com modelos Claude 3.5/3.7 Sonnet.
- O tempo não mede qualidade, segurança ou manutenção do código.
- O conjunto CMU consultado não informa claramente uma licença própria de redistribuição do CSV. Por cautela, este pacote não redistribui o CSV; o script baixa a fonte original quando executado.
- Não confundir análise planejada no Marco 2 (tempo inicial) com a análise secundária de replicação do resultado total publicado pela METR.

## Fontes
- Página/dicionário CMU: https://cmustatistics.github.io/data-repository/technology/metr-ai.html
- CSV METR: https://raw.githubusercontent.com/METR/Measuring-Early-2025-AI-on-Exp-OSS-Devs/main/data_complete.csv
- Código e README original: https://github.com/METR/Measuring-Early-2025-AI-on-Exp-OSS-Devs
- Artigo: https://arxiv.org/abs/2507.09089

Data de acesso consultada para o relatório: 23/11/2026.

## Resultado da execução conferida em 23/11/2026

Na cópia CSV consultada, o desfecho principal (`initial_implementation_time`) apresentou médias de 119,54 min (DP 118,11; mediana 76) na condição IA permitida e 90,90 min (DP 89,40; mediana 65) na condição sem IA. A regressão ajustada estimou +19,83% (IC95% HC3: +1,79% a +41,06%); o p unilateral para a hipótese de redução (`beta_AI < 0`) foi 0,985. A sensibilidade com erro-padrão agrupado por desenvolvedor produziu IC95% de −0,46% a +44,25%.

Como desfecho secundário, o tempo total apresentou +18,84% (IC95% HC3: +1,30% a +39,41%), reproduzindo o resultado publicado pela METR. A cópia CSV baixada continha 16 valores pós-revisão ausentes (6,5%); eles foram imputados pela média do grupo de tratamento. O texto do artigo original menciona 27 issues (~12%), diferença que deve ser observada como possível diferença de versão/recorte dos dados.
