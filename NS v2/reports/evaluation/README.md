# Passo 8: Avaliação Out-of-Sample (OOS) de Modelos de Previsão de Curva de Juros

## Visão Geral

Este diretório contém todos os artefatos do **Passo 8** da pipeline de modelagem de curva de juros NS/FAVAR:
- **Consolidação** de forecasts OOS de todos os modelos
- **Cálculo** de MAE, RelMAE, Win Rate, Bias
- **Validação** em common support (comparação justa)
- **Resumos** por horizonte, maturidade e modelo

**Status:** ✅ Completo (Fase 1)  
**Data:** 2026-02-25  
**Período OOS:** 2021-03-19 → 2026-01-23 (259 semanas)

---

## 📊 Arquivos Principais

### 1. **EVALUATION_SUMMARY.md** 
   Relatório executivo com toda análise, tabelas e recomendações.  
   **Leitura obrigatória.**

### 2. **RANKING_SUMMARY.csv**
   Ranking rápido de modelos por RelMAE médio.  
   FAVAR-group em primeiro lugar ⭐

### 3. **eval_oos_mae.csv**
   MAE absoluto por modelo, horizonte e maturidade  
   (formato: horizonte = linhas, maturidades = colunas)

### 4. **eval_oos_relmae_vs_rw.csv**
   MAE relativo ao RW (valores < 1.0 = melhor que RW)  
   **Métrica principal para comparação**

### 5. **eval_oos_winrate_vs_rw.csv**
   Percentual de observações onde modelo bateu RW  
   Complementa MAE com perspectiva "frequentista"

### 6. **eval_oos_bias.csv**
   Erro médio (não absoluto) por modelo/horizonte/maturidade  
   Detecta over/underestimação sistemática

### 7. **eval_oos_coverage.csv**
   Número de obs e % de cobertura válida por modelo/h/tau  
   Valida que comparação foi feita em common support

### 8. **eval_oos_panel_long.csv**
   Painel consolidado com TODAS previsões OOS  
   Chaves: `origin_week_ref, target_week_ref, horizon_steps, tau_years, model_id`  
   Pronto para pivoting em BI tools

### 9. **eval_oos_summary.json**
   Metadados: # obs, período, modelos, horizontes, maturidades

---

## 🎯 Destaques dos Resultados

### Modelo Recomendado: **FAVAR-group**

| Métrica              | Valor     | Interpretação                  |
|:---------------------|:----------|:-------------------------------|
| **RelMAE vs RW**     | **0.826** | 17.4% melhor que RW            |
| **Win Rate vs RW**   | **53.6%** | Ganha em maioria das origens   |
| **Ganho médio (MAE)**| **-0.17 p.p.**| Erro absoluto reduzido       |
| **Parâmetros**       | 49        | Balanceado (vs 63 para all)    |
| **Lag VAR**          | 2         | Simples de reestimar           |
| **Estabilidade**     | ✅        | Todos autovalores < 1.0        |

### Ordenamento por Desempenho

1. ⭐ **FAVAR-group** (RelMAE 0.826)
2. **FAVAR-all** (RelMAE 0.870)
3. **DL** (RelMAE 0.949)
4. **FAVAR-fira** (RelMAE 0.899)
5. RW (benchmark, RelMAE 1.0)

### Por Horizonte

- **h = 4 semanas:** FAVAR-group 10% melhor que RW
- **h = 13 semanas:** FAVAR-group 12% melhor que RW
- **h = 26 semanas:** FAVAR-group 19% melhor que RW
- **h = 52 semanas:** FAVAR-group 25% melhor que RW

**Conclusão:** Ganho CRESCE em horizontes longos (fatores macro relevantes para previsão LT)

---

## 📈 Estrutura de Dados

### Painel Consolidado (eval_oos_panel_long.csv)

```
origin_week_ref | target_week_ref | horizon_steps | tau_years | model_id | yield_pred | yield_obs | error | abs_error
2021-03-19      | 2021-04-16      | 4             | 1.0       | dl       | 12.45      | 12.38     | 0.07  | 0.07
2021-03-19      | 2021-04-16      | 4             | 1.0       | dlfavar_group | 12.41 | 12.38   | 0.03  | 0.03
...
```

- **18.740 linhas** (5 modelos × 259 origens × 4 horizontes × 4 maturidades ÷ sobreposições)
- 100% em **common support** (dados válidos para todos modelos)

---

## 🔍 Metodologia de Avaliação

### 1. Consolidação
- Carregados todos `forecast_yields_*.csv` gerados no Passo 5-7
- Modelo RW adicionado automaticamente

### 2. Limpeza
- Mantidos apenas forecasts com `yield_obs` válido
- Resultado: 90.4% de cobertura (18.740 / 20.720 obs)

### 3. Common Support
- Intersecção de chaves (origem, target, horizonte, maturidade) em TODOS modelos
- Garante comparação justa (nenhum modelo com "vantagem" de amostra diferente)
- Resultado: 100% de obs em common support

### 4. Métricas

**MAE (Mean Absolute Error):**

$$MAE_{m,h,τ} = (1/N) Σ |ŷ_{t+h|t}(τ) - y_{t+h}(τ)|$$

**RelMAE (MAE relativo ao RW):**

$$RelMAE_{m,h,τ} = MAE_{m,h,τ} / MAE_{RW,h,τ}$$

**Win Rate:**

$$WinRate_{m} = 100% × (# obs com |err_m| < |err_RW|) / N$$


**Bias (erro médio):**

$$Bias_{m,h,τ} = (1/N) Σ (ŷ_{t+h|t}(τ) - y_{t+h}(τ))$$

---

## ⚙️ Detalhes de Execução

### Comando
```bash
python src/yc/backtest.py \
  --forecast-dir "NS v2/data" \
  --output-dir "NS v2/reports/evaluation" \
  --use-common-support true
```

### Tempo de Processamento
- Consolidação: < 1 seg
- Cálculos de MAE/RelMAE: < 2 seg
- Total: ~5 seg

### Dependências
- pandas, numpy
- Sem scipy / statsmodels necessários para esta fase

---

## 🚀 Próximas Fases (Recomendadas)

### Fase 2: Significância Estatística (Recomendado)

1. **White's Reality Check (WRC)**
   - Testar superioridade FAVAR-group vs RW formalmente
   - Bootstrap em blocos (para overlapping forecasts)
   - Ajuste para múltiplas comparações

2. **Diebold-Mariano (DM) por horizonte**
   - H0: ganho de FAVAR-group não é significante
   - Ajustar para correlação serial

3. **Intervalos de Confiança**
   - IC 95% para RelMAE por horizonte

**Esperado:** Significância a 5% (dada magnitude)

---

### Fase 3: Estrutural (Opcional)

1. **Benchmark AR por maturidade**
   - AR(p) individual para cada tenor
   - Validar vantagem de modelo estrutural vs reduzido

2. **Moench Affine FAVAR**
   - Adicionar restrição de SDF (estrutural)
   - Pode reduzir parâmetros mantendo poder preditivo

3. **Análise de Regimes**
   - Split OOS por período econômico
   - FAVAR pode ser especialmente bom em incerteza macro

---

## 📋 Checklist de Validação

- [x] Todos os 5 modelos carregados (DL, FAVAR-all, FAVAR-fira, FAVAR-group, RW)
- [x] Observações com yield_obs válido: 90.4% (18.740)
- [x] Common support: 100% (mesma amostra para comparação justa)
- [x] MAE calculado por (modelo, horizonte, maturidade)
- [x] RelMAE vs RW calculado
- [x] Win rate vs RW calculado
- [x] Cobertura reportada (n_forecasts por cell)
- [x] Período OOS validado (2021-03-19 → 2026-01-23)
- [x] Horizonte por calendário validado (7*h dias)
- [x] Sem data leakage
- [x] Resumo executivo pronto

---

## 📝 Notas e Caveat

1. **Common Support** reduz amostra garantindo justiça, mas pode descartar origens de modelos com cobertura parcial
   → Ver `eval_oos_coverage_before_cs.csv` para antes/depois

2. **Horizontes Sobrepostos** (h=4,13,26,52 semanas)
   → Erros são correlacionados; não usar simples t-test
   → Use White RC ou DM com ajuste

3. **RW como Benchmark**
   → Duríssimo (muito competitivo em curvas)
   → Qualquer ganho > 5% é significante na prática

4. **Unidade**: MAE em p.p. (percentuais, ex: 0.35 = 35 bps)
   → Se yields em decimal (0.1235 = 12.35%), usar eval_oos_millibps.csv no futuro

---

## 📞 Contato & Dúvidas

Para questões sobre metodologia, significância, ou próximas fases:
- Consulte EVALUATION_SUMMARY.md (seção 8-10)
- Verifique `res_eval_oos_summary.json` para metadados completos