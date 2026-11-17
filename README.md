# ML de crédito: prever inadimplência com regressão logística

Modelo de classificação de risco de crédito (bom pagador × calote) em R com **tidymodels**, usando a base
pública `credit_data` do pacote `modeldata`.

## Etapas

1. Divisão treino/teste (75/25) estratificada pelo desfecho.
2. Receita de pré-processamento: imputação pela mediana (numéricas) e pela moda (categóricas).
3. Regressão logística (GLM binomial) em um `workflow`.
4. Avaliação com corte de 50% (acurácia ≈ 79%, matriz de confusão).
5. Corte de 30%: pega mais calotes, mas recusa mais bons pagadores. É o trade-off entre risco e volume
   de crédito.

## Rodar

```r
install.packages(c("tidymodels", "modeldata"))
source("ml_credito.R")
```
