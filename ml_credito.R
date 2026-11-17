# ============================================================
# ML de crédito: prever calote (bad) ou pagamento (good)
# Regressão logística (GLM) com tidymodels
# ============================================================

library(tidymodels)

# ---- 1. Dados ----
data(credit_data, package = "modeldata")
glimpse(credit_data)
count(credit_data, Status)   # ~28% bad, 72% good

# ---- 2. Treino (75%) e teste (25%) ----
set.seed(42)
split  <- initial_split(credit_data, prop = 0.75, strata = Status)
treino <- training(split)
teste  <- testing(split)

# ---- 3. Receita: preencher NAs ----
# Números -> mediana | Categorias -> valor mais comum
receita <- recipe(Status ~ ., data = treino) |>
  step_impute_median(all_numeric_predictors()) |>
  step_impute_mode(all_nominal_predictors())

# ---- 4. Modelo: regressão logística (GLM binomial) ----
fluxo_log <- workflow() |>
  add_recipe(receita) |>
  add_model(logistic_reg() |> set_engine("glm"))

ajuste_log <- fit(fluxo_log, data = treino)

# ---- 5. Avaliação com corte de 50% ----
previsoes <- predict(ajuste_log, teste) |>
  bind_cols(teste |> select(Status))

accuracy(previsoes, truth = Status, estimate = .pred_class)   # ~79%
conf_mat(previsoes, truth = Status, estimate = .pred_class)

# ---- 6. Corte de 30%: pega mais calotes, barra mais bons ----
probs <- predict(ajuste_log, teste, type = "prob") |>
  bind_cols(teste |> select(Status))

previsoes30 <- probs |>
  mutate(palpite = if_else(.pred_bad > 0.3, "bad", "good"),
         palpite = factor(palpite, levels = c("bad", "good")))

conf_mat(previsoes30, truth = Status, estimate = palpite)

# ---- 7. Curva ROC e AUC (todos os cortes de uma vez) ----
roc_curve(probs, truth = Status, .pred_bad) |> autoplot()
roc_auc(probs, truth = Status, .pred_bad)   # ~0.818