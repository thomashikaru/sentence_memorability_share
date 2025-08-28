# Load required libraries
library(ggplot2)
library(dplyr)
library(lme4)
library(lattice)
library(data.table)
library(lmtest)
library(lmerTest)
library(glue)
library(texreg)
library(caret)

# Set working directory to the script's parent directory
# This ensures relative file paths work correctly
script_path <- commandArgs(trailingOnly = FALSE)
script_path <- script_path[grep("--file=", script_path)]
if (length(script_path) > 0) {
  script_dir <- dirname(sub("--file=", "", script_path))
  setwd(script_dir)
} else {
  # If running interactively or not via Rscript, use current directory
  message("Running interactively - using current working directory")
}

# Define file paths and output directory
processed_data_file <- "trial_level_data.csv"
confusability_data_file <- "confusability_output/confusability_data.csv"
sentence_summary_data_file <- "sentence_memorability_data.csv"
output_dir <- "results/"

# Create output directory if it doesn't exist
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Define target predictors for analysis
target_predictors <- c(
  "sum_surp",
  "sum_surp_ngram",
  "pcfg_surp_sum",
  "avg_cosine_dist_broad_glove",
  "avg_cosine_dist_broad_sbert",
  "avg_cosine_dist_broad_use",
  "avg_cosine_dist_broad_electra",
  "avg_cosine_dist_broad_bert",
  "avg_cosine_dist_broad_gpt2"
)

# Load and preprocess sentence summary data
df <- fread(sentence_summary_data_file,
            select = c("sentence", "condition", "mean_mem", "avg_log_freq",
                       "acc", "sum_surp", "sum_surp_ngram", "pcfg_surp_sum",
                       "avg_cosine_dist_broad_glove",
                       "avg_cosine_dist_broad_sbert",
                       "avg_cosine_dist_broad_use",
                       "avg_cosine_dist_broad_electra",
                       "avg_cosine_dist_broad_bert-large-cased_layer-9",
                       "avg_cosine_dist_broad_gpt2-medium_layer-23"))

# Clean column names and add sentence length, then scale numeric variables
df <- df %>%
  rename(avg_cosine_dist_broad_bert =
           "avg_cosine_dist_broad_bert-large-cased_layer-9",
         avg_cosine_dist_broad_gpt2 =
           "avg_cosine_dist_broad_gpt2-medium_layer-23") %>%
  mutate(sentence_nchar = nchar(sentence)) %>%
  mutate(across(c("acc", "mean_mem", "avg_log_freq", "sum_surp",
                  "sum_surp_ngram", "pcfg_surp_sum",
                  "avg_cosine_dist_broad_glove",
                  "avg_cosine_dist_broad_sbert",
                  "avg_cosine_dist_broad_use",
                  "avg_cosine_dist_broad_electra",
                  "avg_cosine_dist_broad_bert",
                  "avg_cosine_dist_broad_gpt2"), scale))

# Filter data for fMRI condition and initialize results storage
data_filt <- df %>% filter(condition %in% c("fmri"))
dll_results <- data.frame(Fold = integer(), Predictor = character(),
                          DeltaLogLik = numeric(),
                          NormalizedDeltaLL = numeric(),
                          stringsAsFactors = FALSE)
folds <- createFolds(data_filt$sentence, k = 10)

# Cross-validation loop: compare each predictor against baseline model
for (i in seq_along(folds)) {
  data_sub <- data_filt[folds[[i]], ]
  form_baseline <- glue("acc ~ avg_log_freq + mean_mem")
  model_baseline <- data_sub %>%
    lm(data = ., as.formula(form_baseline))
  print(summary(model_baseline))

  # Test each target predictor
  for (target_predictor in target_predictors) {
    form <- glue("acc ~ avg_log_freq + mean_mem*{target_predictor}")

    model_target <- data_sub %>%
      lm(data = ., as.formula(form))
    print(summary(model_target))

    # Calculate Delta Log-Likelihood
    delta_loglik <- logLik(model_target) - logLik(model_baseline)

    # Normalize by the number of observations
    n <- nrow(data_sub)
    normalized_delta_ll <- as.numeric(delta_loglik) / n

    # Store results
    dll_results <- rbind(dll_results, data.frame(
      Fold = i, Predictor = target_predictor,
      DeltaLogLik = as.numeric(delta_loglik),
      NormalizedDeltaLL = normalized_delta_ll
    ))

    # Save detailed model results
    result <- as.data.frame(summary(model_target)$coefficients)
    result$formula <- form
    result$response_var <- "correct"
    result$predictor_var <- target_predictor
    identifier <- glue("stats_table_{target_predictor}_fold_{i}.csv")
    write.csv(result, paste(output_dir, identifier, sep = "/"))

    # Generate LaTeX table output
    target_predictor_friendly <- gsub("_", "-", target_predictor)
    latex_output <- texreg(
      model_target,
      digits = 4,
      caption = sprintf("%s as predictor", target_predictor_friendly),
      label = sprintf("table:%s", target_predictor_friendly),
      single.row = TRUE
    )
    output_file <- glue("{output_dir}/stats_table_{target_predictor}.txt")
    writeLines(latex_output, con = output_file)
  }
}

# Sort and save cross-validation results
dll_results <- dll_results[order(-dll_results$DeltaLogLik), ]
print(dll_results)
write.csv(dll_results, glue("{output_dir}/predictive_power.csv"),
          row.names = FALSE)

# Fit final models on full dataset for each predictor
for (target_predictor in target_predictors) {
  form <- glue("acc ~ avg_log_freq + mean_mem*{target_predictor}")

  model_target <- data_filt %>%
    lm(data = ., as.formula(form))
  print(summary(model_target))

  # Save final model results
  result <- as.data.frame(summary(model_target)$coefficients)
  result$formula <- form
  result$response_var <- "correct"
  result$predictor_var <- target_predictor
  identifier <- paste("stats_table", "_", target_predictor, ".csv", sep = "")
  write.csv(result, paste(output_dir, identifier, sep = "/"))

  # Generate final LaTeX table
  target_predictor_friendly <- gsub("_", "-", target_predictor)
  latex_output <- texreg(
    model_target,
    digits = 4,
    caption = sprintf("%s as predictor", target_predictor_friendly),
    label = sprintf("table:%s", target_predictor_friendly),
    single.row = TRUE
  )
  output_file <- paste(output_dir, "/stats_table", "_", target_predictor,
                       ".txt", sep = "")
  writeLines(latex_output, con = output_file)
}

# PART 2: Trial-level analysis with confusability data
# Load trial-level data with all predictors
dat_all <- fread(processed_data_file,
                 select = c("correct", "block_num", "repeat", "response",
                            "worker_id", "sentence", "condition", "mean_mem",
                            "avg_log_freq", "rt", "sum_surp", "sum_surp_ngram",
                            "pcfg_surp_sum", "avg_cosine_dist_broad_glove",
                            "avg_cosine_dist_broad_sbert",
                            "avg_cosine_dist_broad_use",
                            "avg_cosine_dist_broad_electra",
                            "avg_cosine_dist_broad_bert-large-cased_layer-9",
                            "avg_cosine_dist_broad_gpt2-medium_layer-23"))

# Load confusability data
dat_confus <- fread(confusability_data_file,
                    select = c("max_cos_sim_in_hist", "mean_cos_sim_in_hist",
                               "overlap_word_exists", "overlap_word"))

# Combine datasets and preprocess
dat_all <- cbind(dat_all, dat_confus)

dat_all <- dat_all %>%
  rename(avg_cosine_dist_broad_bert =
           "avg_cosine_dist_broad_bert-large-cased_layer-9",
         avg_cosine_dist_broad_gpt2 =
           "avg_cosine_dist_broad_gpt2-medium_layer-23") %>%
  mutate(sentence_nchar = nchar(sentence)) %>%
  mutate(across(c("mean_mem", "avg_log_freq", "sum_surp", "sum_surp_ngram",
                  "pcfg_surp_sum", "avg_cosine_dist_broad_glove",
                  "avg_cosine_dist_broad_sbert", "avg_cosine_dist_broad_use",
                  "avg_cosine_dist_broad_electra", "avg_cosine_dist_broad_bert",
                  "avg_cosine_dist_broad_gpt2", "max_cos_sim_in_hist",
                  "mean_cos_sim_in_hist"), scale))

# Mixed-effects model: predict correctness using confusability features
form_target <- glue("correct ~ avg_log_freq + mean_mem + max_cos_sim_in_hist + 
                   overlap_word_exists + (1 | worker_id) + (1 | sentence)")
model_target <- dat_all %>%
  filter(`repeat` == 0) %>%
  glmer(data = .,
        control = glmerControl(optimizer = "bobyqa"),
        as.formula(form_target),
        family = "binomial")
print(summary(model_target))

# Mixed-effects model: predict response times
form <- glue("rt ~ mean_mem + avg_log_freq + sentence_nchar + `repeat` + 
             avg_cosine_dist_broad_sbert + (1 | worker_id) + (1 | sentence)")

m1 <- dat_all %>%
  filter(`response` == 1) %>%
  lmer(data = ., as.formula(form))
print(summary(m1))
