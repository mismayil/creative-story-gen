library(dplyr)
library(tidyr)
library(lme4)

# setwd
setwd("~/Research/creative-story-gen/")

# import data
story_dat <- read.csv("human_eval/story_data.csv")
story_id_dat <- read.csv("human_eval/story_id_data.csv")
autometrics_dat <- read.csv("human_eval/autometric_data.csv") 
claude_rating <- read.csv("llm_eval/claude-3-5-sonnet-20240620_judge_results.csv")
gemini_rating <- read.csv("llm_eval/gemini-1.5-pro_judge_results.csv")
gpt_rating <- read.csv("llm_eval/gpt-4o_judge_results.csv")

source("llm_eval/analysis/csg-llm-eval-dataprep.R")
source("llm_eval/analysis/csg-llm-eval-interraterreliability.R")

### descriptive statistics
# get Means of experts by storyid
llm_ratings_by_storyid <- llm_ratings_wide %>%
  rowwise() %>%
  summarize(story_ext_id = story_ext_id,
            creativity_mean = mean(c(creativity_claude, creativity_gemini, creativity_gpt), na.rm = TRUE),
            originality_mean = mean(c(originality_claude, originality_gemini, originality_gpt), na.rm = TRUE),
            surprise_mean = mean(c(surprise_claude, surprise_gemini, surprise_gpt), na.rm = TRUE),
            value_mean = mean(c(value_claude, value_gemini, value_gpt), na.rm = TRUE),
            author_ai_mean = mean(c(author_ai_claude, author_ai_gemini, author_ai_gpt), na.rm = TRUE),
            author_ai_median = median(c(author_ai_claude, author_ai_gemini, author_ai_gpt), na.rm = TRUE)) %>%
  left_join(story_dat) %>%
  left_join(autometrics_standardized) %>%
  # drop rows where autometrics aren't present
  filter(!(is.na(dsi))) %>%
  mutate(author_is_ai = ifelse(story_author == "AI", 1, 0))


# print descriptive stats for LLM ratings
llm_ratings_by_storyid %>% 
  filter(author_is_ai == 0) %>%
  select(creativity_mean, originality_mean, surprise_mean, value_mean) %>%
  describe()
llm_ratings_by_storyid %>% 
  filter(author_is_ai == 1) %>%
  select(creativity_mean, originality_mean, surprise_mean, value_mean) %>%
  describe()

# correlation matrix of relevant ratings and autometrics values
llm_ratings_corrs <- llm_ratings_by_storyid %>%
  # select relevant vars to correlate
  select(c(creativity_mean, originality_mean, surprise_mean, value_mean,
           author_ai_mean, dsi, novelty, surprise, theme_uniqueness, inv_homogen,
           length_in_unique_words, readability_flesch_ease, avg_constituency_tree_depth, 
           avg_pos_adj_ratio, avg_pos_adv_ratio, avg_pos_pron_ratio, avg_pos_noun_ratio)) %>%
  cor() %>%
  round(2) 

### Choose relevant variables for analysis:
# DVs: means/medians creativity and perhaps originality, surprise, value and author_ai
# IVs: novelty and surprise and clearly defined in paper
#      diversity choose from: inverse homogenization vs theme uniqueness 
#      complexity: use PCA to determine main component

# TODO HERE PCA

### ANALYSIS
# who's more creative? #and what if we control for # words?
ttest_llms <- lm(creativity_mean ~ author_is_ai, data = llm_ratings_by_storyid)
summary(ttest_llms)
# turing test accuracy - can they predict AI authors? correct 76% of the time
author_ai_pred <- ifelse(llm_ratings_by_storyid$author_ai_mean < .5, 0, 1)
table(llm_ratings_by_storyid$author_is_ai, author_ai_pred)
print("LLM turing test acc = ")
((158+170)/(158+50+53+170))
# predict llm scores by autometrics
lm_llms <- lm(data = llm_ratings_by_storyid, creativity_mean ~ novelty + surprise + inv_homogen  +
                length_in_unique_words)
summary(lm_llms)
car::vif(lm_llms)

lm_llms1 <- lm(data = llm_ratings_by_storyid, creativity_mean ~ novelty + surprise + inv_homogen  +
                n_gram_diversity + length_in_unique_words)
summary(lm_llms1)
car::vif(lm_llms1)
anova(lm_llms, lm_llms1)

lm_llms2 <- lm(data = llm_ratings_by_storyid, creativity_mean ~ novelty + surprise + inv_homogen  +
                 n_gram_diversity + length_in_unique_words + avg_constituency_tree_depth)
summary(lm_llms2)
anova(lm_llms1, lm_llms2)

# check what best predicts dsi:
lm_dsi <- lm(data = llm_ratings_by_storyid, dsi ~ surprise + inv_homogen  +
               n_gram_diversity + length_in_unique_words + avg_constituency_tree_depth)
summary(lm_dsi)
