library(dplyr)
library(tidyr)
library(lme4)

# setwd
setwd("~/Research/creative-story-gen/human_eval")

# import data
story_dat <- read.csv("story_data.csv")
story_id_dat <- read.csv("story_id_data.csv")
autometrics_dat <- read.csv("autometric_data.csv") 
expert_dat <- read.csv("raw_expert_data.csv")
nonexpert_dat <- read.csv("filtered_non_expert_data.csv")

source("analysis/csg-human-eval-datacleaning.R")
#source("analysis/csg-human-eval-interraterreliability.R")

### descriptive statistics
# get Means of experts by storyid
expert_by_storyid <- expert_wide %>%
  rowwise() %>%
  summarize(story_id = story_id,
            creativity_mean = mean(c(creativity_expert_2, creativity_expert_1), na.rm = TRUE),
            originality_mean = mean(c(originality_expert_2, originality_expert_1), na.rm = TRUE),
            surprise_mean = mean(c(surprise_expert_1, surprise_expert_2), na.rm = TRUE),
            value_mean = mean(c(value_expert_2, value_expert_1), na.rm = TRUE),
            author_ai_mean = mean(c(author_ai_expert_2, author_ai_expert_1), na.rm = TRUE)) %>%
  left_join(story_dat) %>%
  left_join(autometrics_standardized) %>%
  # drop rows where autometrics aren't present
  filter(!(is.na(dsi)))
  
# correlation matrix of relevant ratings and autometrics values
expert_corrs <- expert_by_storyid %>%
  # select relevant vars to correlate
  select(c(creativity_mean, originality_mean, surprise_mean, value_mean,
           author_ai_mean, dsi, novelty, surprise, theme_uniqueness, inv_homogen,
           length_in_unique_words, readability_flesch_ease, avg_constituency_tree_depth, 
           avg_pos_adj_ratio, avg_pos_adv_ratio, avg_pos_pron_ratio, avg_pos_noun_ratio)) %>%
  cor() %>%
  round(2) 

# get Means, Medians and Mode of non-experts by storyid
nonexpert_by_storyid <- nonexpert_dat %>%
  select(-c(pred_author, rater, user_id)) %>%
  group_by(story_id) %>%
  summarize(creativity_median = median(creativity, na.rm = TRUE),
            originality_median = median(originality, na.rm = TRUE),
            surprise_median = median(surprise, na.rm = TRUE),
            value_median = median(value, na.rm = TRUE),
            author_ai_median = median(author_ai, na.rm = TRUE)) %>%
  left_join(story_dat) %>%
  left_join(autometrics_standardized)
#creativity_mean = mean(creativity, na.rm = TRUE),
#creativity_sd = sd(creativity, na.rm = TRUE), 
#creativity_min = min(creativity, na.rm = TRUE),
#creativity_max = max(creativity, na.rm = TRUE),
#creativity_n = n()) %>%

# correlation matrix of relevant ratings and autometrics values
nonexpert_corrs <- nonexpert_by_storyid %>%
  # select relevant vars to correlate
  select(c(creativity_median, originality_median, surprise_median, value_median,
           author_ai_median, dsi, novelty, surprise, theme_uniqueness, inv_homogen,
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
# predict expert and non-expert scores by autometrics
lm_expert <- lm(data = expert_by_storyid, creativity_mean ~ novelty + surprise + inv_homogen  +
                     length_in_unique_words)
summary(lm_expert)
car::vif(lm_expert2)

lm_expert2 <- lm(data = expert_by_storyid, creativity_mean ~ novelty + surprise + inv_homogen  +
                  length_in_unique_words + avg_constituency_tree_depth)
summary(lm_expert2)
anova(lm_expert, lm_expert2)

lm_nonexpert <- lm(data = nonexpert_by_storyid, creativity_median ~ novelty + surprise + inv_homogen +
     length_in_unique_words)
summary(lm_nonexpert)
car::vif(lm_nonexpert)

lm_nonexpert2 <- lm(data = nonexpert_by_storyid, creativity_median ~ novelty + surprise + inv_homogen +
                     length_in_unique_words + avg_constituency_tree_depth)
summary(lm_nonexpert2)
anova(lm_nonexpert, lm_nonexpert2)
