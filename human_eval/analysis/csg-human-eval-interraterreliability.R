### check expert interrater reliability: ICC, should be > .60 to be considered "good"
# we'll do this separately for each DV (creativity, originality, surprise, value-=>effectiveness)
library(psych)

## compute ICCs for experts
# ICC crea .67 good
icc_creativity <- ICC(expert_wide %>% select(creativity_expert_2, creativity_expert_1))
print("ICC creativity expert 1 vs expert 2: ")
print(icc_creativity$results$ICC[6])

# ICC orig .63 good
icc_originality <- ICC(expert_wide %>% select(originality_expert_2, originality_expert_1))
print("ICC originality expert 1 vs expert 2: ")
print(icc_originality$results$ICC[6])

# ICC surprise  .68 good
icc_surprise <- ICC(expert_wide %>% select(surprise_expert_2, surprise_expert_1))
print("ICC surprise expert 1 vs expert 2: ")
print(icc_surprise$results$ICC[6])

# ICC value .74  good
icc_value <- ICC(expert_wide %>% select(value_expert_2, value_expert_1))
print("ICC value expert 1 vs expert 2: ")
print(icc_value$results$ICC[6])

# ICC author_ai .91 excellent
icc_author_ai <- ICC(expert_wide %>% select(author_ai_expert_2, author_ai_expert_1))
print("ICC author_ai expert 1 vs expert 2: ")
print(icc_author_ai$results$ICC[6])

rm(icc_author_ai, icc_creativity, icc_originality, icc_surprise, icc_value)

## compute ICCs for nonexperts
# ICC crea .71 good
icc_creativity <- ICC(nonexpert_wide %>% select(creativity_rater_0, creativity_rater_1,
                                             creativity_rater_2, creativity_rater_3,
                                             creativity_rater_4))
print("ICC creativity nonexperts: ")
print(icc_creativity$results$ICC[6])

# ICC orig .56 moderate
icc_originality <- ICC(nonexpert_wide %>% select(originality_rater_0, originality_rater_1,
                                                 originality_rater_2, originality_rater_3,
                                                 originality_rater_4))
print("ICC originality nonexperts: ")
print(icc_originality$results$ICC[6])

# ICC surprise  .55 moderate
icc_surprise <- ICC(nonexpert_wide %>% select(surprise_rater_0, surprise_rater_1,
                                              surprise_rater_2, surprise_rater_3,
                                              surprise_rater_4))
print("ICC surprise nonexperts: ")
print(icc_surprise$results$ICC[6])

# ICC value .43 moderate
icc_value <- ICC(nonexpert_wide %>% select(value_rater_0, value_rater_1,
                                           value_rater_2, value_rater_3,
                                           value_rater_4))
print("ICC value nonexperts: ")
print(icc_value$results$ICC[6])

# ICC author_ai .55 moderate
icc_author_ai <- ICC(nonexpert_wide %>% select(author_ai_rater_0, author_ai_rater_1,
                                               author_ai_rater_2, author_ai_rater_3,
                                               author_ai_rater_4))
print("ICC nonexperts: ")
print(icc_author_ai$results$ICC[6])

rm(icc_author_ai, icc_creativity, icc_originality, icc_surprise, icc_value)
