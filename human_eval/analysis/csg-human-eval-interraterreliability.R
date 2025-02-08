### check expert interrater reliability: ICC, should be > .75 to be considered "good"
# see tutorial https://www.datanovia.com/en/lessons/intraclass-correlation-coefficient-in-r/
# we'll do this separately for each DV (creativity, originality, surprise, value-=>effectiveness)
library(psych)

# ICC crea .67 moderate
icc_creativity <- ICC(expert_wide %>% select(creativity_expert_2, creativity_expert_1))
print("ICC creativity expert 1 vs expert 2: ")
print(icc_creativity$results$ICC[6])

# ICC orig .63 moderate
icc_originality <- ICC(expert_wide %>% select(originality_expert_2, originality_expert_1))
print("ICC originality expert 1 vs expert 2: ")
print(icc_originality$results$ICC[6])

# ICC surprise  .68 moderate
icc_surprise <- ICC(expert_wide %>% select(surprise_expert_2, surprise_expert_1))
print("ICC surprise expert 1 vs expert 2: ")
print(icc_surprise$results$ICC[6])

# ICC value .74 moderate to good
icc_value <- ICC(expert_wide %>% select(value_expert_2, value_expert_1))
print("ICC value expert 1 vs expert 2: ")
print(icc_value$results$ICC[6])

# ICC author_ai .91 excellent
icc_author_ai <- ICC(expert_wide %>% select(author_ai_expert_2, author_ai_expert_1))
print("ICC author_ai expert 1 vs expert 2: ")
print(icc_author_ai$results$ICC[6])

