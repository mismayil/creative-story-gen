### check expert interrater reliability: ICC, should be > .75 to be considered "good"
# see tutorial https://www.datanovia.com/en/lessons/intraclass-correlation-coefficient-in-r/
# we'll do this separately for each DV (creativity, originality, surprise, value-=>effectiveness, author_ai)
library(psych)

# ICC crea .95 excellent
icc_creativity <- ICC(llm_ratings_wide %>% select(creativity_claude, creativity_gemini, creativity_gpt))
print("ICC creativity 3 LLMs: ")
print(icc_creativity$results$ICC[6])

# ICC orig .94 excellent
icc_originality <- ICC(llm_ratings_wide %>% select(originality_claude, originality_gemini, originality_gpt))
print("ICC originality 3 LLMs: ")
print(icc_originality$results$ICC[6])

# ICC surprise .92 excellent
icc_surprise <- ICC(llm_ratings_wide %>% select(surprise_claude, surprise_gemini, surprise_gpt))
print("ICC surprise 3 LLMs: ")
print(icc_surprise$results$ICC[6])

# ICC value .86 excellent
icc_value <- ICC(llm_ratings_wide %>% select(value_claude, value_gemini, value_gpt))
print("ICC value 3 LLMs: ")
print(icc_value$results$ICC[6])

# ICC author_ai 0.43 fair to moderate agreement
icc_author_ai <- ICC(llm_ratings_wide %>% select(author_ai_claude, author_ai_gemini, author_ai_gpt))
print("ICC author_ai 3 LLMs: ")
print(icc_author_ai$results$ICC[6])
#table(llm_ratings_wide$author_ai_claude, llm_ratings_wide$author_ai_gemini)
#table(llm_ratings_wide$author_ai_gpt, llm_ratings_wide$author_ai_gemini)
rm(icc_creativity, icc_originality, icc_surprise, icc_value, icc_author_ai)
