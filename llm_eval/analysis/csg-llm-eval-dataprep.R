### data cleaning and preparation

# standardize data from autometrics
story_ext_ids <- autometrics_dat$story_ext_id
autometrics_standardized <- autometrics_dat %>%
  select(-c(story_ext_id)) %>%
  mutate_all(~(scale(.) %>% as.vector)) %>%
  cbind(story_ext_id = story_ext_ids)
rm(story_ext_ids)

# join story id and story data
orig_story_dat <- story_id_dat %>%
  select(story_id, story_ext_id) %>%
  left_join(story_dat)
story_dat <- orig_story_dat
rm(orig_story_dat, story_id_dat)

## llm ratings
# add model name to separate rating dfs
claude_rating$rater = "claude"
gemini_rating$rater = "gemini"
gpt_rating$rater = "gpt"
# join llm ratings data
llm_ratings <- claude_rating %>%
  rbind(gemini_rating) %>%
  rbind(gpt_rating)
rm(claude_rating, gemini_rating, gpt_rating)
# add vars, make right types
llm_ratings <- llm_ratings %>%
  mutate(author_ai = ifelse(pred_author == "AI", 1, 0)) %>%
  # make variables right type
  mutate(creativity = as.numeric(creativity),
         originality = as.numeric(originality),
         surprise = as.numeric(surprise),
         value = as.numeric(value),
         author_ai = as.numeric(author_ai),
         rater = as.factor(rater))

## prep llm ratings data
# llm ratings in long format
llm_ratings_long <- llm_ratings %>% 
  select(-c(story_author, pred_author)) %>%
  pivot_longer(cols=c(creativity, originality, surprise, value, author_ai),
               names_to='rated',
               values_to='rating')

# expert data in wide format
llm_ratings_wide <- llm_ratings_long %>%
  pivot_wider(names_from = rated, values_from = rating) %>%
  pivot_wider(names_from = rater, values_from = c(creativity, originality, surprise, value, author_ai))
