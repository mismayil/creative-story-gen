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

## expert data
# clean expert data
expert_dat <- expert_dat %>%
  select(user_id, story_id, creativity, originality, surprise, value, author) %>%
  # remove test data
  filter(user_id != 106) %>%
  # remove NULL values, i.e. errors from initial data storage problems
  filter(!(creativity=="NULL")) %>%
  # create new variables
  mutate(rater = if_else(user_id == 103, "expert_1", "expert_2"),
         author_ai = ifelse(author == "AI", 1, 0)) %>%
  # make variables right type
  mutate(creativity = as.numeric(creativity),
         originality = as.numeric(originality),
         surprise = as.numeric(surprise),
         value = as.numeric(value),
         author_ai = as.numeric(author_ai),
         rater = as.factor(rater))

## prep expert data
# expert data in long format
expert_long <- expert_dat %>% 
  select(-c(user_id, author)) %>%
  pivot_longer(cols=c(creativity, originality, surprise, value, author_ai),
               names_to='rated',
               values_to='rating')

# expert data in wide format
expert_wide <- expert_long %>%
  pivot_wider(names_from = rated, values_from = rating) %>%
  pivot_wider(names_from = rater, values_from = c(creativity, originality, surprise, value, author_ai))

## non-expert data
# clean non-expert data
nonexpert_dat <- nonexpert_dat %>%
  select(user_id, story_id, creativity, originality, surprise, value, pred_author) %>%
  # create new variables
  mutate(rater = paste0("nonexpert_", user_id),
         author_ai = ifelse(pred_author == "AI", 1, 0)) %>%
  # make variables right type
  mutate(creativity = as.numeric(creativity),
         originality = as.numeric(originality),
         surprise = as.numeric(surprise),
         value = as.numeric(value),
         author_ai = as.numeric(author_ai),
         rater = as.factor(rater))

## prep non-expert data
# expert data in long format
nonexpert_long <- nonexpert_dat %>% 
  select(-c(user_id, pred_author)) %>%
  pivot_longer(cols=c(creativity, originality, surprise, value, author_ai),
               names_to='rated',
               values_to='rating')

# expert data in wide format
nonexpert_wide <- nonexpert_long %>%
  pivot_wider(names_from = rated, values_from = rating) %>%
  pivot_wider(names_from = rater, values_from = c(creativity, originality, surprise, value, author_ai))