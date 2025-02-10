# Default prompt templates
DEFAULT_SYSTEM_INSTRUCTION_TEMPLATE = "You are an expert creative story writer. You will be given three words (e.g., car, wheel, drive) and then asked to write a creative short story that contains these three words. The idea is that instead of writing a standard story such as \"I went for a drive in my car with my hands on the steering wheel.\", you come up with a novel and unique story that uses the required words in unconventional ways or settings."
DEFAULT_USER_INSTRUCTION_TEMPLATE = "Write a creative short story using a maximum of five sentences. The story must include the following three words: {items}. However, the story should not be about {boring_theme}."

# Test prompt templates
TEST_SYSTEM_INSTRUCTION_TEMPLATE = ""
TEST_USER_INSTRUCTION_TEMPLATE = "You are an expert story writer. Write a creative short story using a maximum of five sentences. The story must include the following three words: {items}. However, the story should not be about {boring_theme}."

# Controlled prompt templates
CONTROLLED_SYSTEM_INSTRUCTION_TEMPLATE = "You are an expert creative story writer. You will be given three words (e.g., car, wheel, drive) and then asked to write a creative short story that contains these three words. The idea is that instead of writing a standard story such as \"I went for a drive in my car with my hands on the steering wheel.\", you come up with a novel and unique story that uses the required words in unconventional ways or settings."
CONTROLLED_USER_INSTRUCTION_TEMPLATE = "Write a creative short story using exactly five sentences and each sentence should contain at least five words. The story must include the following three words: {items}. However, the story should not be about {boring_theme}."

# Simple prompt templates
SIMPLE_SYSTEM_INSTRUCTION_TEMPLATE = "You are an expert creative story writer. You will be given three words (e.g., car, wheel, drive) and then asked to write a creative short story that contains these three words. The idea is that instead of writing a standard story such as \"I went for a drive in my car with my hands on the steering wheel.\", you come up with a novel and unique story that uses the required words in unconventional ways or settings."
SIMPLE_USER_INSTRUCTION_TEMPLATE = "Write a creative short story using exactly five sentences and each sentence should contain at least five words. The story must include the following three words: {items}. However, the story should not be about {boring_theme}. Please, use simple words and sentence structures."

# Controlled prompt templates
PARAPHRASED_SYSTEM_INSTRUCTION_TEMPLATE = "As a skilled creative story writer, you will receive three words (e.g., car, wheel, drive) and be asked to craft a short story that incorporates them. The goal is to avoid typical scenarios like \"I went for a drive in my car with my hands on the steering wheel.\" Instead, you will create a fresh and imaginative story that uses the given words in unexpected and original contexts."
PARAPHRASED_USER_INSTRUCTION_TEMPLATE = "Create a short interesting story using exactly five sentences, with each sentence containing at least five words. The story must incorporate the following three words: {items}. However, the story should not focus on {boring_theme}."

# Summarization templates
SUMMARY_SYSTEM_INSTRUCTION_TEMPLATE = ""
SUMMARY_USER_INSTRUCTION_TEMPLATE = "Summarize the story into a key point using the following three words: {items}."
SUMMARY_SHOT_TEMPLATE = """Example {index}:
Story: {story}
Answer: {summary}
"""

# LLM-as-a-judge templates
LLM_JUDGE_SYSTEM_INSTRUCTION_TEMPLATE = """
You will be given three words, several stories written about these words and your task is to rate each story on a scale of 1 to 5 across following dimensions and predict whether each story was written by a human or an AI model:
-  Creativity: Rate the overall creativity of the story. When rating the story, try not to focus on the length of the story, or how good the language is. Instead, consider the overall creativity of the story. You could consider how creatively the three words were used; how original, clever, suprising, or interesting the story was.
- Originality: Rate how unique the story is or to what extent the given three words are used in unconventional ways.
- Surprise: Rate how unexpected the story is or to what extent the story contains unexpected twists or turns.
- Effectiveness: Rate to what extent the story is easy to follow and enjoyable.
For each dimension, use the 1 to 5 scale with discrete increments. Here is an example of the full rating scale for creativity dimension:
1: Very Uncreative
2: Uncreative
3: Undecided
4: Creative
5: Very Creative
When you rate these stories, here's a few things to keep in mind:
* The stories were written by either participants in a psychology experiment or by an AI model.
* Compare the stories to one another, not to another standard (what is a great story).
* Try to use the full 1-5 scale. For example, don't rate all stories either 1, 2, or 3.
* Ignore any misspellings or typos the stories might have and focus on the underlying content.
For each dimension, output the rating enclosed with the dimension tag, for example, <creativity>3</creativity>.
Finally, predict the author of the story by guessing whether it was written by human or AI and output your guess enclosed with <author> tag, for example, <author>human</author> or <author>AI</author>.
Enclose each story ratings in a tag with the story number, such as below:
<story1>
<creativity>3</creativity>
<originality>3</originality>
<surprise>3</surprise>
<effectiveness>3</effectiveness>
<author>1</author>
</story1>
Only output the ratings as described above and nothing else.
"""
LLM_JUDGE_USER_INSTRUCTION_TEMPLATE = """
Here is the three words: {words}

Here are the stories:
{stories}

Your ratings:
"""