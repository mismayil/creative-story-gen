# Default prompt templates
DEFAULT_SYSTEM_INSTRUCTION_TEMPLATE = "You are an expert creative story writer. You will be given three words (e.g., car, wheel, drive) and then asked to write a creative short story that contains these three words. The idea is that instead of writing a standard story such as \"I went for a drive in my car with my hands on the steering wheel.\", you come up with a novel and unique story that uses the required words in unconventional ways or settings."
DEFAULT_USER_INSTRUCTION_TEMPLATE = "Write a creative short story using a maximum of five sentences. The story must include the following three words: {items}. However, the story should not be about {boring_theme}."

# Test prompt templates
TEST_SYSTEM_INSTRUCTION_TEMPLATE = ""
TEST_USER_INSTRUCTION_TEMPLATE = "You are an expert story writer. Write a creative short story using a maximum of five sentences. The story must include the following three words: {items}. However, the story should not be about {boring_theme}."