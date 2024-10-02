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
CONTROLLED_ALT1_SYSTEM_INSTRUCTION_TEMPLATE = "As a skilled creative story writer, you will receive three words (e.g., car, wheel, drive) and be asked to craft a short story that incorporates them. The goal is to avoid typical scenarios like \"I went for a drive in my car with my hands on the steering wheel.\" Instead, you will create a fresh and imaginative story that uses the given words in unexpected and original contexts."
CONTROLLED_ALT1_USER_INSTRUCTION_TEMPLATE = "Create a short interesting story using exactly five sentences, with each sentence containing at least five words. The story must incorporate the following three words: {items}. However, the story should not focus on {boring_theme}."