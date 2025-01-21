import re

class ChainOfThoughtProcessorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_in": ("STRING", {"multiline": True, "default": ""})
            }
        }

    RETURN_TYPES = ("STRING", "STRING")  # Two outputs: answer_text, cot_content
    RETURN_NAMES = ("Answer Text", "CoT Content")
    FUNCTION = "process_cot"
    CATEGORY = "Text"

    def process_cot(self, text_in):
        """
        Splits out <think>...</think> content into its own output,
        and returns the input string with those sections removed.
        """
        # Regex pattern to capture everything between <think>...</think>
        pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)

        # Extract all <think>...</think> contents
        cot_matches = pattern.findall(text_in)

        # Remove them from the main text
        answer_text = pattern.sub("", text_in).strip()

        # Combine multiple <think> blocks (if any) with newlines
        cot_content = "\n".join(cot_matches)

        return (answer_text, cot_content)
