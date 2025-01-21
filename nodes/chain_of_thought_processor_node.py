import re

class TextExtractorNode:
    """
    Removes any <think>...</think> sections from a string, 
    outputting both the cleaned text and the extracted CoT content.
    """

    # Informal metadata; some Comfy forks or future versions 
    # may use these for search / tooltips
    tags = ["LLM", "ollama", "extraction", "text", "think", "cot"]
    description = "Removes <think>...</think> content from LLM output."

    @classmethod
    def INPUT_TYPES(cls):
        # forceInput=True ensures 'text_in' is always an incoming link 
        # instead of a user-editable widget.
        return {
            "required": {
                "text_in": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")  # (cleaned_text, extracted_thoughts)
    RETURN_NAMES = ("Answer Text", "CoT Content")
    FUNCTION = "process_cot"
    CATEGORY = "Text"

    def process_cot(self, text_in):
        """
        Splits out <think>...</think> content into its own output,
        and returns the input string with those sections removed.
        """
        # Regex to capture everything between <think>...</think>, DOTALL for multiline
        pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)

        # Extract CoT content
        cot_matches = pattern.findall(text_in)

        # Remove them from the main text
        answer_text = pattern.sub("", text_in).strip()

        # If multiple <think> blocks exist, join them with newlines
        cot_content = "\n".join(cot_matches)

        return (answer_text, cot_content)
