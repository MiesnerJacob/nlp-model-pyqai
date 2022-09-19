from transformers import pipeline


class MaskedLanguageMasking:
    """
    Masked Language masking via Transformers Models.
    Attributes:
        unmasker: An instance of Huggingface Pipline for the fill-mask task.
    """

    def __init__(self):
        self.unmasker = pipeline('fill-mask', model='roberta-base')

    def generate_masked_tokens(self, text):
        """
        Predict masked tokens within a text sequence.
        Parameters:
            text (str): The user input string with masked tokens
        Returns:
            predictions (str): The predicted probabilities for masked tokens
        """

        return self.unmasker(text)
