from typing import Iterable

REDUCE_MEAN_SYSTEM_PROMPT = (
    "You are part of an optimization system that improves a given text (i.e. the variable). "
    "Your only responsibility is to critically aggregate and summarize the feedback from sources. "
    "The variables may be solutions to problems, prompts to language models, code, or any other text-based variable. "
    "The multiple sources of feedback will be given to you in <FEEDBACK> </FEEDBACK> tags. "
    "When giving a response, only provide the core summary of the feedback. Do not recommend a new version for the variable -- only summarize the feedback critically. "
)


def construct_reduce_prompt(gradients: Iterable) -> str:
    """
    Construct a prompt that reduces the gradients.
    """
    gradient_texts: list[str] = []
    for gradient in gradients:
        gradient_texts.append(f"<FEEDBACK>{gradient.get_value()}</FEEDBACK>")
    return "\n".join(gradient_texts)
