from typing import Iterable

from ..defaults import SYSTEM_PROMPT_DEFAULT_ROLE, VARIABLE_OUTPUT_DEFAULT_ROLE
from ..engine import EngineLM, validate_multimodal_engine
from ..logger import logger
from ..variable import Variable
from .function import BackwardContext, Function
from .llm_backward_prompts import (BACKWARD_SYSTEM_PROMPT,
                                   CONVERSATION_START_INSTRUCTION_BASE,
                                   CONVERSATION_START_INSTRUCTION_CHAIN,
                                   EVALUATE_VARIABLE_INSTRUCTION,
                                   OBJECTIVE_INSTRUCTION_BASE,
                                   OBJECTIVE_INSTRUCTION_CHAIN)
from .multimodal_backward_prompts import MULTIMODAL_CONVERSATION_TEMPLATE


class MultimodalLLMCall(Function):
    """The MultiModalLM call function. This function will call the LLM with the input (image) and return the response,
    also register the grad_fn for backpropagation.

    :param engine: engine to use for the LLM call
    :type engine: EngineLM
    :param system_prompt: system prompt to use for the LLM call, default depends on the engine.
    :type system_prompt: Variable, optional
    """

    def __init__(self, engine: EngineLM, system_prompt: Variable | None = None):
        super().__init__()
        self.engine = engine
        validate_multimodal_engine(self.engine)

        self.system_prompt = system_prompt
        if self.system_prompt and self.system_prompt.get_role_description() is None:
            self.system_prompt.set_role_description(SYSTEM_PROMPT_DEFAULT_ROLE)

    def forward(
        self,
        inputs: list[Variable],
        response_role_description: str = VARIABLE_OUTPUT_DEFAULT_ROLE,
    ) -> Variable:
        """
        Forward pass for the multimodal LLM call function.

        :param inputs: list of input variables to the multimodal LLM call. One is an image and the second one is text
        :type inputs: List[Variable]
        :param response_role_description: role description for the response variable
        :type response_role_description: str, optional

        >>> from textgrad import Variable, get_engine
        >>> from textgrad.autograd import MultimodalLLMCall
        >>> target_image = "A byte representation of the image"
        >>> question_variable = Variable("What do you see here?", role_description="question to answer", requires_grad=False)
        >>> response = MultimodalLLMCall("gpt-4o")([target_image, question_variable])
        """
        # First ensure that all keys are present in the fields

        # Assert that all variables are either strings or bytes
        for variable in inputs:
            if not isinstance(variable.get_value(), (str, bytes)):
                raise ValueError(
                    f"MultimodalLLMCall only accepts str or bytes, got {type(variable.get_value())}"
                )

        system_prompt_value = self.system_prompt.value if self.system_prompt else None
        input_content = [inp.value for inp in inputs]
        # Make the LLM Call
        response_text = self.engine(input_content, system_prompt=system_prompt_value)

        # Create the response variable
        response = Variable(
            value=response_text,
            predecessors=(
                {self.system_prompt, *inputs} if self.system_prompt else {*inputs}
            ),
            role_description=response_role_description,
        )

        logger.info(
            "MultimodalLLMCall function forward",
            extra={"text": f"System:{system_prompt_value}\n{inputs}"},
        )

        # Populate the gradient function, using a container to store the backward function and the context
        response.set_grad_fn(
            BackwardContext(
                backward_fn=self.backward,
                response=response,
                input_content=input_content,
                system_prompt=system_prompt_value,
            )
        )

        return response

    def backward(
        self,
        response: Variable,
        input_content: list[str],
        system_prompt: str,
        backward_engine: EngineLM,
    ) -> None:
        validate_multimodal_engine(backward_engine)

        children_variables = response.predecessors
        if response.get_gradient_text() == "":
            self._backward_through_multimodal_llm_base(
                children_variables,
                response,
                input_content,
                system_prompt,
                backward_engine,
            )
        else:
            self._backward_through_multimodal_llm_chain(
                children_variables,
                response,
                input_content,
                system_prompt,
                backward_engine,
            )

    @staticmethod
    def _construct_multimodal_llm_chain_backward_content(
        backward_info: dict[str, str],
    ) -> str:
        content = list(backward_info["input_content"])
        conversation = MULTIMODAL_CONVERSATION_TEMPLATE.format(**backward_info)
        backward_prompt = CONVERSATION_START_INSTRUCTION_CHAIN.format(
            conversation=conversation, **backward_info
        )
        backward_prompt += OBJECTIVE_INSTRUCTION_CHAIN.format(**backward_info)
        backward_prompt += EVALUATE_VARIABLE_INSTRUCTION.format(**backward_info)
        content.append(backward_prompt)
        return " ".join(content)

    @staticmethod
    def _backward_through_multimodal_llm_chain(
        variables: Iterable[Variable],
        response: Variable,
        input_content: list[str],
        system_prompt: str,
        backward_engine: EngineLM,
    ) -> None:
        for variable in variables:
            if not variable.requires_grad:
                continue

            backward_info: dict[str, str] = {
                "response_desc": response.get_role_description(),
                "response_value": response.get_value(),
                "response_gradient": response.get_gradient_text(),
                "input_content": input_content,
                "system_prompt": system_prompt,
                "variable_desc": variable.get_role_description(),
                "variable_short": variable.get_short_value(),
            }

            backward_content = (
                MultimodalLLMCall._construct_multimodal_llm_chain_backward_content(
                    backward_info
                )
            )

            logger.info(
                "_backward_through_llm prompt",
                extra={"_backward_through_llm": backward_content},
            )
            gradient_value = backward_engine(
                backward_content, system_prompt=BACKWARD_SYSTEM_PROMPT
            )
            logger.info(
                "_backward_through_llm gradient",
                extra={"_backward_through_llm": gradient_value},
            )

            var_gradients = Variable(
                value=gradient_value,
                role_description=f"feedback to {variable.get_role_description()}",
            )
            variable.gradients.add(var_gradients)
            conversation = MULTIMODAL_CONVERSATION_TEMPLATE.format(**backward_info)
            variable.gradients_context[var_gradients] = {
                "context": input_content + [conversation],
                "response_desc": response.get_role_description(),
                "variable_desc": variable.get_role_description(),
            }

            if response.reduce_meta:
                var_gradients.reduce_meta.extend(response.reduce_meta)
                variable.reduce_meta.extend(response.reduce_meta)

    @staticmethod
    def _construct_multimodal_llm_base_backward_content(
        backward_info: dict[str, str],
    ) -> str:
        content = list(backward_info["input_content"])
        conversation = MULTIMODAL_CONVERSATION_TEMPLATE.format(**backward_info)
        backward_prompt = CONVERSATION_START_INSTRUCTION_BASE.format(
            conversation=conversation, **backward_info
        )
        backward_prompt += OBJECTIVE_INSTRUCTION_BASE.format(**backward_info)
        backward_prompt += EVALUATE_VARIABLE_INSTRUCTION.format(**backward_info)
        content.append(backward_prompt)
        return " ".join(content)

    @staticmethod
    def _backward_through_multimodal_llm_base(
        variables: Iterable[Variable],
        response: Variable,
        input_content: list[str],
        system_prompt: str,
        backward_engine: EngineLM,
    ) -> None:
        for variable in variables:
            if not variable.requires_grad:
                continue

            backward_info = {
                "response_desc": response.get_role_description(),
                "response_value": response.get_value(),
                "input_content": input_content,
                "system_prompt": system_prompt,
                "variable_desc": variable.get_role_description(),
                "variable_short": variable.get_short_value(),
            }

            backward_content = (
                MultimodalLLMCall._construct_multimodal_llm_base_backward_content(
                    backward_info
                )
            )

            logger.info(
                "_backward_through_llm prompt",
                extra={"_backward_through_llm": backward_content},
            )
            gradient_value = backward_engine(
                backward_content, system_prompt=BACKWARD_SYSTEM_PROMPT
            )
            logger.info(
                "_backward_through_llm gradient",
                extra={"_backward_through_llm": gradient_value},
            )

            conversation = MULTIMODAL_CONVERSATION_TEMPLATE.format(**backward_info)
            var_gradients = Variable(
                value=gradient_value,
                role_description=f"feedback to {variable.get_role_description()}",
            )
            variable.gradients.add(var_gradients)
            variable.gradients_context[var_gradients] = {
                "context": input_content + [conversation],
                "response_desc": response.get_role_description(),
                "variable_desc": variable.get_role_description(),
            }

            if response.reduce_meta:
                var_gradients.reduce_meta.extend(response.reduce_meta)
                variable.reduce_meta.extend(response.reduce_meta)


class OrderedFieldsMultimodalLLMCall(MultimodalLLMCall):
    def __init__(
        self,
        engine: EngineLM,
        fields: list[str],
        system_prompt: Variable | None = None,
    ) -> None:
        super().__init__(engine=engine, system_prompt=system_prompt)

        self.fields = fields

    def forward(
        self,
        inputs: dict[str, Variable],
        response_role_description: str = VARIABLE_OUTPUT_DEFAULT_ROLE,
    ) -> Variable:
        # Assert that all variables are either strings or bytes
        for variable in inputs.values():
            if not isinstance(variable.get_value(), (str, bytes)):
                raise ValueError(
                    f"MultimodalLLMCall only accepts str or bytes, got {type(variable.get_value())}"
                )

        assert set(inputs.keys()) == set(self.fields)
        forward_content = []
        for field in self.fields:
            forward_content.append(f"{field}: {inputs[field].value}")

        system_prompt_value = self.system_prompt.value if self.system_prompt else None

        # Make the LLM Call
        response_text = self.engine(forward_content, system_prompt=system_prompt_value)

        # Create the response variable
        response = Variable(
            value=response_text,
            predecessors=(
                {self.system_prompt, *inputs.values()}
                if self.system_prompt
                else set(inputs.values())
            ),
            role_description=response_role_description,
        )

        logger.info(
            "MultimodalLLMCall function forward",
            extra={"text": f"System:{system_prompt_value}\n{forward_content}"},
        )

        # Populate the gradient function, using a container to store the backward function and the context
        response.set_grad_fn(
            BackwardContext(
                backward_fn=self.backward,
                response=response,
                input_content=forward_content,
                system_prompt=system_prompt_value,
            )
        )

        return response
