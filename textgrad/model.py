from typing import Union

from textgrad.autograd import LLMCall
from textgrad.autograd.function import Module
from textgrad.engine import EngineLM
from textgrad.variable import Variable


class BlackboxLLM(Module):
    def __init__(
        self,
        engine: Union[EngineLM, str] | None = None,
        system_prompt: Union[Variable, str] | None = None,
    ) -> None:
        """
        Initialize the LLM module.

        :param engine: The language model engine to use.
        :type engine: EngineLM
        :param system_prompt: The system prompt variable, defaults to None.
        :type system_prompt: Variable, optional
        """
        super().__init__(engine=engine)
        if isinstance(system_prompt, str):
            system_prompt = Variable(
                system_prompt,
                requires_grad=False,
                role_description="system prompt for the language model",
            )
        self.system_prompt = system_prompt
        self.llm_call = LLMCall(self.engine, self.system_prompt)

    def parameters(self) -> list[Variable]:
        """
        Get the parameters of the blackbox LLM.

        :return: A list of parameters.
        :rtype: list
        """
        params = []
        if self.system_prompt:
            params.append(self.system_prompt)
        return params

    def forward(self, x: Variable) -> Variable:
        """
        Perform an LLM call.

        :param x: The input variable.
        :type x: Variable
        :return: The output variable.
        :rtype: Variable
        """
        return self.llm_call(x)
