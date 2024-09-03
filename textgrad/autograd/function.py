from abc import ABC, abstractmethod
from typing import Any, Callable, Generator

from textgrad.engine import EngineLM, get_engine
from textgrad.variable import Variable

from ..config import SingletonBackwardEngine


class Function(ABC):
    """
    The class to define a function that can be called and backpropagated through.
    """

    def __call__(self, *args, **kwargs) -> Variable:
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> Variable:
        pass

    @abstractmethod
    def backward(self, *args, **kwargs) -> None:
        pass


class BackwardContext:
    """
    Represents a context for backward computation.

    :param backward_fn: The backward function to be called during backward computation.
    :param args: Variable length argument list to be passed to the backward function.
    :param kwargs: Arbitrary keyword arguments to be passed to the backward function.

    :ivar backward_fn: The backward function to be called during backward computation.
    :ivar fn_name: The fully qualified name of the backward function.
    :vartype fn_name: str
    :ivar args: Variable length argument list to be passed to the backward function.
    :ivar kwargs: Arbitrary keyword arguments to be passed to the backward function.

    :method __call__(backward_engine: EngineLM) -> Any:
        Calls the backward function with the given backward engine and returns the result.
    :method __repr__() -> str:
        Returns a string representation of the BackwardContext object.
    """

    def __init__(self, backward_fn: Callable, *args, **kwargs) -> None:
        self.backward_fn = backward_fn
        self.fn_name = f"{backward_fn.__module__}.{backward_fn.__qualname__}"
        self.args = args
        self.kwargs = kwargs

    def __call__(self, backward_engine: EngineLM):
        return self.backward_fn(
            *self.args, **self.kwargs, backward_engine=backward_engine
        )

    def __repr__(self) -> str:
        return f"{self.fn_name}"


class Module(ABC):
    """Abstract module class with parameters akin to PyTorch's nn.Module."""

    def __init__(
        self,
        engine: EngineLM | str | None = None,
    ) -> None:
        """
        :param engine: The language model to use for the comparison.
        :type engine: EngineLM
        """
        if engine is None:
            engine = SingletonBackwardEngine().get_engine()
        if isinstance(engine, str):
            engine = get_engine(engine)
        assert engine is not None
        self.engine: EngineLM = engine

    @abstractmethod
    def parameters(self) -> list[Variable]:
        pass

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.reset_gradients()

    def named_parameters(self) -> Generator:
        for p in self.parameters():
            yield p.get_role_description(), p

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Variable:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Variable:
        return self.forward(*args, **kwargs)
