from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, Iterable, Self, Set

from graphviz import Digraph

from .config import validate_engine_or_get_default
from .engine import EngineLM
from .logger import logger


class Variable:
    def __init__(
        self,
        value: str = "",
        predecessors: set[Self] | None = None,
        requires_grad: bool = True,
        *,
        role_description: str,
    ) -> None:
        """The main thing. Nodes in the computation graph. Really the heart and soul of textgrad.

        :param value: The string value of this variable, defaults to "". In the future, we'll go multimodal, for sure!
        :type value: str or bytes, optional
        :param predecessors: predecessors of this variable in the computation graph, defaults to None. Here, for instance, if we have a prompt -> response through an LLM call, we'd call the prompt the predecessor, and the response the successor.
        :type predecessors: List[Variable], optional
        :param requires_grad: Whether this variable requires a gradient, defaults to True. If False, we'll not compute the gradients on this variable.
        :type requires_grad: bool, optional
        :param role_description: The role of this variable. We find that this has a huge impact on the optimization performance, and being specific often helps quite a bit!
        :type role_description: str
        """

        if predecessors is None:
            predecessors = set()

        _predecessor_requires_grad = [v for v in predecessors if v.requires_grad]

        if (not requires_grad) and len(_predecessor_requires_grad) > 0:
            raise Exception(
                "If the variable does not require grad, none of its predecessors should require grad."
                f"In this case, following predecessors require grad: {_predecessor_requires_grad}"
            )

        assert isinstance(value, str | bytes | int)
        if isinstance(value, int):
            value = str(value)

        self.value: str = value

        self.gradients: set[Variable] = set()
        self.gradients_context: Dict[Variable, str | dict] = {}
        self.grad_fn: None | Callable = None
        self.role_description = role_description
        self.predecessors = predecessors
        self.requires_grad: bool = requires_grad
        self.reduce_meta: list = []

        if requires_grad and isinstance(value, bytes):
            raise ValueError(
                "Gradients are not yet supported for image inputs. Please provide a string input instead."
            )

    def __repr__(self) -> str:
        return f"Variable(value={self.value}, role={self.get_role_description()}, grads={self.gradients})"

    def __str__(self) -> str:
        return str(self.value)

    def __add__(self, to_add: Any) -> Any:
        # For now, let's just assume variables can be passed to models
        if isinstance(to_add, Variable):
            # Somehow handle the addition of variables
            total = Variable(
                value=self.value + to_add.value,
                # Add the predecessors of both variables
                predecessors={self, to_add},
                # Communicate both of the roles
                role_description=f"{self.role_description} and {to_add.role_description}",
                # We should require grad if either of the variables require grad
                requires_grad=(self.requires_grad | to_add.requires_grad),
            )
            total.set_grad_fn(
                partial(
                    _backward_idempotent,
                    variables=total.predecessors,
                    summation=total,
                )
            )
            return total
        return to_add.__add__(self)

    def set_role_description(self, role_description) -> None:
        self.role_description = role_description

    def reset_gradients(self):
        self.gradients = set()
        self.gradients_context = {}
        self.reduce_meta = []

    def get_role_description(self) -> str:
        return self.role_description

    def get_short_value(self, n_words_offset: int = 10) -> str:
        """
        Returns a short version of the value of the variable. We sometimes use it during optimization, when we want to see the value of the variable, but don't want to see the entire value.
        This is sometimes to save tokens, sometimes to reduce repeating very long variables, such as code or solutions to hard problems.
        :param n_words_offset: The number of words to show from the beginning and the end of the value.
        :type n_words_offset: int
        """
        words = self.value.split(" ")
        if len(words) <= 2 * n_words_offset:
            return self.value
        short_value = (
            " ".join(words[:n_words_offset])
            + " (...) "
            + " ".join(words[-n_words_offset:])
        )
        return short_value

    def get_value(self) -> str:
        return self.value

    def set_value(self, value: str) -> None:
        self.value = value

    def set_grad_fn(self, grad_fn: Callable) -> None:
        self.grad_fn = grad_fn

    def get_grad_fn(self) -> Callable | None:
        return self.grad_fn

    def get_gradient_text(self) -> str:
        """Aggregates and returns the gradients on a variable."""

        return "\n".join([g.value for g in self.gradients])

    def backward(self, engine: EngineLM | None = None) -> None:
        """
        Backpropagate gradients through the computation graph starting from this variable.

        :param engine: The backward engine to use for gradient computation. If not provided, the global engine will be used.
        :type engine: EngineLM, optional

        :raises Exception: If no backward engine is provided and no global engine is set.
        :raises Exception: If both an engine is provided and the global engine is set.
        """

        backward_engine = validate_engine_or_get_default(engine)
        """Taken from https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py"""
        # topological order all the predecessors in the graph
        topo = []
        visited: set[Self] = set()

        def build_topo(v: Self) -> None:
            if v not in visited:
                visited.add(v)
                for predecessor in v.predecessors:
                    build_topo(predecessor)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        # TODO: we should somehow ensure that we do not have cases such as the predecessors of a variable requiring a gradient, but the variable itself not requiring a gradient

        self.gradients = set()
        for v in reversed(topo):
            if v.requires_grad:
                v.gradients = _check_and_reduce_gradients(v, backward_engine)
                if v.grad_fn is not None:
                    v.grad_fn(backward_engine=backward_engine)

    def generate_graph(self, print_gradients: bool = False) -> Digraph:
        """
        Generates a computation graph starting from the variable itself.

        :param print_gradients: A boolean indicating whether to print gradients in the graph.
        :return: A visualization of the computation graph.
        """

        def wrap_text(text: str, width: int = 40) -> str:
            """Wraps text at a given number of characters using HTML line breaks."""
            words = text.split()
            wrapped_text = ""
            line = ""
            for word in words:
                if len(line) + len(word) + 1 > width:
                    wrapped_text += line + "<br/>"
                    line = word
                else:
                    if line:
                        line += " "
                    line += word
            wrapped_text += line
            return wrapped_text

        def wrap_and_escape(text: str, width: int = 40) -> str:
            return wrap_text(text.replace("<", "&lt;").replace(">", "&gt;"), width)

        topo = []
        visited = set()

        def build_topo(v: Self) -> None:
            if v not in visited:
                visited.add(v)
                for predecessor in v.predecessors:
                    build_topo(predecessor)
                topo.append(v)

        def get_grad_fn_name(name):
            ws = name.split(" ")
            ws = [w for w in ws if "backward" in w]
            return " ".join(ws)

        build_topo(self)

        graph = Digraph(
            comment=f"Computation Graph starting from {self.role_description}"
        )
        graph.attr(rankdir="TB")  # Set the graph direction from top to bottom
        graph.attr(ranksep="0.2")  # Adjust the spacing between ranks
        graph.attr(bgcolor="lightgrey")  # Set the background color of the graph
        graph.attr(fontsize="7.5")  # Set the font size of the graph

        for v in reversed(topo):
            # Add each node to the graph
            label_color = "darkblue"

            node_label = (
                f"<b><font color='{label_color}'>Role: </font></b> {wrap_and_escape(v.role_description.capitalize())}"
                f"<br/><b><font color='{label_color}'>Value: </font></b> {wrap_and_escape(v.value)}"
            )

            if v.grad_fn is not None:
                node_label += f"<br/><b><font color='{label_color}'>Grad Fn: </font></b> {wrap_and_escape(get_grad_fn_name(str(v.grad_fn)))}"

            if v.reduce_meta != []:
                node_label += f"<br/><b><font color='{label_color}'>Reduce Meta: </font></b> {wrap_and_escape(str(v.reduce_meta))}"

            if print_gradients:
                node_label += f"<br/><b><font color='{label_color}'>Gradients: </font></b> {wrap_and_escape(v.get_gradient_text())}"
            # Update the graph node with modern font and better color scheme
            graph.node(
                str(id(v)),
                label=f"<{node_label}>",
                shape="rectangle",
                style="filled",
                fillcolor="lavender",
                fontsize="8",
                fontname="Arial",
                margin="0.1",
                pad="0.1",
                width="1.2",
            )
            # Add forward edges from predecessors to the parent
            for predecessor in v.predecessors:
                graph.edge(str(id(predecessor)), str(id(v)))

        return graph


def _check_and_reduce_gradients(
    variable: Variable, backward_engine: EngineLM
) -> Set[Variable]:
    """
    Check and reduce gradients for a given variable.

    This function checks if the gradients of the variable need to be reduced based on the reduction groups
    specified in the variable's metadata. If reduction is required, it performs the reduction operation
    for each reduction group and returns the reduced gradients.

    For example, we do things like averaging the gradients using this functionality.

    :param variable: The variable for which gradients need to be checked and reduced.
    :type variable: Variable
    :param backward_engine: The backward engine used for gradient computation.
    :type backward_engine: EngineLM

    :return: The reduced gradients for the variable.
    :rtype: Set[Variable]
    """
    if not variable.reduce_meta:
        return variable.gradients
    if variable.get_gradient_text() == "":
        return variable.gradients

    if len(variable.gradients) == 1:
        return variable.gradients

    id_to_gradient_set = defaultdict(set)
    id_to_op = (
        {}
    )  # Note: there must be a way to ensure that the op is the same for all the variables with the same id

    # Go through each gradient, group them by their reduction groups
    for gradient in variable.gradients:
        for reduce_item in gradient.reduce_meta:
            id_to_gradient_set[reduce_item["id"]].add(gradient)
            id_to_op[reduce_item["id"]] = reduce_item["op"]
    # For each reduction group, perform the reduction operation
    new_gradients = set()
    for group_id, gradients in id_to_gradient_set.items():
        logger.info(
            f"Reducing gradients for group {group_id}", extra={"gradients": gradients}
        )
        new_gradients.add(id_to_op[group_id](gradients, backward_engine))

    return new_gradients


def _backward_idempotent(
    variables: Iterable[Variable], summation: Variable, backward_engine: EngineLM
) -> None:
    """
    Perform an idempotent backward pass e.g. for textgrad.sum or Variable.__add__.
    In particular, this function backpropagates the gradients of the `summation` variable to all the variables in the `variables` list.

    :param variables: The list of variables to backpropagate the gradients to.
    :type variables: List[Variable]
    :param summation: The variable representing the summation operation.
    :type summation: Variable
    :param backward_engine: The backward engine used for backpropagation.
    :type backward_engine: EngineLM

    :return: None

    :notes:
        - The idempotent backward pass is used for textgrad.sum or Variable.__add__ operations.
        - The gradients of the `summation` variable are backpropagated to all the variables in the `variables` list.
        - The feedback from each variable is stored in their respective gradients.
        - The feedback from the `summation` variable is combined and stored in the `summation_gradients` variable.
        - The feedback from each variable is later used for feedback propagation to other variables in the computation graph.
        - The `reduce_meta` attribute of the `summation` variable is used to reduce the feedback if specified.
    """
    summation_gradients = summation.get_gradient_text()
    for variable in variables:
        if summation_gradients == "":
            variable_gradient_value = ""
        else:
            variable_gradient_value = f"Here is the combined feedback we got for this specific {variable.get_role_description()} and other variables: {summation_gradients}."

        logger.info(
            "Idempotent backward",
            extra={
                "v_gradient_value": variable_gradient_value,
                "summation_role": summation.get_role_description(),
            },
        )

        var_gradients = Variable(
            value=variable_gradient_value,
            role_description=f"feedback to {variable.get_role_description()}",
        )
        variable.gradients.add(var_gradients)

        if not summation.reduce_meta:
            var_gradients.reduce_meta.extend(summation.reduce_meta)
            variable.reduce_meta.extend(summation.reduce_meta)

        variable.gradients.add(
            Variable(
                value=variable_gradient_value,
                role_description=f"feedback to {variable.get_role_description()}",
            )
        )
