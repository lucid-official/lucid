"""Names of the task heads a model carries, by what they predict (private).

A head is the module that maps the trunk's hidden states to a task's output.
Two things need to know it by name: the wrapper of the task, which trains it
beside the adapter (``modules_to_save``), and a prompt in front of the input,
which has to cut its own rows off the head's input again.
"""

from typing import Final

#: A classification head: ``classifier`` in most families, ``score`` in the causal ones.
CLASSIFIER_HEADS: Final = ("classifier", "score")
#: A span head: one start and one end logit per position.
SPAN_HEADS: Final = ("qa_outputs",)
