from agents.base_agent import BaseAgent
from agents.delegator_agent import DelegatorAgent

# Prompt Generators
from agents.simple_task_generator import SimpleTaskPromptGenerator
from agents.complex_task_generator import ComplexTaskPromptGenerator
from agents.unclear_task_generator import UnclearTaskPromptGenerator
from agents.object_based_generator import ObjectBasedPromptGenerator
from agents.opinion_based_generator import OpinionBasedPromptGenerator
from agents.generic_generator import GenericPromptGenerator

# Reviewers
from agents.simple_task_reviewer import SimpleTaskReviewer
from agents.complex_task_reviewer import ComplexTaskReviewer
from agents.unclear_task_reviewer import UnclearTaskReviewer
from agents.object_based_reviewer import ObjectBasedReviewer
from agents.opinion_based_reviewer import OpinionBasedReviewer
from agents.generic_reviewer import GenericReviewer

# Editors
from agents.simple_task_editor import SimpleTaskEditor
from agents.complex_task_editor import ComplexTaskEditor
from agents.unclear_task_editor import UnclearTaskEditor
from agents.object_based_editor import ObjectBasedEditor
from agents.opinion_based_editor import OpinionBasedEditor
from agents.generic_editor import GenericEditor

__all__ = [
    'BaseAgent',
    'DelegatorAgent',
    # Generators
    'SimpleTaskPromptGenerator',
    'ComplexTaskPromptGenerator',
    'UnclearTaskPromptGenerator',
    'ObjectBasedPromptGenerator',
    'OpinionBasedPromptGenerator',
    'GenericPromptGenerator',
    # Reviewers
    'SimpleTaskReviewer',
    'ComplexTaskReviewer',
    'UnclearTaskReviewer',
    'ObjectBasedReviewer',
    'OpinionBasedReviewer',
    'GenericReviewer',
    # Editors
    'SimpleTaskEditor',
    'ComplexTaskEditor',
    'UnclearTaskEditor',
    'ObjectBasedEditor',
    'OpinionBasedEditor',
    'GenericEditor'
]