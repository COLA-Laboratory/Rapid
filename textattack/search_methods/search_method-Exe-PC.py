"""
Search Method Abstract Class
===============================
"""

from abc import ABC, abstractmethod

from pyabsa import TADCheckpointManager

from textattack.shared.utils import ReprMixin

from textattack.goal_function_results import GoalFunctionResultStatus

class SearchMethod(ReprMixin, ABC):
    """This is an abstract class that contains main helper functionality for
    search methods.

    A search method is a strategy for applying transformations until the
    goal is met or the search is exhausted.
    """

    def __call__(self, initial_result, **kwargs):
        """Ensures access to necessary functions, then calls
        ``perform_search``"""
        if not hasattr(self, "get_transformations"):
            raise AttributeError(
                "Search Method must have access to get_transformations method"
            )
        if not hasattr(self, "get_goal_results"):
            raise AttributeError(
                "Search Method must have access to get_goal_results method"
            )
        if not hasattr(self, "filter_transformations"):
            raise AttributeError(
                "Search Method must have access to filter_transformations method"
            )
        reactive_defender = kwargs.get('reactive_defender')

        result = self.perform_search(initial_result)

        if reactive_defender:
            tad_res = reactive_defender.reactive_defense(result.attacked_text.text)
            if tad_res['label'] == str(initial_result.ground_truth_output):
                result.goal_status = GoalFunctionResultStatus.SEARCHING
                result.output = int(tad_res['label'])
        # ensure that the number of queries for this GoalFunctionResult is up-to-date
        result.num_queries = self.goal_function.num_queries
        return result

    @abstractmethod
    def perform_search(self, initial_result, **kwargs):
        """Perturbs `attacked_text` from ``initial_result`` until goal is
        reached or search is exhausted.

        Must be overridden by specific search methods.
        """
        raise NotImplementedError()

    def check_transformation_compatibility(self, transformation):
        """Determines whether this search method is compatible with
        ``transformation``."""
        return True

    @property
    def is_black_box(self):
        """Returns `True` if search method does not require access to victim
        model's internal states."""
        raise NotImplementedError()

    def get_victim_model(self):
        if self.is_black_box:
            raise NotImplementedError(
                "Cannot access victim model if search method is a black-box method."
            )
        else:
            return self.goal_function.model
