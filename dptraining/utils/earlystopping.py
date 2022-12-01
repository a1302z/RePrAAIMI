"""
Adapted ignite early stopping. So far only for increasing scores (i.e. no losses)
"""
from dptraining.utils.misc import StateDictObjectMetricTracker

# It would actually be nice to incorporate this pylint warning
# and merge early stopping with reduce on plateau scheduler


class EarlyStopping(StateDictObjectMetricTracker):
    """EarlyStopping handler can be used to stop the training if
    no improvement after a given number of events.
    Args:
        patience: Number of events to wait if no improvement and then stop the training.
        min_delta: A minimum increase in the score to qualify as an improvement,
            i.e. an increase of less than or equal to `min_delta`, will count
            as no improvement.
        cumulative_delta: It True, `min_delta` defines an increase since the last
            `patience` reset, otherwise,
            it defines an increase after the last event. Default value is False.
    """

    def __call__(self, score: float) -> bool:
        return self.update(score)
