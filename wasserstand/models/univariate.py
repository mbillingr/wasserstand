from wasserstand.models.combinator_chain import ChainCombinator
from wasserstand.models.mean import MeanPredictor
from wasserstand.models.uvar import UnivariateAR


class UnivariatePredictor(ChainCombinator):
    def __init__(self, order, mean_learning_rate, ar_learning_rate):
        mean_model = MeanPredictor(learning_rate=mean_learning_rate)
        ar_model = UnivariateAR(order=order, learning_rate=ar_learning_rate)
        super().__init__(mean_model, ar_model)
