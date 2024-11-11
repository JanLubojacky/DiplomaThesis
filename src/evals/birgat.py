import optuna

from src.models.birgat import BiRGAT
from src.base_classes.evaluator import ModelEvaluator
from src.base_classes.omic_data_loader import OmicDataManager


class BiRGATEvaluator(ModelEvaluator):
    def __init__(
        self,
        data_manager: OmicDataManager,
        n_trials: int = 30,
        verbose: bool = True,
        params: dict = {},
    ):
        """ """
        super().__init__(data_manager, n_trials, verbose)
        self.params = params
        self.model = None

        data, _, _, _ = data_manager.get_split(0)
        self.n_classes = data.y.unique().shape[0]
        self.in_channels = [data.x_dict[omics].shape[1] for omics in data.x_dict.keys()]
        self.omic_names = list(data.x_dict.keys())


    def create_model(self, trial: optuna.Trial):
        """Create and return model instance with trial parameters"""

        # here we can select hyperparameters from the trial later
        self.model = BiRGAT(
            
        )

    def train_model(self, train_x, train_y) -> None:
        """Train model implementation"""

    def test_model(self, test_x, test_y) -> dict:
        """Test model implementation"""

