import torch

from camylanet.training.nnUNetTrainer.UMambaTrainer import UMambaTrainer


class UMambaSmoke2Epochs(UMambaTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        plans_identifier: str = "nnUNetPlans",
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            unpack_dataset=unpack_dataset,
            plans_identifier=plans_identifier,
            device=device,
        )
        self.num_epochs = 2
