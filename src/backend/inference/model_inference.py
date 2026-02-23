"""
Inference
=========
Runs single or batch inference with a trained surrogate model.
"""


class Inference:
    """Run predictions (single or batch) using a trained surrogate model."""

    def __init__(self, batch_size: int = 1):
        self.batch_size = batch_size

    def execute(self, inputs: dict) -> dict:
        """
        Run inference.

        Parameters
        ----------
        inputs : dict
            Must contain 'model' (a trained predictor) and 'X' (input data).

        Returns
        -------
        dict with 'predictions'.
        """
        model = inputs.get("model")
        X = inputs.get("X")
        if model is None or X is None:
            raise ValueError("Inference requires 'model' and 'X' in inputs.")

        # TODO: implement batched prediction loop using self.batch_size
        return {**inputs, "predictions": None}
