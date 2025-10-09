from .route import MeanRouting, MLPRouting, LSTMRouting


ROUTING_MODELS = {
    "mean": MeanRouting,
    "mlp": MLPRouting,
    "lstm": LSTMRouting,
}


__all__ = ["ROUTING_MODELS", ""]