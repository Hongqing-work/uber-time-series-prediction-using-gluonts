from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.mx import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions

from gluonts.dataset.common import FileDataset

# TODO(Hongqing): Try to use the transform.
"""
from gluonts.dataset.field_names import FieldName
from gluonts.transform import ExpectedNumInstanceSampler, InstanceSplitter

transformation = InstanceSplitter(
    target_field=FieldName.TARGET,
    is_pad_field=FieldName.IS_PAD,
    start_field=FieldName.START,
    forecast_start_field=FieldName.FORECAST_START,
    instance_sampler=ExpectedNumInstanceSampler(
        num_instances=1,
        min_future=10,
    ),
    past_length=30,
    future_length=10,
)
"""

train_data = FileDataset("./data/train-dataset", freq="D")
test_data = FileDataset("./data/train-dataset", freq="D")

prediction_length = 10
estimator = SimpleFeedForwardEstimator(
    num_hidden_dimensions=[10],
    prediction_length=prediction_length,
    context_length=100,
    freq='D',
    trainer=Trainer(
        ctx="cpu",
        epochs=5,
        learning_rate=1e-3,
        num_batches_per_epoch=100
    )
)

predictor = estimator.train(train_data)

forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_data,
    predictor=predictor,
    num_samples=100
)

forecasts = list(forecast_it)
tss = list(ts_it)
