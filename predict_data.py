from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from matplotlib import pyplot
import pandas

import argparse
import enum

import xgboost_prediction
import regression_prediction
import multivariate_prediction

# Helper for parsing enums
class EnumAction(argparse.Action):
    """
    Argparse action for handling Enums
    """
    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        kwargs.setdefault("choices", tuple(e.value for e in enum_type))

        super(EnumAction, self).__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        value = self._enum(values)
        setattr(namespace, self.dest, value)
# ======================================================

class PredictionMethod(enum.Enum):
    RegressionImputation = 'regression-imputation'
    MultivariateRegressionImputation = 'multivariate-regression-imputation'
    XGBoost = 'xgboost'

def main():
    parser = argparse.ArgumentParser(description='Predict missing data')
    parser.add_argument('file', type=str,
                    help='file with complete data')
    parser.add_argument('method', type=PredictionMethod, default='xgboost',
                    help='method for predicting missing data')

    args = parser.parse_args()

    series = read_csv(args.file, header=0, index_col=0)
    flattened_data = pandas.DataFrame(series.to_numpy().flatten())

    mae = 0
    y = []
    yhat = []

    if args.method == PredictionMethod.XGBoost:
        mae, y, yhat = xgboost_prediction.predict_data(flattened_data)
    elif args.method == PredictionMethod.RegressionImputation:
        mae, y, yhat = regression_prediction.predict_data(flattened_data)
    elif args.method == PredictionMethod.MultivariateRegressionImputation:
        mae, y, yhat = multivariate_prediction.predict_data(series)
  

    print('MAE: %.3f' % mae)
    # plot expected vs preducted
    pyplot.plot(y, label='Actual')
    pyplot.legend()
    pyplot.show()

    pyplot.plot(yhat, label='Predicted')
    pyplot.legend()
    pyplot.show()


if __name__ == "__main__":
    main()
