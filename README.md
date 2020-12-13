# LSTM_Stock_Prediction

Predicts stock based on its past performance, taken from a the past history csv from investing.com. It is required that you have two csv to get the prediction. The stock you want to predict and the composite index it is a part of.

## How it works

It will take the current year from each csv as the testing sample and the rest as the training set. First, the data will be normalized before it is fed to the LSTM in batches. After the model is trained in several epochs, it will then be tested to the test sets.

Afterwards, both model is stacked with each other and then trained again. This will be the final model.

This model will try to predict the next day result based on the data from x days ago where x can be determined by the user.

## Result

The initial LSTM, which is only from the features from investing.com on the particular stock, on average has an error of around Rp. 200 on this particular dataset. By combining the two, it reduces the error to Rp. 150. I believe that it is a really good start and by stacking more features we can get a more closer prediction.

However with that said, predicting stocks is an incredibly difficult task. There is a lot of agents that could fluctuate the market and the stocks itself, thus making it extremely volatile. Any attempt to predict the stocks, in my opinion, would fall flat and can merely be used to form a second opinion for experienced traders.
