% Test the classification accuracy of the model by comparing the predictions on the validation set with the true labels.
% After training, making predictions on new data does not require the labels. Create minibatchqueue object containing only the predictors of the test data:
% To ignore the labels for testing, set the number of outputs of the mini-batch queue to 1.
% Specify the same mini-batch size used for training.
% Preprocess the predictors using the preprocessMiniBatchPredictors function, listed at the end of the example.
% For the single output of the datastore, specify the mini-batch format "SSCB" (spatial, spatial, channel, batch).

numOutputs = 1;

mbqTest = minibatchqueue(augimdsValidation,numOutputs, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFcn=@preprocessMiniBatchPredictors, ...
    MiniBatchFormat="SSCB");


%Loop over the mini-batches and classify the images using modelPredictions function, listed at the end of the example.
YTest = modelPredictions(net,mbqTest,classes);

%Evaluate the classification accuracy.
TTest = imdsValidation.Labels;
accuracy = mean(TTest == YTest)