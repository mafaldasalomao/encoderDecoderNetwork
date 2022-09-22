

imds = imageDatastore("C:\Program Files\MATLAB\R2022a\toolbox\vision\visiondata\vehicles", "IncludeSubfolders",true);%stopSignImages/
imds.ReadFcn = @customreader;
I = read(imds); % % read first image from imds

% initialize

% inputs to the network
inputSize = [224 224 3];
inputSizeBranch = [224 224 3];
X1 = dlarray(rand(inputSize),"SSCB");
X2 = dlarray(rand(inputSizeBranch),"SSCB");
%net = initialize(net,X1,X2);
unet.Learnables
unet = initialize(unet);
unet.Learnables

%% Train Network Using Custom Training Loop
dlnet  = unet;
learnables = dlnet.Learnables;
%To freeze the learnable parameters of the network, loop over the learnable
% parameters and set the learn rate to 0 using the setLearnRateFactor function.
actor = 0;

% numLearnables = size(learnables,1);
% for i = 1:numLearnables
%     layerName = learnables.Layer(i);
%     parameterName = learnables.Parameter(i);
%     
%     dlnet = setLearnRateFactor(dlnet,layerName,parameterName,factor);
% end

% To use the updated learn rate factors when training, you must pass the
% dlnetwork object to the update function in the custom training loop.
% For example, use the command
% [dlnet,velocity] = sgdmupdate(dlnet,gradients,velocity);


%options

numEpochs = 10;
miniBatchSize = 10;
initialLearnRate = 0.01;
decay = 0.01;
momentum = 0.9;

%create mbq
mbq = minibatchqueue(imds,...
    MiniBatchSize=miniBatchSize,...
    MiniBatchFcn=@preprocessMiniBatch,...
    MiniBatchFormat=["SSCB"]);
%Initialize the velocity parameter for the SGDM solver.
velocity = [];

%Calculate the total number of iterations for the training progress monitor.

numObservationsTrain = numel(imds.Files);
numIterationsPerEpoch = ceil(numObservationsTrain / miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;

%monitor = trainingProgressMonitor(Metrics="Loss",Info=["Epoch","LearnRate"],XLabel="Iteration");


% Train the network using a custom training loop. For each epoch, shuffle the data and loop over mini-batches of data. For each mini-batch:
% Evaluate the model loss, gradients, and state using the dlfeval and modelLoss functions and update the network state.
% Determine the learning rate for the time-based decay learning rate schedule.
% Update the network parameters using the sgdmupdate function.
% Update the loss, learn rate, and epoch values in the training progress monitor.
% Stop if the Stop property is true. The Stop property value of the TrainingProgressMonitor object changes to true when you click the Stop button.

net = dlnet;
epoch = 0;
iteration = 0;

% Loop over epochs.
while epoch < numEpochs
    
    epoch = epoch + 1;

    % Shuffle data.
    shuffle(mbq);
    
    % Loop over mini-batches.
    while hasdata(mbq)

        iteration = iteration + 1;
        
        % Read mini-batch of data.
        X = next(mbq);
        
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelLoss function and update the network state.
        [loss,gradients,state] = dlfeval(@modelLoss,net,X);
        net.State = state;
        
        % Determine learning rate for time-based decay learning rate schedule.
        learnRate = initialLearnRate/(1 + decay*iteration);
        
        % Update the network parameters using the SGDM optimizer.
        [net,velocity] = sgdmupdate(net,gradients,velocity,learnRate,momentum);
        
        % Update the training progress monitor.
       % recordMetrics(monitor,iteration,Loss=loss);
        %updateInfo(monitor,Epoch=epoch,LearnRate=learnRate);
        %monitor.Progress = 100 * iteration/numIterations;
        % Display the training progress.
        D = duration(0,0,toc(start),Format="hh:mm:ss");
        loss = double(loss);
        addpoints(lineLossTrain,iteration,loss)
        title("Epoch: " + epoch + ", Elapsed: " + string(D))
        drawnow
    end
end















