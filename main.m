%clear all;
%in this case the block to be increase the filter under 2^5+indexblock
%the first filter size is 64, second 128 etc
% convolution2dLayer(filterSize,numFilters)
encoderBlock = @(block) [
    convolution2dLayer(3,2^(5+block),"Padding",'same')
    reluLayer
    convolution2dLayer(3,2^(5+block),"Padding",'same')
    reluLayer
    maxPooling2dLayer(2,"Stride",2)];
encoder = blockedNetwork(encoderBlock,4,"NamePrefix","encoder_");

%Create the decoder module consisting of four decoder blocks.

decoderBlock = @(block) [
    transposedConv2dLayer(2,2^(10-block),'Stride',2)
    convolution2dLayer(3,2^(10-block),"Padding",'same')
    reluLayer
    convolution2dLayer(3,2^(10-block),"Padding",'same')
    reluLayer];
decoder = blockedNetwork(decoderBlock,4,"NamePrefix","decoder_");



%Create the bridge layers.
bridge = [
    convolution2dLayer(3,1024,"Padding",'same')
    reluLayer
    convolution2dLayer(3,1024,"Padding",'same')
    reluLayer
    dropoutLayer(0.5)];

%Specify the network input size.

inputSize = [224 224 3];

%Create the U-Net network by connecting the encoder module, bridge, and decoder module and adding skip connections.

unet = encoderDecoderNetwork(inputSize,encoder,decoder, ...
    "OutputChannels",3, ...
    "SkipConnections","concatenate", ...
    "LatentNetwork",bridge)   %LatentNetwork — Network connecting encoder and decoder specified as a layer or array of layers.
%If you specify the 'OutputChannels' argument, then the final network is connected after the final 1-by-1 convolution layer of the decoder
%Number of output channels of the decoder network, specified as a positive integer. If you specify this argument, then the final layer of the decoder performs a 1-by-1 convolution operation with the specified number of channels.
%SkipConnectionNames — Names of pairs of encoder/decoder layers
%

%analyzeNetwork(unet)