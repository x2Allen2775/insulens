% trainDR_CNN.m
% -------------------------------------------------------------------------
% Description: 
%   Trains a Convolutional Neural Network (ResNet-50 via Transfer Learning)
%   to classify Diabetic Retinopathy severity (Levels 0-4).
%
% Usage:
%   1. Download the Kaggle APTOS 2019 Blindness Detection dataset.
%   2. Extract the images into a folder.
%   3. Ensure images are sorted into subfolders by class (0, 1, 2, 3, 4) OR
%      modify the datastore to read from a CSV label file.
%   4. Set the `datasetPath` below and run this script.
%
% Hardware:
%   Will automatically utilize Mac M-series GPU/Neural Engine if available.
% -------------------------------------------------------------------------

disp('--- Starting DR CNN Training Pipeline ---');

% 1. Setup the Dataset Path
% CHANGE THIS PATH to where you downloaded the APTOS dataset!
datasetPath = fullfile(fileparts(mfilename('fullpath')), '..', 'datasets', 'APTOS_Train'); 

if ~exist(datasetPath, 'dir')
    error(['Dataset folder not found at: ', datasetPath, ...
           newline, 'Please download the APTOS dataset and update the path.']);
end

% Create Image Datastore (assuming subfolders 0, 1, 2, 3, 4)
disp('Loading dataset...');
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Split data into Training and Validation (80% / 20%)
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

% 2. Load Pre-trained ResNet-50 Network
disp('Loading Pre-trained ResNet-50 Model...');
net = resnet50;

% Optional: analyzeNetwork(net) % To view the architecture

% 3. Modify Network for 5-Class Classification (DR Levels 0-4)
inputSize = net.Layers(1).InputSize;

% ResNet-50's final fully connected layer is 'fc1000' and classification layer is 'ClassificationLayer_fc1000'
lgraph = layerGraph(net);
newFc = fullyConnectedLayer(5, 'Name', 'new_fc', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
newClass = classificationLayer('Name', 'new_classoutput');

lgraph = replaceLayer(lgraph, 'fc1000', newFc);
lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', newClass);

% 4. Data Augmentation & Resizing
% ResNet-50 requires 224x224x3 input
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandXTranslation', pixelRange, ...
    'RandYTranslation', pixelRange, ...
    'RandScale', scaleRange);

augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);

% 5. Define Training Options
disp('Setting up training options...');
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 15, ... % Adjust based on time/accuracy needs
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 50, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto'); % 'auto' uses Mac GPU (MPS) if available

% 6. Train the Network
disp('Starting Training (This may take several hours)...');
trainedNet = trainNetwork(augimdsTrain, lgraph, options);

% 7. Save the Model
savePath = fullfile(fileparts(mfilename('fullpath')), 'trained_DR_ResNet.mat');
save(savePath, 'trainedNet');
disp(['Training Complete! Model saved successfully to: ', savePath]);
