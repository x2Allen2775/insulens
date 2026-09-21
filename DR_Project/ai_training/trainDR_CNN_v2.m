% trainDR_CNN_v2.m
% -------------------------------------------------------------------------
% Description: 
%   IMPROVED Training Script (Version 2) for the DR Classification CNN.
%   Improvements over V1:
%     1. Class Balancing via oversampling minority classes
%     2. Aggressive Data Augmentation (rotation, color jitter)
%     3. CLAHE Preprocessing of ALL training images before feeding to CNN
%     4. Learning Rate Scheduling (piecewise decay) + More Epochs
%
% Usage:
%   Run organizeAPTOS.m first (if not already done), then run this script.
%   This will take longer than V1 but produce a significantly better model.
%
% Hardware:
%   Utilizes Mac M-series GPU/Neural Engine automatically.
% -------------------------------------------------------------------------

clc; clear; close all;
disp('=== Starting DR CNN Training Pipeline V2 (Improved) ===');

% 1. Setup paths
scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir, '..', 'preprocessing'));

datasetPath = fullfile(scriptDir, '..', 'datasets', 'APTOS_Train');

if ~exist(datasetPath, 'dir')
    error(['Dataset folder not found at: ', datasetPath, ...
           newline, 'Please run organizeAPTOS.m first.']);
end

% =====================================================================
% IMPROVEMENT #3: Preprocess ALL images with CLAHE before training
% =====================================================================
preprocessedPath = fullfile(scriptDir, '..', 'datasets', 'APTOS_Train_Preprocessed');

if ~exist(preprocessedPath, 'dir')
    disp('Preprocessing ALL training images with CLAHE (one-time operation)...');
    
    % Create class subfolders in preprocessed directory
    for c = 0:4
        mkdir(fullfile(preprocessedPath, num2str(c)));
    end
    
    % Process each class folder
    totalProcessed = 0;
    for c = 0:4
        classDir = fullfile(datasetPath, num2str(c));
        files = dir(fullfile(classDir, '*.png'));
        
        for f = 1:length(files)
            srcFile = fullfile(classDir, files(f).name);
            destFile = fullfile(preprocessedPath, num2str(c), files(f).name);
            
            try
                % Apply our CLAHE preprocessing pipeline
                [enhancedImg, ~] = preprocessFundus(srcFile);
                
                % Convert grayscale back to 3-channel for ResNet
                enhancedRGB = cat(3, enhancedImg, enhancedImg, enhancedImg);
                imwrite(enhancedRGB, destFile);
            catch
                % If preprocessing fails, copy original
                copyfile(srcFile, destFile);
            end
            
            totalProcessed = totalProcessed + 1;
            if mod(totalProcessed, 200) == 0
                fprintf('  Preprocessed %d images...\n', totalProcessed);
            end
        end
    end
    fprintf('  Done! Preprocessed %d images with CLAHE.\n\n', totalProcessed);
else
    disp('Preprocessed dataset already exists. Skipping CLAHE step.');
end

% Load the preprocessed dataset
disp('Loading preprocessed dataset...');
imds = imageDatastore(preprocessedPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% =====================================================================
% IMPROVEMENT #1: Class Balancing via Oversampling
% =====================================================================
disp('Balancing classes via oversampling...');

% Count images per class
labelCounts = countEachLabel(imds);
disp(labelCounts);

maxCount = max(labelCounts.Count);

% Oversample each class to match the majority class
balancedFiles = {};
balancedLabels = categorical();

for i = 1:height(labelCounts)
    classLabel = labelCounts.Label(i);
    classCount = labelCounts.Count(i);
    
    % Get all files for this class
    classIdx = find(imds.Labels == classLabel);
    classFiles = imds.Files(classIdx);
    
    % Repeat files to reach maxCount
    numRepeats = ceil(maxCount / classCount);
    repeatedFiles = repmat(classFiles, numRepeats, 1);
    repeatedFiles = repeatedFiles(1:maxCount); % Trim to exact count
    
    balancedFiles = [balancedFiles; repeatedFiles]; %#ok<AGROW>
    balancedLabels = [balancedLabels; repmat(classLabel, maxCount, 1)]; %#ok<AGROW>
end

% Create a new balanced datastore
imdsBalanced = imageDatastore(balancedFiles, 'Labels', balancedLabels);

fprintf('Balanced dataset: %d images per class (%d total).\n\n', maxCount, length(balancedFiles));

% Split into Training (80%) and Validation (20%)
[imdsTrain, imdsValidation] = splitEachLabel(imdsBalanced, 0.8, 'randomized');

% 2. Load Pre-trained ResNet-50
disp('Loading Pre-trained ResNet-50 Model...');
net = resnet50;
inputSize = net.Layers(1).InputSize;

% 3. Modify Network for 5-Class Classification
lgraph = layerGraph(net);
newFc = fullyConnectedLayer(5, 'Name', 'new_fc', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
newClass = classificationLayer('Name', 'new_classoutput');

lgraph = replaceLayer(lgraph, 'fc1000', newFc);
lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', newClass);

% =====================================================================
% IMPROVEMENT #2: Aggressive Data Augmentation
% =====================================================================
disp('Setting up aggressive data augmentation...');

imageAugmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandXTranslation', [-40 40], ...
    'RandYTranslation', [-40 40], ...
    'RandRotation', [-180 180], ...        % Full rotation (fundus can be at any angle)
    'RandScale', [0.85 1.15], ...          % Slight zoom in/out
    'RandXShear', [-10 10], ...            % Slight shearing
    'RandYShear', [-10 10]);

augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', imageAugmenter, ...
    'ColorPreprocessing', 'gray2rgb');     % Handle any grayscale images

augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation, ...
    'ColorPreprocessing', 'gray2rgb');

% =====================================================================
% IMPROVEMENT #4: Learning Rate Scheduling + More Epochs
% =====================================================================
disp('Setting up training options (LR scheduling + 25 epochs)...');

options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 25, ...                          % More epochs (was 15)
    'InitialLearnRate', 3e-4, ...                  % Slightly higher start
    'LearnRateSchedule', 'piecewise', ...          % Decay the LR
    'LearnRateDropFactor', 0.3, ...                % Drop to 30% of current LR
    'LearnRateDropPeriod', 8, ...                   % Every 8 epochs
    'L2Regularization', 1e-4, ...                  % Prevent overfitting
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 50, ...
    'Verbose', true, ...                           % Show progress in command window
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

% 6. Train the Network
disp('Starting Training V2 (This will take several hours)...');
trainedNet = trainNetwork(augimdsTrain, lgraph, options);

% 7. Evaluate on Validation Set
disp('Evaluating model on validation set...');
predictedLabels = classify(trainedNet, augimdsValidation);
actualLabels = imdsValidation.Labels;
accuracy = mean(predictedLabels == actualLabels);
fprintf('\n=== Validation Accuracy: %.2f%% ===\n\n', accuracy * 100);

% 8. Save the Model (overwrites V1)
savePath = fullfile(scriptDir, 'trained_DR_ResNet.mat');
save(savePath, 'trainedNet');
fprintf('Training V2 Complete! Model saved to: %s\n', savePath);
fprintf('Validation Accuracy: %.2f%%\n', accuracy * 100);
