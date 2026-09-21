% runHybridPipeline.m
% -------------------------------------------------------------------------
% Description: 
%   The Master Script for the Insulens Hybrid AI Architecture.
%   Integrates the Deep Learning CNN with the Classical Rule-Based Fallback.
%   Demonstrates the "Fail-Safe Switch" and "Discordance Flagging".
%
% USAGE:
%   Option 1: Run this script as-is. A file picker dialog will open
%             so you can choose ANY fundus image from your computer.
%   Option 2: Set the variable below to a specific image path to skip 
%             the dialog.
% -------------------------------------------------------------------------

clc; clear; close all;

% 1. Setup paths
scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir, 'quality_assessment'));
addpath(fullfile(scriptDir, 'preprocessing'));
addpath(fullfile(scriptDir, 'segmentation'));
addpath(fullfile(scriptDir, 'lesion_detection'));
addpath(fullfile(scriptDir, 'classification'));
addpath(fullfile(scriptDir, 'explainability'));
addpath(fullfile(scriptDir, 'ai_modules'));

% =====================================================================
% 2. SELECT YOUR IMAGE HERE
% =====================================================================
% Option A: Set a specific path (uncomment one of these):
% testImagePath = fullfile(scriptDir, 'raw_images', 'A. Segmentation', '1. Original Images', 'a. Training Set', 'IDRiD_12.jpg');
% testImagePath = '/path/to/your/own/image.jpg';

% Option B: Open a file picker dialog (DEFAULT):
[fileName, filePath] = uigetfile({'*.jpg;*.jpeg;*.png;*.tiff;*.tif;*.bmp', 'Image Files'}, ...
    'Select a Fundus Image for Screening', scriptDir);
if isequal(fileName, 0)
    disp('No image selected. Pipeline cancelled.');
    return;
end
testImagePath = fullfile(filePath, fileName);
% =====================================================================

fprintf('==================================================\n');
fprintf('  INSULENS HYBRID AI PIPELINE INITIALIZED\n');
fprintf('==================================================\n');
fprintf('Processing Image: %s\n\n', testImagePath);

% Read the original image for display
originalImage = imread(testImagePath);

% ---------------------------------------------------------
% PILLAR 1: Quality Assessment
% ---------------------------------------------------------
fprintf('[1/6] Running Quality Assessment...\n');
[status, feedback, metrics] = assessQuality(testImagePath);

if status == "UNGRADABLE"
    fprintf('  -> ERROR: Image is UNGRADABLE. Pipeline stopped. Please recapture.\n');
    figure('Name', 'UNGRADABLE IMAGE', 'Position', [100, 100, 600, 400]);
    imshow(originalImage);
    title('IMAGE REJECTED: Ungradable Quality', 'FontSize', 16, 'Color', 'red');
    return;
end
fprintf('  -> PASS: %s\n\n', feedback);

% ---------------------------------------------------------
% PILLAR 2: Preprocessing
% ---------------------------------------------------------
fprintf('[2/6] Running Preprocessing & Enhancement...\n');
[enhancedImage, origGreen] = preprocessFundus(testImagePath);
fprintf('  -> Image normalized.\n\n');

% ---------------------------------------------------------
% PILLAR 3: Segmentation (Always run for visuals)
% ---------------------------------------------------------
fprintf('[3/6] Running Retinal Structure Segmentation...\n');
[discCenter, discMask] = locateOpticDisc(enhancedImage);
vesselMask = segmentVessels(enhancedImage);
fprintf('  -> Optic Disc located. Vessels segmented.\n\n');

% ---------------------------------------------------------
% PILLAR 4: Lesion Detection (Always run for visuals)
% ---------------------------------------------------------
fprintf('[4/6] Running Lesion Detection...\n');
exudateMask = detectExudates(enhancedImage, discMask);
maMask = detectMicroaneurysms(enhancedImage, vesselMask);
hemMask = detectHemorrhages(enhancedImage, vesselMask);

features.numMA = numel(regionprops(maMask, 'Area'));
features.numHem = numel(regionprops(hemMask, 'Area'));
features.numExudates = numel(regionprops(exudateMask, 'Area'));

fprintf('  -> Detected %d Microaneurysms.\n', features.numMA);
fprintf('  -> Detected %d Hemorrhages.\n', features.numHem);
fprintf('  -> Detected %d Exudate regions.\n\n', features.numExudates);

% Classical Classification
[classicLevel, classicName] = classifyDR(features);

% ---------------------------------------------------------
% PILLAR 5: Deep Learning Inference (The AI Engine)
% ---------------------------------------------------------
fprintf('[5/6] Executing Deep Learning CNN Inference...\n');
% Save the enhanced image temporarily for the CNN to read
tempImgPath = fullfile(scriptDir, 'temp_enhanced.jpg');
imwrite(enhancedImage, tempImgPath);

[aiLevel, aiConfidence] = predictDR_CNN(tempImgPath);

fprintf('  -> AI Prediction: Level %d\n', aiLevel);
fprintf('  -> AI Confidence: %.2f%%\n\n', aiConfidence * 100);

% ---------------------------------------------------------
% GRAD-CAM HEATMAP GENERATION
% ---------------------------------------------------------
fprintf('[6/6] Generating Explainable AI Heatmap (Grad-CAM)...\n');

% Check if the real model is loaded for Grad-CAM
modelPath = fullfile(scriptDir, 'ai_training', 'trained_DR_ResNet.mat');
gradCAMMap = [];

if exist(modelPath, 'file')
    try
        modelData = load(modelPath, 'trainedNet');
        net = modelData.trainedNet;
        
        % Read and resize image for Grad-CAM
        imgForGradCAM = imread(tempImgPath);
        if size(imgForGradCAM, 3) == 1
            imgForGradCAM = cat(3, imgForGradCAM, imgForGradCAM, imgForGradCAM);
        end
        imgResized = imresize(imgForGradCAM, [224 224]);
        
        % Find the last convolutional layer for Grad-CAM
        layerNames = {net.Layers.Name};
        convLayers = {};
        for li = 1:length(net.Layers)
            if isa(net.Layers(li), 'nnet.cnn.layer.Convolution2DLayer') || ...
               isa(net.Layers(li), 'nnet.cnn.layer.BatchNormalizationLayer')
                convLayers{end+1} = net.Layers(li).Name; %#ok<SAGROW>
            end
        end
        
        if ~isempty(convLayers)
            lastConvLayer = convLayers{end};
            gradCAMMap = gradCAM(net, imgResized, aiLevel + 1, 'FeatureLayer', lastConvLayer);
            % Resize heatmap to match original enhanced image size
            gradCAMMap = imresize(gradCAMMap, [size(enhancedImage, 1), size(enhancedImage, 2)]);
            fprintf('  -> Grad-CAM heatmap generated successfully.\n\n');
        else
            fprintf('  -> Could not find convolutional layer for Grad-CAM.\n\n');
        end
    catch e
        fprintf('  -> Grad-CAM generation skipped: %s\n\n', e.message);
    end
else
    fprintf('  -> Grad-CAM skipped (Mock Mode - no trained model file).\n\n');
end

% Cleanup temp file
if exist(tempImgPath, 'file')
    delete(tempImgPath);
end

% ---------------------------------------------------------
% HYBRID FAIL-SAFE EVALUATION
% ---------------------------------------------------------
confidenceThreshold = 0.75;
finalLevel = -1;
finalReason = "";
decisionSource = "";

if aiConfidence < confidenceThreshold
    fprintf('  [!] WARNING: AI Confidence is below safety threshold (%.0f%%).\n', confidenceThreshold * 100);
    fprintf('  [!] TRIGGERING FAIL-SAFE SWITCH...\n');
    fprintf('  -> Classical Severity: %s\n\n', classicName);
    
    finalLevel = classicLevel;
    finalReason = "AI Confidence too low. Resorted to Classical mathematical lesion counting (Fail-Safe).";
    decisionSource = "FAIL-SAFE (Classical Algorithm)";
    
elseif aiLevel == 0 && classicLevel >= 2
    fprintf('  [!] CRITICAL WARNING: Discordance Flagged!\n');
    fprintf('  -> AI predicts Healthy, but Classical algorithms found %d lesions.\n', ...
        (features.numMA + features.numHem + features.numExudates));
    
    finalLevel = 99;
    finalReason = "Discordance: Severe disagreement between Neural Network and Classical algorithms. Forced Human Review.";
    decisionSource = "DISCORDANCE FLAG (Human Review Required)";
else
    fprintf('  -> Safety Check Passed. Ensemble Agreement confirmed.\n\n');
    finalLevel = aiLevel;
    finalReason = sprintf('AI diagnosed Level %d with high confidence (%.0f%%).', aiLevel, aiConfidence*100);
    decisionSource = "AI (High Confidence)";
end

% ---------------------------------------------------------
% GENERATE CLINICAL REPORT TEXT
% ---------------------------------------------------------
if finalLevel == 99
    reportFeatures = features;
    finalReport = generateReport("REQUIRES HUMAN REVIEW", finalLevel, reportFeatures);
else
    reportFeatures = features;
    if finalLevel == 0
        sName = "Level 0: No DR";
    elseif finalLevel == 1
        sName = "Level 1: Mild NPDR";
    elseif finalLevel == 2
        sName = "Level 2: Moderate NPDR";
    elseif finalLevel == 3
        sName = "Level 3: Severe NPDR";
    else
        sName = "Level 4: Proliferative DR";
    end
    finalReport = generateReport(sName, finalLevel, reportFeatures);
end

% =================================================================
% VISUAL OUTPUT - FIGURE 1: Preprocessing & Segmentation
% =================================================================
figure('Name', 'Insulens: Preprocessing & Segmentation', 'Position', [50, 50, 1400, 500], 'Color', [0.1 0.1 0.15]);

subplot(1, 4, 1);
imshow(originalImage);
title('Original Fundus Image', 'Color', 'w', 'FontSize', 12);

subplot(1, 4, 2);
imshow(enhancedImage);
title('Enhanced (CLAHE)', 'Color', 'w', 'FontSize', 12);

subplot(1, 4, 3);
imshow(imoverlay(enhancedImage, bwperim(discMask), [1 0 0]));
title('Optic Disc (Red)', 'Color', 'w', 'FontSize', 12);

subplot(1, 4, 4);
imshow(imoverlay(origGreen, vesselMask, [0 1 0]));
title('Blood Vessels (Green)', 'Color', 'w', 'FontSize', 12);

sgtitle('INSULENS: Preprocessing & Segmentation Pipeline', 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'cyan');

% =================================================================
% VISUAL OUTPUT - FIGURE 2: Lesion Detection Overlays
% =================================================================
figure('Name', 'Insulens: Lesion Detection', 'Position', [50, 600, 1400, 500], 'Color', [0.1 0.1 0.15]);

subplot(1, 4, 1);
imshow(imoverlay(enhancedImage, maMask, [0 0.5 1]));
title(sprintf('Microaneurysms: %d (Blue)', features.numMA), 'Color', 'w', 'FontSize', 12);

subplot(1, 4, 2);
imshow(imoverlay(enhancedImage, hemMask, [1 0 0]));
title(sprintf('Hemorrhages: %d (Red)', features.numHem), 'Color', 'w', 'FontSize', 12);

subplot(1, 4, 3);
imshow(imoverlay(enhancedImage, exudateMask, [1 1 0]));
title(sprintf('Hard Exudates: %d (Yellow)', features.numExudates), 'Color', 'w', 'FontSize', 12);

% Combined Lesion Overlay
subplot(1, 4, 4);
combinedOverlay = enhancedImage;
combinedOverlay = imoverlay(combinedOverlay, maMask, [0 0.5 1]);
combinedOverlay = imoverlay(combinedOverlay, hemMask, [1 0 0]);
combinedOverlay = imoverlay(combinedOverlay, exudateMask, [1 1 0]);
imshow(combinedOverlay);
title('All Lesions Combined', 'Color', 'w', 'FontSize', 12);

sgtitle('INSULENS: Explainable AI Lesion Detection', 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'cyan');

% =================================================================
% VISUAL OUTPUT - FIGURE 3: Grad-CAM Heatmap & Final Decision
% =================================================================
figure('Name', 'Insulens: AI Decision & Explainability', 'Position', [100, 100, 1200, 500], 'Color', [0.1 0.1 0.15]);

subplot(1, 3, 1);
imshow(enhancedImage);
title('Input to AI', 'Color', 'w', 'FontSize', 12);

subplot(1, 3, 2);
if ~isempty(gradCAMMap)
    imshow(enhancedImage);
    hold on;
    imagesc(gradCAMMap, 'AlphaData', 0.5);
    colormap(gca, jet);
    colorbar('Color', 'w');
    hold off;
    title('Grad-CAM: Where the AI Looked', 'Color', 'w', 'FontSize', 12);
else
    imshow(combinedOverlay);
    title('Lesion Overlay (Grad-CAM N/A)', 'Color', 'w', 'FontSize', 12);
end

subplot(1, 3, 3);
axis off;
% Build the decision text box
decisionText = {
    '--- HYBRID DECISION ---', ...
    '', ...
    sprintf('AI Prediction: Level %d', aiLevel), ...
    sprintf('AI Confidence: %.1f%%', aiConfidence * 100), ...
    '', ...
    sprintf('Classical: %s', classicName), ...
    sprintf('MAs: %d | Hem: %d | Ex: %d', features.numMA, features.numHem, features.numExudates), ...
    '', ...
    sprintf('Decision Source: %s', decisionSource), ...
    ''
};

if finalLevel == 99
    decisionText{end+1} = 'RESULT: HUMAN REVIEW REQUIRED';
    resultColor = [1 0.5 0]; % Orange
else
    decisionText{end+1} = sprintf('FINAL RESULT: LEVEL %d', finalLevel);
    if finalLevel <= 1
        resultColor = [0 1 0]; % Green
    elseif finalLevel == 2
        resultColor = [1 1 0]; % Yellow
    else
        resultColor = [1 0 0]; % Red
    end
end

text(0.5, 0.5, decisionText, ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', ...
    'FontSize', 13, ...
    'FontName', 'Courier', ...
    'Color', resultColor, ...
    'FontWeight', 'bold', ...
    'BackgroundColor', [0.15 0.15 0.2], ...
    'EdgeColor', resultColor, ...
    'LineWidth', 2, ...
    'Margin', 15);
title('Final Hybrid Decision', 'Color', 'w', 'FontSize', 14);

sgtitle('INSULENS: AI Explainability & Clinical Decision', 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'cyan');

% =================================================================
% PRINT THE FINAL CLINICAL REPORT TO COMMAND WINDOW
% =================================================================
fprintf('\n');
disp(finalReport);

fprintf('==================================================\n');
fprintf('  PIPELINE COMPLETE\n');
fprintf('==================================================\n');
