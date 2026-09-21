% test_pipeline.m
% This script tests Pillar 1 (Quality Assessment) and Pillar 2 (Preprocessing)

% 1. Setup paths robustly based on where this script is located
scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir, 'quality_assessment'));
addpath(fullfile(scriptDir, 'preprocessing'));

% 2. Select a test image from the IDRiD dataset
testImagePath = fullfile(scriptDir, 'raw_images', 'A. Segmentation', '1. Original Images', 'a. Training Set', 'IDRiD_12.jpg');

fprintf('--- Testing Image: %s ---\n', testImagePath);

% ---------------------------------------------------------
% PILLAR 1: Image Quality Assessment
% ---------------------------------------------------------
fprintf('\nExecuting Pillar 1: Image Quality Assessment...\n');
[status, feedback, metrics] = assessQuality(testImagePath);

fprintf('Status: %s\n', status);
fprintf('Feedback: %s\n', feedback);
fprintf('Blur Score (Var of Laplacian): %.6f\n', metrics.blur_score);
fprintf('Mean Illumination: %.4f\n', metrics.mean_illumination);

% ---------------------------------------------------------
% PILLAR 2: Preprocessing and Enhancement
% ---------------------------------------------------------
% Only proceed to enhancement if the image is NOT UNGRADABLE
if status ~= "UNGRADABLE"
    fprintf('\nExecuting Pillar 2: Preprocessing...\n');
    [enhancedImage, origGreen] = preprocessFundus(testImagePath);
    
    % ---------------------------------------------------------
    % PILLAR 3: Retinal Structure Segmentation
    % ---------------------------------------------------------
    fprintf('\nExecuting Pillar 3: Segmentation...\n');
    addpath(fullfile(scriptDir, 'segmentation'));
    
    [discCenter, discMask] = locateOpticDisc(enhancedImage);
    vesselMask = segmentVessels(enhancedImage);
    
    fprintf('Optic Disc located at: X=%d, Y=%d\n', discCenter(1), discCenter(2));
    
    % Display Results
    figure('Name', 'Pillars 1, 2 & 3 Results', 'Position', [100, 100, 1200, 800]);
    
    % Row 1: Preprocessing
    subplot(2, 3, 1);
    imshow(testImagePath);
    title('Original Image');
    
    subplot(2, 3, 2);
    imshow(origGreen);
    title('Original Green Channel');
    
    subplot(2, 3, 3);
    imshow(enhancedImage);
    title('Enhanced (CLAHE)');
    
    % Row 2: Segmentation
    subplot(2, 3, 4);
    % Overlay Optic Disc on enhanced image
    imshow(imoverlay(enhancedImage, bwperim(discMask), [1 0 0]));
    title('Optic Disc Localization');
    
    subplot(2, 3, 5);
    imshow(vesselMask);
    title('Vessel Mask');
    
    subplot(2, 3, 6);
    % Overlay Vessels in green on original image
    imshow(imoverlay(origGreen, vesselMask, [0 1 0]));
    title('Vessels Overlay');
    
    % ---------------------------------------------------------
    % PILLAR 4: Lesion Detection
    % ---------------------------------------------------------
    fprintf('\nExecuting Pillar 4: Lesion Detection...\n');
    addpath(fullfile(scriptDir, 'lesion_detection'));
    
    exudateMask = detectExudates(enhancedImage, discMask);
    maMask = detectMicroaneurysms(enhancedImage, vesselMask);
    hemMask = detectHemorrhages(enhancedImage, vesselMask);
    
    fprintf('Detected %d Microaneurysms.\n', numel(regionprops(maMask, 'Area')));
    fprintf('Detected %d Hemorrhages.\n', numel(regionprops(hemMask, 'Area')));
    fprintf('Detected %d Exudate regions.\n', numel(regionprops(exudateMask, 'Area')));
    
    % Display Lesion Results in a new Figure
    figure('Name', 'Pillar 4: Lesion Detection', 'Position', [150, 150, 1200, 400]);
    
    % Microaneurysms
    subplot(1, 3, 1);
    imshow(imoverlay(enhancedImage, maMask, [0 0 1])); % Blue
    title('Microaneurysms (Blue)');
    
    % Hemorrhages
    subplot(1, 3, 2);
    imshow(imoverlay(enhancedImage, hemMask, [1 0 0])); % Red
    title('Hemorrhages (Red)');
    
    % Exudates
    subplot(1, 3, 3);
    imshow(imoverlay(enhancedImage, exudateMask, [1 1 0])); % Yellow
    title('Hard Exudates (Yellow)');
    
    % ---------------------------------------------------------
    % PILLAR 5: DR Severity Classification
    % ---------------------------------------------------------
    fprintf('\nExecuting Pillar 5: DR Severity Classification...\n');
    addpath(fullfile(scriptDir, 'classification'));
    
    % Build feature vector
    features.numMA = numel(regionprops(maMask, 'Area'));
    features.numHem = numel(regionprops(hemMask, 'Area'));
    features.numExudates = numel(regionprops(exudateMask, 'Area'));
    
    [severityLevel, severityName] = classifyDR(features);
    
    % ---------------------------------------------------------
    % PILLAR 6 & 7 & 8: Explainable AI & Final Report
    % ---------------------------------------------------------
    fprintf('\nExecuting Pillar 6: Generating Explainable Report...\n\n');
    addpath(fullfile(scriptDir, 'explainability'));
    
    finalReport = generateReport(severityName, severityLevel, features);
    
    % Print the final report to the Command Window
    disp(finalReport);
    
    % Optional: Add a title to the main figure showing the prediction
    figure(1); % Focus back on the first figure
    sgtitle(['Final AI Prediction: ', char(severityName)], 'FontSize', 16, 'FontWeight', 'bold', 'Color', 'red');

else
    fprintf('\nImage is UNGRADABLE. Pipeline stopped. Please recapture.\n');
end
