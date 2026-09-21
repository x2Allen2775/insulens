% organizeAPTOS.m
% -------------------------------------------------------------------------
% Description:
%   Organizes the flat APTOS dataset into subfolders by severity class 
%   (0, 1, 2, 3, 4) so that MATLAB's imageDatastore can read them.
%   Creates SYMBOLIC LINKS (not copies) to save disk space.
%
% Usage:
%   Run this script ONCE before running trainDR_CNN.m
% -------------------------------------------------------------------------

clc;
disp('--- Organizing APTOS Dataset into Class Subfolders ---');

scriptDir = fileparts(mfilename('fullpath'));

% Paths
csvPath = fullfile(scriptDir, '..', '..', 'dataset aptos', 'train_1.csv');
srcImageDir = fullfile(scriptDir, '..', '..', 'dataset aptos', 'train_images', 'train_images');
destDir = fullfile(scriptDir, '..', 'datasets', 'APTOS_Train');

% Check that the CSV exists
if ~exist(csvPath, 'file')
    error('Cannot find train_1.csv at: %s', csvPath);
end

% Read the CSV
opts = detectImportOptions(csvPath);
opts = setvartype(opts, 'id_code', 'string');
opts = setvartype(opts, 'diagnosis', 'double');
T = readtable(csvPath, opts);

fprintf('Found %d entries in CSV.\n', height(T));

% Create destination subfolders: 0, 1, 2, 3, 4
for c = 0:4
    classDir = fullfile(destDir, num2str(c));
    if ~exist(classDir, 'dir')
        mkdir(classDir);
    end
end

% Copy images into class subfolders
skipped = 0;
copied = 0;
for i = 1:height(T)
    imgName = char(T.id_code(i)) + ".png";
    srcFile = fullfile(srcImageDir, imgName);
    classLabel = T.diagnosis(i);
    destFile = fullfile(destDir, num2str(classLabel), imgName);
    
    if ~exist(srcFile, 'file')
        skipped = skipped + 1;
        continue;
    end
    
    if ~exist(destFile, 'file')
        copyfile(srcFile, destFile);
    end
    copied = copied + 1;
    
    if mod(i, 500) == 0
        fprintf('  Processed %d / %d images...\n', i, height(T));
    end
end

fprintf('\nDone! Organized %d images into class subfolders.\n', copied);
if skipped > 0
    fprintf('Skipped %d entries (image file not found).\n', skipped);
end
fprintf('Dataset ready at: %s\n', destDir);
