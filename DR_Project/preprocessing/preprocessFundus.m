function [enhancedImage, I_green] = preprocessFundus(imagePath)
% preprocessFundus standardizes and enhances a fundus image.
% It applies Illumination Normalization and CLAHE to the Green channel.
%
% Inputs:
%   imagePath - String path to the fundus image
%
% Outputs:
%   enhancedImage - The fully preprocessed and enhanced image (uint8)
%   I_green - The original extracted green channel (for comparison)

    % 1. Read Image
    if ischar(imagePath) || isstring(imagePath)
        I = imread(imagePath);
    else
        I = imagePath; % Assume it's already an image array
    end
    
    % Resize for consistency (optional, but helps standardizing filter sizes)
    % Target width of 800px, keeping aspect ratio
    targetWidth = 800;
    [rows, cols, ~] = size(I);
    scale = targetWidth / cols;
    I_resized = imresize(I, scale);

    % 2. Color Channel Analysis (Extract Green Channel)
    % The green channel provides the best contrast for blood vessels and hemorrhages
    if size(I_resized, 3) == 3
        I_green = I_resized(:,:,2);
    else
        I_green = I_resized;
    end
    
    % Create a mask for the Field of View (FOV) to exclude black borders
    threshold = 10;
    mask = I_green > threshold;
    mask = bwareaopen(mask, 1000); % Remove small noise
    % Create a morphological disk to slightly erode mask to avoid edge artifacts
    se = strel('disk', 5);
    mask = imerode(mask, se);

    % 3. Illumination Normalization
    % Estimate the background by applying a large median or average filter
    % This captures the uneven lighting without capturing fine vessels
    backgroundFilterSize = round(min(size(I_green)) / 10); % Adaptive size
    background = medfilt2(I_green, [backgroundFilterSize backgroundFilterSize]);
    
    % Subtract background to normalize
    I_normalized = imsubtract(I_green, background);
    
    % Shift the values back to a positive range and apply mask
    I_normalized = I_normalized + 128; % Shift back to middle
    I_normalized(~mask) = 0; % Clear outside FOV

    % 4. Contrast Normalization (CLAHE)
    % Contrast Limited Adaptive Histogram Equalization
    % Extract non-zero pixels for better histogram equalization
    clipLimit = 0.02; % Tuning parameter
    numTiles = [8 8];
    
    enhancedImage = adapthisteq(I_normalized, 'NumTiles', numTiles, 'ClipLimit', clipLimit, 'Distribution', 'uniform');
    
    % 5. Minor Denoising
    % A small median filter to remove salt and pepper noise without losing microaneurysms
    enhancedImage = medfilt2(enhancedImage, [3 3]);
    
    % Apply mask one last time
    enhancedImage(~mask) = 0;

end
