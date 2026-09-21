function exudateMask = detectExudates(enhancedImage, discMask)
% detectExudates extracts hard exudates (bright yellow lesions) from a fundus image.
% It uses intensity thresholding while ignoring the bright optic disc.
%
% Inputs:
%   enhancedImage - The preprocessed image (uint8), ideally CLAHE enhanced
%   discMask - Binary mask of the optic disc (so we don't confuse it for an exudate)
%
% Outputs:
%   exudateMask - Binary mask of the hard exudates

    % 1. Brightness Thresholding
    % Exudates are highly reflective and appear very bright.
    % We use adaptive thresholding or a hard percentile threshold.
    
    % Let's use the green channel equivalent (enhancedImage is already based on green)
    % A simple global threshold based on intensity works well on CLAHE images
    threshold = 200; % Intensity out of 255 (needs tuning based on enhancement level)
    candidateMask = enhancedImage > threshold;
    
    % 2. Remove the Optic Disc
    % The optic disc is also very bright. We must subtract its mask.
    % We dilate the disc mask slightly just to be safe around its edges.
    seDisc = strel('disk', 20);
    dilatedDisc = imdilate(discMask, seDisc);
    
    candidateMask(dilatedDisc) = false;
    
    % 3. Morphological Cleaning
    % Remove single pixel noise and bridge small gaps
    exudateMask = bwareaopen(candidateMask, 10); % Exudates must be at least 10 pixels
    
    % Optional: ensure they are not perfectly tubular (like reflections on vessels)
    % by analyzing shape (eccentricity), but bwareaopen is a good MVP start.
end
