function vesselMask = segmentVessels(enhancedImage)
% segmentVessels extracts the blood vessel network from a fundus image.
% It uses morphological bottom-hat filtering on the CLAHE enhanced image.
%
% Inputs:
%   enhancedImage - The preprocessed image (uint8)
%
% Outputs:
%   vesselMask - Binary mask of the blood vessels (1 = vessel, 0 = background)

    % 1. Morphological Bottom-Hat Transform
    % Blood vessels appear darker than their surroundings.
    % The bottom-hat filter highlights dark structures on a bright background.
    % We use a disk structuring element slightly larger than the thickest vessel.
    se = strel('disk', 12);
    bottomHat = imbothat(enhancedImage, se);
    
    % 2. Thresholding
    % Adaptive thresholding works well for vessels due to varying thicknesses
    T = adaptthresh(bottomHat, 0.4, 'ForegroundPolarity', 'bright');
    binaryVessels = imbinarize(bottomHat, T);
    
    % 3. Noise Removal (Post-processing)
    % Remove small isolated noise pixels that aren't part of a continuous vessel network
    vesselMask = bwareaopen(binaryVessels, 150); % Remove objects smaller than 150 pixels
    
    % Optional: Fill small holes inside vessels
    vesselMask = imfill(vesselMask, 'holes');
end
