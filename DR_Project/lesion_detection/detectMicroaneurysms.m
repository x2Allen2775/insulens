function maMask = detectMicroaneurysms(enhancedImage, vesselMask)
% detectMicroaneurysms extracts microaneurysms (small dark red dots).
% It finds small dark circular regions and ignores the blood vessels.
%
% Inputs:
%   enhancedImage - The preprocessed image (uint8)
%   vesselMask - Binary mask of the blood vessels
%
% Outputs:
%   maMask - Binary mask of the microaneurysms

    % 1. Invert image (so dark lesions become bright spots)
    invertedImg = imcomplement(enhancedImage);
    
    % 2. Morphological Top-Hat Transform
    % Top-hat isolates small bright regions that fit inside the structuring element
    % Microaneurysms are small, so a small disk is perfect.
    seMA = strel('disk', 6);
    topHat = imtophat(invertedImg, seMA);
    
    % 3. Thresholding
    % Extract the strongest candidate spots
    threshold = 40; % Out of 255 (tuning needed based on CLAHE)
    candidateMask = topHat > threshold;
    
    % 4. Remove Blood Vessels
    % Microaneurysms can lie on vessels, but our simple model will just 
    % subtract the vessel mask to avoid classifying vessel bends as lesions.
    % We dilate the vessel mask slightly.
    seVessel = strel('disk', 2);
    dilatedVessels = imdilate(vesselMask, seVessel);
    candidateMask(dilatedVessels) = false;
    
    % 5. Shape Filtering (Circularity & Area)
    % Keep only small, somewhat circular objects
    maMask = false(size(candidateMask));
    CC = bwconncomp(candidateMask);
    props = regionprops(CC, 'Area', 'Eccentricity', 'PixelIdxList');
    
    for i = 1:length(props)
        % Microaneurysms are small (area < 150) and relatively round (low eccentricity)
        if props(i).Area > 3 && props(i).Area < 150 && props(i).Eccentricity < 0.85
            maMask(props(i).PixelIdxList) = true;
        end
    end
end
