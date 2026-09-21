function hemMask = detectHemorrhages(enhancedImage, vesselMask)
% detectHemorrhages extracts retinal hemorrhages (larger dark regions).
%
% Inputs:
%   enhancedImage - The preprocessed image (uint8)
%   vesselMask - Binary mask of the blood vessels
%
% Outputs:
%   hemMask - Binary mask of the hemorrhages

    % 1. Invert image
    invertedImg = imcomplement(enhancedImage);
    
    % 2. Morphological Top-Hat Transform
    % We use a larger disk to capture hemorrhages, but not as large as the whole retina
    seHem = strel('disk', 15); 
    topHat = imtophat(invertedImg, seHem);
    
    % 3. Thresholding
    threshold = 30; % Slightly lower threshold than MAs because hemorrhages can be diffuse
    candidateMask = topHat > threshold;
    
    % 4. Remove Blood Vessels
    seVessel = strel('disk', 3);
    dilatedVessels = imdilate(vesselMask, seVessel);
    candidateMask(dilatedVessels) = false;
    
    % 5. Shape Filtering (Area)
    % Keep only larger objects (MAs are < 150, Hemorrhages are generally > 150)
    hemMask = false(size(candidateMask));
    CC = bwconncomp(candidateMask);
    props = regionprops(CC, 'Area', 'PixelIdxList');
    
    for i = 1:length(props)
        if props(i).Area >= 150
            hemMask(props(i).PixelIdxList) = true;
        end
    end
end
