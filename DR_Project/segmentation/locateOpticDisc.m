function [discCenter, discMask] = locateOpticDisc(enhancedImage)
% locateOpticDisc finds the optic disc in a preprocessed fundus image.
% It uses intensity thresholding and morphological operations.
%
% Inputs:
%   enhancedImage - The preprocessed image (uint8), ideally CLAHE enhanced
%
% Outputs:
%   discCenter - [x, y] coordinates of the optic disc center
%   discMask - Binary mask of the optic disc

    % The optic disc is typically the brightest large structure.
    % We use a heavy median filter to remove small bright lesions (exudates)
    blurredImage = medfilt2(enhancedImage, [25 25]);
    
    % Find the maximum intensity pixel to locate the approximate region
    maxVal = max(blurredImage(:));
    
    % Threshold to find the brightest regions (Optic Disc candidate)
    threshold = maxVal * 0.85; % Top 15% of brightest pixels
    binaryDisc = blurredImage > threshold;
    
    % Morphological closing to fill holes, then opening to remove small artifacts
    se = strel('disk', 15);
    binaryDisc = imclose(binaryDisc, se);
    binaryDisc = imopen(binaryDisc, se);
    
    % Find the largest connected component (which should be the optic disc)
    CC = bwconncomp(binaryDisc);
    numPixels = cellfun(@numel, CC.PixelIdxList);
    [~, idx] = max(numPixels);
    
    discMask = false(size(binaryDisc));
    if ~isempty(idx)
        discMask(CC.PixelIdxList{idx}) = true;
    end
    
    % Calculate the centroid (center of the optic disc)
    props = regionprops(discMask, 'Centroid');
    if ~isempty(props)
        discCenter = round(props(1).Centroid);
    else
        discCenter = [0, 0];
    end
end
