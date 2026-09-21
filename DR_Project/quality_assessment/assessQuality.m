function [status, feedback, metrics] = assessQuality(imagePath)
% assessQuality evaluates a fundus image for adequacy.
% It checks for focus (blur) and illumination.
%
% Inputs:
%   imagePath - String path to the fundus image
%
% Outputs:
%   status - "GOOD", "BORDERLINE", or "UNGRADABLE"
%   feedback - String explaining the problems if any
%   metrics - Struct containing the computed quality metrics

    % 1. Read the image
    try
        I = imread(imagePath);
    catch
        status = "UNGRADABLE";
        feedback = "Error: Could not read image file.";
        metrics = struct();
        return;
    end
    
    % Convert to grayscale for metric calculations
    if size(I, 3) == 3
        I_gray = rgb2gray(I);
    else
        I_gray = I;
    end
    
    % Ensure double precision for calculations (0 to 1 range)
    I_double = im2double(I_gray);

    %% 2. Calculate Blur (Focus) using Variance of Laplacian
    % A sharp image has high variance of Laplacian. A blurred image has low variance.
    laplacianFilter = fspecial('laplacian', 0.2);
    laplacianImage = imfilter(I_double, laplacianFilter, 'replicate');
    varianceLaplacian = var(laplacianImage(:));
    
    % Thresholds for blur (Empirically tuned for IDRiD dataset)
    BLUR_THRESHOLD_SEVERE = 0.0001; % Highly blurred
    BLUR_THRESHOLD_MILD = 0.0002;   % Slightly blurred
    
    %% 3. Calculate Illumination (Brightness)
    % Mean intensity of the image (excluding the dark background borders if possible)
    % Simple threshold to separate circular fundus from black background
    fundusMask = I_double > 0.05; 
    
    if any(fundusMask(:))
        meanIllumination = mean(I_double(fundusMask));
    else
        meanIllumination = mean(I_double(:));
    end
    
    % Thresholds for illumination
    ILLUM_TOO_DARK = 0.15;
    ILLUM_TOO_BRIGHT = 0.85;
    
    %% 4. Compile Metrics
    metrics.blur_score = varianceLaplacian;
    metrics.mean_illumination = meanIllumination;
    
    %% 5. Decision Logic
    problems = string([]);
    isUngradable = false;
    isBorderline = false;
    
    % Check Blur
    if varianceLaplacian < BLUR_THRESHOLD_SEVERE
        problems(end+1) = "Severe blur detected.";
        isUngradable = true;
    elseif varianceLaplacian < BLUR_THRESHOLD_MILD
        problems(end+1) = "Mild blur detected.";
        isBorderline = true;
    end
    
    % Check Illumination
    if meanIllumination < ILLUM_TOO_DARK
        problems(end+1) = "Image is too dark (low illumination).";
        isUngradable = true;
    elseif meanIllumination > ILLUM_TOO_BRIGHT
        problems(end+1) = "Image is overexposed (too bright).";
        isBorderline = true;
    end
    
    % Determine Final Status
    if isUngradable
        status = "UNGRADABLE";
        feedback = "IMAGE QUALITY: UNGRADABLE. " + strjoin(problems, " ");
        feedback = feedback + " Recommendation: Please recapture the retinal image. Hold the camera steady and check lighting.";
    elseif isBorderline
        status = "BORDERLINE";
        feedback = "IMAGE QUALITY: BORDERLINE. " + strjoin(problems, " ");
        feedback = feedback + " Proceeding to enhancement, but results may be suboptimal.";
    else
        status = "GOOD";
        feedback = "IMAGE QUALITY: GOOD.";
    end
end
