function [severityLevel, severityName] = classifyDR(features)
% classifyDR classifies the Diabetic Retinopathy severity based on extracted features.
% It uses an explainable logic-based approach (Decision Tree equivalent) 
% based on the International Clinical DR severity scale.
%
% Inputs:
%   features - Struct containing:
%       .numMA (Microaneurysms)
%       .numHem (Hemorrhages)
%       .numExudates (Hard Exudates)
%
% Outputs:
%   severityLevel - Integer (0 to 4)
%   severityName - String (e.g., "Moderate NPDR")

    % Unpack features for readability
    ma = features.numMA;
    hem = features.numHem;
    ex = features.numExudates;
    
    % Explainable Logic Tree based on clinical guidelines
    if ma == 0 && hem == 0 && ex == 0
        severityLevel = 0;
        severityName = "Level 0: No DR";
        
    elseif hem > 20 || (ma > 0 && hem > 15 && ex > 5)
        % Severe signs: extensive hemorrhages or combinations of multiple lesions
        % Note: True PDR requires Neovascularization, which we omit in MVP
        % so we cap at Severe NPDR unless extreme.
        severityLevel = 3;
        severityName = "Level 3: Severe NPDR";
        
    elseif (ma > 5 || hem > 3) || ex > 0
        % Moderate signs: multiple MAs, some hemorrhages, or presence of exudates
        severityLevel = 2;
        severityName = "Level 2: Moderate NPDR";
        
    else
        % Mild signs: only a few microaneurysms, very few hemorrhages
        severityLevel = 1;
        severityName = "Level 1: Mild NPDR";
    end
    
    % Note: Level 4 (Proliferative DR) requires neovascularization detection, 
    % which requires a deep learning module in a future phase.
end
