function reportString = generateReport(severityName, severityLevel, features)
% generateReport builds a clinical and explainable text report.
%
% Inputs:
%   severityName - String (e.g., "Level 2: Moderate NPDR")
%   severityLevel - Integer (0 to 4)
%   features - Struct with lesion counts
%
% Outputs:
%   reportString - Formatted text block

    % 1. Header
    report = sprintf('================================================\n');
    report = sprintf('%sEXPLAINABLE AI SCREENING REPORT\n', report);
    report = sprintf('%s================================================\n\n', report);
    
    % 2. AI Prediction
    report = sprintf('%sAI PREDICTION: %s\n\n', report, severityName);
    
    % 3. Explanation (The "Why")
    report = sprintf('%sREASONING (EXPLAINABILITY):\n', report);
    report = sprintf('%sThe AI made this decision because it detected:\n', report);
    report = sprintf('%s  - %d Microaneurysms (Early vascular damage)\n', report, features.numMA);
    report = sprintf('%s  - %d Hemorrhages (Disease progression / bleeding)\n', report, features.numHem);
    if features.numExudates > 0
        report = sprintf('%s  - %d regions of Hard Exudates (Vascular leakage)\n', report, features.numExudates);
    else
        report = sprintf('%s  - No Hard Exudates detected.\n', report);
    end
    
    % 4. Clinical Referral Recommendation
    report = sprintf('%s\nREFERRAL RECOMMENDATION:\n', report);
    if severityLevel == 0
        report = sprintf('%s🟢 LOW RISK: No significant DR features detected. Routine annual eye screening.\n', report);
    elseif severityLevel == 1
        report = sprintf('%s🟡 POSSIBLE EARLY DR: Minor abnormalities detected. Ophthalmic examination recommended within 6 months.\n', report);
    elseif severityLevel == 2
        report = sprintf('%s🟠 MODERATE RISK: Multiple lesions detected. Medical eye examination recommended soon.\n', report);
    else
        report = sprintf('%s🔴 HIGH RISK / URGENT REFERRAL: Severe disease markers present. Urgent ophthalmologist evaluation required.\n', report);
    end
    
    % 5. Disclaimer
    report = sprintf('%s\n------------------------------------------------\n', report);
    report = sprintf('%sDISCLAIMER: This system is intended for screening \n', report);
    report = sprintf('%sand decision support in rural settings. It does NOT \n', report);
    report = sprintf('%sreplace an examination or diagnosis by a qualified ophthalmologist.\n', report);
    report = sprintf('%s================================================\n', report);
    
    reportString = report;
end
