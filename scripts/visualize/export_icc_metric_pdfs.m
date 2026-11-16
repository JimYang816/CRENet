% Export the three metric plots as ICC single-column vector PDFs.
% Run this script explicitly to replace the PDFs in outputs/paper.
% The individual plot scripts still only display figures.

visualizeDirectory = fileparts(mfilename('fullpath'));
paperDirectory = fullfile(fileparts(fileparts(visualizeDirectory)), 'outputs', 'paper');
if ~isfolder(paperDirectory)
    mkdir(paperDirectory);
end

run(fullfile(visualizeDirectory, 'plot_nof1_ncs1_ber.m'));
berFigures = figures;
run(fullfile(visualizeDirectory, 'plot_nof1_ncs1_nmse.m'));

% print honors the exact custom PaperSize on MATLAB R2024b.
print(berFigures.nof1, fullfile(paperDirectory, 'NOF1-BER.pdf'), '-dpdf', '-painters'); %#ok<PRTPT>
print(berFigures.ncs1, fullfile(paperDirectory, 'NCS1-BER.pdf'), '-dpdf', '-painters'); %#ok<PRTPT>
print(figureNmse, fullfile(paperDirectory, 'NMSE-SNR.pdf'), '-dpdf', '-painters'); %#ok<PRTPT>
fprintf('Saved three 3.5-inch ICC single-column vector PDFs to %s\n', paperDirectory);
