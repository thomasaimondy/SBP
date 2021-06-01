%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Save values, This is necessary
%%% especially for parpool training
%%% Input: path, value
%%% Output: Save as mat
%%% Revision: 2020-5-28
%%% Author: Tielin Zhang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function savepar(fname_model, x)
    % Save directly as mat
    save(fname_model, 'x');
end