%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FOR SNN DALE LAW CONSTRAIN FOR W
%%% Input: W, opts, alpha, opts
%%% Output: revised alpha
%%% Revision: 2020-6-14
%%% Author: Tielin Zhang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function alpha_change = snn_dalelaw(W,deltaW,alpha,opts)
    % Standard alpha
    alpha_change = repmat(alpha,1,length(W));
    % Temporal DetW.*W
    tmp = W.*deltaW;
    % Index for E neurons
    positive_index = find(tmp>=0);
    % only E neurons unchanged
    alpha_change(positive_index) = 1*alpha;
    % Index for I neurons
    negative_index = find(tmp<0);
    % W./DetW
    declay = -W./deltaW;
    % Only I neurons changed
    eta = 0.1;
    alpha_change(negative_index) = eta.*declay(negative_index);
end