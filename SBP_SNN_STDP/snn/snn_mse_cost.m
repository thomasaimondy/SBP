%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Calculate MSE cost of SNNs
%%% Input: snn, y^
%%% Output: MSE Cost
%%% Revision: 2020-5-28
%%% Author: Tielin Zhang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function result=snn_mse_cost(nn,y)
    % sum(y-y^)^2
    result=sum((nn.layers{end}-y).^2,2);
end