%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Calculate Hebb Energy as Learning Indicator of SNN
%%% Input: snn
%%% Output: energy
%%% Revision: 2020-5-28
%%% Author: Tielin Zhang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function in_energy=snn_hebb_energy(nn)
    % Layers
    n=nn.n;
    % Energy initialization
    pre_energy=0;
    post_energy=0;
    hebb_energy=0;
    % Layer-wise calculation
    % Pre, Post state
    for num=1:n
        % Pre energy
        pre_energy=pre_energy+sum(nn.layers{num}.*nn.layers{num},2)/2;
        % Post energy
        post_energy=post_energy-nn.layers{num}*nn.b{num}';
    end
    % Hebb energy
    for i=1:n-1
        hebb_energy=hebb_energy-sum((nn.layers{i}*nn.W{i}).*nn.layers{i+1},2);
    end
    % Integration of all energy
    in_energy=pre_energy+post_energy+hebb_energy;
end
