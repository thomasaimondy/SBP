%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Feedforward propagation of SNN
%%% Input: snn, batch_index, train_x, ops, current_sim_time (ti)
%%% Output: snn
%%% Revision: 2020-5-28
%%% Author: Tielin Zhang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function nn=snnff(nn,index,train_x,opts,ti)
    % SNN input
    nn.layers{1}=train_x((index-1)*opts.batchsize+1:index*opts.batchsize,:);
    % Number of layers
    n=nn.n;
    % Dynamic proportion of Homeo-V
    influence = ti/opts.T;
    % Feedforward propagation
    % Hidden and output layers
    for i=1:n
        if (i==1)
            nn.lifspike{1} = nn.layers{1} > rand * opts.V_th;
            continue;
        end
        % Load saved Homeo-V state
        homeostate = nn.particles{i};
        % LIF input
        inputs = snn_sigmoid(nn.layers{i-1}*nn.W{i-1} + repmat(nn.b{i},opts.batchsize,1),opts,1);
        % Feedforward-V
        nn = snn_lif(nn,i,inputs,opts);
        % Save V
        lifstate = nn.lifvolt{i};
        % Update V with feedforward-V and homeo-V
        nn.layers{i} = influence*lifstate + (1-influence)*homeostate ;
        % Normalization as new spikes
        nn.lifspike{i} = nn.layers{i}> opts.V_th;  %%%%%%
        nn.lifvolt{i} = nn.layers{i};
        nn.layers{i} = snn_sigmoid(nn.layers{i},opts,1);
    end
end
