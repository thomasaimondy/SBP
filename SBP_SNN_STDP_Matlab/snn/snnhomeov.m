%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FOR SNN Homeo-V
%%% Input: snn, opts
%%% Output: snn
%%% Revision: 2020-5-28
%%% Author: Tielin Zhang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function nn=snnhomeov(nn,opts)
    % Number of layers
    n=nn.n;
    % Configuration of epsilon
    eps=opts.epsilon;
    % Configuration of iteration
    n_itreations=opts.n_it_neg;
    % Homeo-V propagation
    % TIPS: input-output balance tuning
    for num=1:n_itreations
        % Only for hidden layers
        for i=2:n-1
            % Number of neurons
            [m,~]=size(nn.layers{i});
            % Homeostatic V balance
            % TIPS: Input and output of LIF is balanced
            % E.G.: sum(W_{i,j}*X{i}) toward Y{j}
            detHomeo=nn.layers{i}-(nn.layers{i-1}*nn.W{i-1}+nn.layers{i+1}*nn.W{i}')-repmat(nn.b{i},m,1);
            % Update of layers
            nn.layers{i}=nn.layers{i}-eps*detHomeo;
            % Normalization
            nn.layers{i}=snn_sigmoid(nn.layers{i},opts,1);
        end
        % For the output layer
        [m,~]=size(nn.layers{n});
        % Homeostatic V balance is based on only pre-layers
        change=nn.layers{n}-(nn.layers{n-1}*nn.W{n-1})-repmat(nn.b{n},m,1);
        % Update of output layer
        nn.layers{n}=nn.layers{n}-eps*change;
        % Normalization
        nn.layers{n} = snn_sigmoid(nn.layers{n},opts,5);
    end
    % Save Homeo-V states
    nn.particles = nn.layers;
end


