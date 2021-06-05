%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FOR SNN SETUP
%%% Input: opts
%%% Output: snn
%%% Revision: 2020-5-28 2020-6-14 2020-8-12
%%% Author: Tielin Zhang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function nn = snnsetup(opts)
    % Neurons in SNN
    nn.size   = opts.architecture;
    % Layers in SNN
    nn.n      = numel(nn.size);
    % Proportion of E/I neurons
    if(opts.set_EI)
        for i=1:nn.n
            % Random select ids
            rand_choose = randperm(nn.size(i));
            num_neurnon_negtive = ceil(nn.size(i)*double(opts.negtive_rate(i)));
            % Index of I neurons
            nn.index_negative{i} = rand_choose(1:num_neurnon_negtive);
            % Index of E neurons
            nn.index_positive{i} = rand_choose(num_neurnon_negtive+1:end);
        end
    else
        for i=1:nn.n
            % Index of I neurons
            nn.index_negative{i} = 1:0;
            % Index of E neurons (all E)
            nn.index_positive{i} = 1:nn.size(i);
        end
    end
    % Set initial W weights
    for i = 2 : nn.n
        % Only set hidden layers
        if(opts.set_EI && i~=nn.n)
            % Initialization of W 
            % TIPS: E/I is dominated by index instead of initial W
            nn.W{i - 1} = (rand(nn.size(i-1), nn.size(i)) - opts.negtive_rate(i)-0.35) * 2 * sqrt(6/ (nn.size(i) + nn.size(i - 1)));
        else
            nn.W{i - 1} = (rand(nn.size(i-1), nn.size(i)) - 0.5) * 2 * sqrt(6 / (nn.size(i) + nn.size(i - 1)));
        end
    end
    
    % Dale's law
    if(opts.set_EI)
        % For hidden layers
        for i = 1 : nn.n-1
            % Revise W to satisfy Dale's law
            % TIPS: All wegiths for E are positive
            for j = 1:length(nn.index_positive{i})
                nn.W{i}(nn.index_positive{i}(j),:) = snn_positive_relu(nn.W{i}(nn.index_positive{i}(j),:));
            end
            % TIPS: All weights for I are negative
            for j = 1:length(nn.index_negative{i})
                nn.W{i}(nn.index_negative{i}(j),:) = snn_negative_relu(nn.W{i}(nn.index_negative{i}(j),:));
            end
        end
    end
end
