%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Only for SNN analysis, NOT the test after train
%%% Input: snn, text_x, text_y
%%% Output: snn states
%%% Revision: 2020-5-28
%%% Author: Tielin Zhang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [layer1,layer2,layer3]=snntest(nn,test_x,test_y)
    % Test samples
    N = size(test_x,1);
    % Number of neurons in each layer
    d1 = nn.size(1);
    d2= nn.size(2);
    d3 = nn.size(3);
    % Neuron states
    layer1 = zeros(N,d1);
    layer2 = zeros(N,d2);
    layer3 = zeros(N,d3);
    % Batch size
    batchsize=nn.opts.batchsize;
    % Number of batch
    numbatches_test=N/batchsize;
    % Time slot
    dt = nn.opts.dt;
    % Simulation time
    T = nn.opts.T;
    % Test samples
    for index=1:numbatches_test
        % Feedforward propagation
        for ti=dt:dt:T
            nn=snnff(nn,index,test_x,nn.opts,ti);
        end
        % Input state clamp
        nn.layers{1}=test_x((index-1)*batchsize+1:index*batchsize,:);
        % Homeo-V
        nn=snnhomeov(nn,nn.opts);
        % Save neuron states
       layer1((index-1)*batchsize+1:index*batchsize,:) = nn.layers{1,1};
       layer2((index-1)*batchsize+1:index*batchsize,:) = nn.layers{1,2};
       layer3((index-1)*batchsize+1:index*batchsize,:) = nn.layers{1,3};
    end
end