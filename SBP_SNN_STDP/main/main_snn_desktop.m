%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FOR SNN TRAINING WITH SBP, Non-paralleling
%%% Feedforward: weak LIF signal propagation
%%% Learning methods: STDP, Homeostatic V, STP, Dale's law, SBP
%%% Datasets: MNIST, NETTALK, DvsGesture
%%% Input: taskname and method
%%% Output: Training and test accuracy
%%% Revision: 2020-04-07 2020-5-28 2020-6-14 2020-8-12 2021-5-8
%%% Author: Tielin Zhang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Preparation
clearvars;
currentFolder = pwd;
addpath(genpath(currentFolder));
% Random seed configuration
rand('seed',100);
% Generate log file
datestr = datestr(now, 'yyyymmddTHHMMSS');
fname_txt =['logs/log_' datestr '.txt'];
fname_model = ['models/snn_' datestr '.mat'];
fname_figpath = ['figures/' datestr '/'];
fid=fopen(fname_txt,'a+');
opts.fname_txt = fname_txt;
opts.fname_model = fname_model;
opts.fname_figpath = fname_figpath;
% Set task name, candidates [mnist, nettalk, guesture]
opts.task = 'mnist';
switch opts.task
    case 'mnist'
        opts.architecture = [784 500 10];
        opts.batchsize=50;
        load('mnist_uint8.mat');
        % Normalization
        train_x = double(train_x) / 255.0;
        test_x  = double(test_x)  / 255.0;
        train_y = double(train_y);
        test_y  = double(test_y);
        opts.numepochs=100;
        % number 0 to 1 represents 0% to 100%
        opts.proportion = 1;
    case 'nettalk'
        opts.architecture = [189 500 26];
        opts.batchsize=1;
        load('nettalk_small.mat')
        opts.numepochs=100;
        % number 0 to 1 represents 0% to 100%
        opts.proportion =1;
    case 'guesture'
        opts.architecture = [1024 500 11];
        opts.batchsize=50;
        load('DVS_gesture_100.mat');
        load('gesture_label.mat');
        [train_x,train_y] = normalizedvs(train_x,train_y);
        [test_x,test_y] = normalizedvs(test_x,test_y);
        opts.numepochs=100;
        opts.proportion =1;
    otherwise
        fprintf('error in opts.task\n');  
end

opts.n_it_neg= 20;% negative iteration times
opts.n_it_pos= 5;% positive iteration times
opts.epsilon=0.5;   % learning rate
opts.beta=0.5;   %  the proportion of cost
opts.alpha=[0.1 0.05]; % learning rates in different layers
opts.noise_type = 'none'; % norm, poisson, none
opts.set_EI = false; % dale's principle
opts.negtive_rate = [0.5 0.5 0.5];
opts.stdp_type = 'SBP';
opts.propagation_range = 0.1;
opts.fraction = 0.2;
opts.lamda_sp_pre = 1;  %[0,1]
opts.lamda_sp_post = 0.5; %[0,1]
opts.lamda_sp_bp = 1;    %[0,1]
opts.V_E = 0.2; 
opts.V_I = 0.1;
opts.V_leak = 0; % mV
opts.V_th = 0.05; % mV
opts.reset = -0.1;
opts.taum = 10; % ms
opts.tau_syn_E = 2; % ms
opts.tau_syn_I = 2; % ms
opts.g_leak_E = 5; % nS
opts.g_leak_I = 5; % nS
opts.dt = 0.1; % Integration  dt [ms]
opts.T = 10;  % ms
opts.setSTP = false;
opts.U = 0.45; %stp U
opts.tauf = 10;% stp tauf,750
opts.taud = 2;% stp taud,50
opts.taus = 2; % stp taus
opts.A = 10; % stp A,Jee
opts.fid = fid;
opts.plot = true; % plot spikes, V, g 
saveconfig(opts);

% Use part or full datasets
[items,~] = size(train_x);
train_x  = train_x(1:round(items * opts.proportion),:,:);
train_y = train_y(1:round(items * opts.proportion),:,:);
[items,~] = size(test_x);
test_x = test_x(1:round(items * opts.proportion),:,:);
test_y = test_y(1:round(items * opts.proportion),:,:);

%% Task1: shallow and deep SNN learning
tic;
nn = snnsetup(opts);
nn = snntrain(nn,train_x,train_y,test_x,test_y,opts);
% Save mdoel
nn.opts = opts;
save(fname_model,'nn');

toc;

%% Task 2: inhibitory proportion in SNN
tic;
for ii=0:0.01:1
    clearvars -except ii train_x train_y test_x test_y opts;
    opts.negtive_rate = [0.5 ii 0.5];
    disp(ii);
    fprintf(opts.fid,'\n\n Negativerate \t= \t%f \n',opts.negtive_rate(2));
    nn = snnsetup(opts);
    % Performance is saved inner snntrain
    nn = snntrain(nn,train_x,train_y,test_x,test_y,opts);
end
toc;

fclose(opts.fid);