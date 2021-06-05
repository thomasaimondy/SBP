%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FOR SNN TRAINING WITH SBP, Paralleling {Prop/Frac}
%%% Feedforward: weak LIF signal propagation
%%% Learning methods: STDP, Homeostatic V, STP, Dale's law, SBP
%%% Datasets: MNIST, NETTALK, DvsGesture
%%% Input: taskname and method
%%% Output: Training and test accuracy
%%% Revision: 2020-04-07 2020-5-28 2020-6-14 2020-8-12 2021-5-8
%%% Author: Tielin Zhang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clearvars;
currentFolder = pwd;
addpath(genpath(currentFolder))
rand('seed',100);

tasks = {};
% Fractor of prop and frac
for i=0.1:0.1:0.1
    for j=0.1:0.1:0.1
        %tasks{end+1}={'mnist',i,j};
        %tasks{end+1}={'nettalk',i,j};
        tasks{end+1}={'guesture',i,j};
    end
end

parpool('local',length(tasks));

for ii = 1:length(tasks)
    % Make log file
    datestr1 = datestr(now, 'yyyymmddTHHMMSS');
    fname_txt =['logs/log_', datestr1, '_', tasks{ii}{1},'_',num2str(tasks{ii}{2}) '_',num2str(tasks{ii}{3}) '.txt'];
    fname_model = ['models/snn_' datestr1 '_' tasks{ii}{1},'_',num2str(tasks{ii}{2}) '_',num2str(tasks{ii}{3}) '.mat'];
    fname_figpath = ['figures/' datestr1  '_' tasks{ii}{1},'_',num2str(tasks{ii}{2}) '_',num2str(tasks{ii}{3}) '/'];
    fid=fopen(fname_txt,'a+');
    opts = struct('configvariables',[]);
    opts.fname_txt = fname_txt;
    opts.fname_model = fname_model;
    opts.fname_figpath = fname_figpath;
    opts.task = tasks{ii}{1}; % mnist, nettalk, guesture
    train_x=[];
    test_x=[];
    train_y=[];
    test_y=[];
    switch opts.task
        case 'mnist'
            opts.architecture = [784 500 10];
            opts.batchsize=50;
            data = load('mnist_uint8.mat');
            train_x = double(data.train_x) / 255.0;
            test_x  = double(data.test_x)  / 255.0;
            train_y = double(data.train_y);
            test_y  = double(data.test_y);
            opts.numepochs=100;
            opts.proportion =1; % 0.1 for fast test
        case 'nettalk'
            opts.architecture = [189 500 26];
            opts.batchsize=1;
            data = load('nettalk_small.mat');
            train_x = data.train_x;
            train_y = data.train_y;
            test_x = data.test_x;
            test_y = data.test_y;
            opts.numepochs=100;
            opts.proportion =1;
        case 'guesture'
            opts.architecture = [1024 500 11];
            opts.batchsize=50;
            data1 = load('DVS_gesture_100.mat');
            train_x = data1.train_x;
            test_x = data1.test_x;
            data2 = load('gesture_label.mat');
            train_y = data2.train_y;
            test_y = data2.test_y;
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
    opts.alpha=[0.1 0.05];; % learning rates in different layers
    opts.noise_type = 'none'; % norm, poisson, none
    opts.set_EI = false; % dale's principle
    opts.negtive_rate = [0.5 0.5 0.5];
    opts.stdp_type = 'SBP';% differentialSTDP,symmetricSTDP, SBP
    opts.propagation_range = tasks{ii}{2}; %0.1;
    opts.fraction =tasks{ii}{3}; %0.2;
    opts.lamda_sp_pre = 1;  %[0,1]
    opts.lamda_sp_post = 0.5; %[0,1]
    opts.lamda_sp_bp = 1;    %[0,1]
    opts.V_E = 0.2; 
    opts.V_I = 0.1;
    opts.V_leak = 0; % mV
    opts.V_th = 0.05; % mV
    opts.reset = -0.5;
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
    opts.plot = false; % plot spikes, V, g 
    saveconfig(opts);
    
    [items,~] = size(train_x);
    refineitems = round(items * opts.proportion) - mod(round(items * opts.proportion),opts.batchsize) ;
    train_x  = train_x(1:refineitems,:,:);
    train_y = train_y(1:refineitems,:,:);
    [items,~] = size(test_x);
    refineitems = round(items * opts.proportion) - mod(round(items * opts.proportion),opts.batchsize) ;
    test_x = test_x(1:refineitems,:,:);
    test_y = test_y(1:refineitems,:,:);
    % Train and test the network
    nn = snnsetup(opts);
    nn = snntrain(nn,train_x,train_y,test_x,test_y,opts);
    % Save the mdoels
    nn.opts = opts;
    savepar(fname_model,nn);
    fclose(opts.fid);
end

try
      fprintf('Closing any pools....\n');
      parpool close;
    catch
end