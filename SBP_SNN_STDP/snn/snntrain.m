%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FOR SNN train (and test inner it)
%%% Input: snn, datasets (train_x, train_y, test_x, test_y), opts
%%% Output: snn
%%% Revision: 2020-5-28
%%% Author: Tielin Zhang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function nn=snntrain(nn,train_x,train_y,test_x,test_y,opts)
    % Number of samples
    m=size(train_x,1);
    n=size(test_x,1);
    % Load configuration of batch and epoch
    batchsize=opts.batchsize;
    numepochs=opts.numepochs;
    % Batch number
    % TIPS: it have to be integer
    numbatches_train=m/batchsize; 
    numbatches_test=n/batchsize;
    % Time slot and duration
    dt = opts.dt;
    T = opts.T;
    % List for saving performance
    nn.error_train=zeros(numepochs,1);
    nn.error_test=zeros(numepochs,1);
    % Initialization of LIF
    for i=1:nn.n
        n_neuron = nn.size(i);
        % Membrane potential
        nn.lifvolt{i} = zeros(batchsize,n_neuron);
        % Conductance
        nn.lifge{i}= zeros(batchsize,n_neuron);
        % Spike
        nn.lifspike{i} = zeros(batchsize,n_neuron);
        % Spike counter
        nn.spikescount{i} = 0;
        % No bias
        nn.b{i}=zeros(1,nn.size(i));
        % Saved layer state
        % TIPS: for Homeo-V 
        nn.particles{i}=zeros(batchsize,nn.size(i));
        % Initialization of variables of STP
        nn.lifu{i} = 0.1*ones(batchsize,n_neuron);
        nn.lifx{i} = 1*ones(batchsize,n_neuron);
        nn.lifI{i} = 0.1*ones(batchsize,n_neuron); 
    end
    
    % Plot spikes
    if opts.plot == true
        % Plot 10 samples
        input_num=10;
        record_inputs = zeros(nn.size(2),round(T/dt)*input_num);
        % Record beginning spikes
        record1_spikes = zeros(nn.size(1),round(T/dt)*input_num);
        record2_spikes = zeros(nn.size(2),round(T/dt)*input_num);
        record3_spikes = zeros(nn.size(3),round(T/dt)*input_num);
        % Record target spike
        target_spikes = zeros(nn.size(3),round(T/dt)*input_num);
    end
    
    % Training of SNNs
    for epoch=1:numepochs
        % Current epoch index
        opts.epoch = epoch;
        % List for saving performance
        measurelist_train = zeros(numbatches_train,3);
        % Batch learning
        for index=1:numbatches_train
            % (1) Feedforward propagation
            for ti=dt:dt:T
                % Feedforward-V and Homeo-V integration
                nn=snnff(nn,index,train_x,opts,ti);
                % Plot Curves
                if opts.plot
                    % Save beginning epoch
                    if (ismember(index,1:1:input_num)) && (epoch==1 || mod(epoch,10)==0 || epoch==numepochs)
                        % Injection
                        %Iinj = snn_sigmoid(nn.layers{1}*nn.W{1} + repmat(nn.b{2},opts.batchsize,1),opts,1);
                        record_inputs(:,round(ti/dt)+(index-1)*(T/dt)) = nn.lifvolt{2}(1,:)';
                        % Spike
                        record1_spikes(:,round(ti/dt)+(index-1)*(T/dt)) = nn.lifspike{1}(1,:)';
                        record2_spikes(:,round(ti/dt)+(index-1)*(T/dt)) = nn.lifspike{2}(1,:)';
                        record3_spikes(:,round(ti/dt)+(index-1)*(T/dt)) = nn.lifspike{3}(1,:)';
                        batchy = train_y((index-1)*batchsize+1:index*batchsize,:);
                        target_spikes(:,round(ti/dt)+(index-1)*(T/dt)) = logical(batchy(1,:)');
                        % Save beginnings
                        save(['save_spikes_start_' num2str(epoch)], 'opts', 'record_inputs','record1_spikes','record2_spikes','record3_spikes','target_spikes');
                    end
                   
                end
            end
            % (2) Homeostatic V learning
            nn=snnhomeov(nn,opts);
            % (3) Synaptic modification with SBP
            nn=snnplasticitysbp(nn,opts,index,train_y);
            % Measurement for save performance
            [measures, y_predict,expected]=snn_build_measure(nn,index,train_y,opts);
            measurelist_train(index,:) = measures;
            % Plot recording
            if opts.plot
                disp(strcat('training epoch:',num2str(epoch),'/',num2str(numepochs),', batch:',num2str(index),'/',num2str(numbatches_train),', layer2 spike num:',num2str(nn.spikescount{2})));
            end
        end
        % Mean performance
        measuremean_train = mean(measurelist_train,1);
        % Plot performance
        fprintf('%2i-train-End E=%5f C=%5f error=%5f\n',epoch,measuremean_train(1),measuremean_train(2),measuremean_train(3));
        % Save performance
        nn.error_train(epoch)=measuremean_train(3);
        
       % Test of SNNs in each epoch
        measurelist_test = zeros(numbatches_test,3);
        % Batch for test dataset
        for index=1:numbatches_test
            % Feedforward propagation
            for ti=dt:dt:T
                nn=snnff(nn,index,test_x,opts,ti);
            end
            % Input layer clamp
            nn.layers{1}=test_x((index-1)*batchsize+1:index*batchsize,:);
            % Homeo-V
            nn=snnhomeov(nn,opts);
            % Save performance
            [measures, y_predict,expected]=snn_build_measure(nn,index,test_y,opts);
            nn.error_test(epoch)=measures(3);
            measurelist_test(index,:) = measures;
        end
        % Mean performance
        measuremean_test = mean(measurelist_test,1);
        fprintf('%2i-test-End E=%1f C=%5f error=%3f\n',epoch,measuremean_test(1),measuremean_test(2),measuremean_test(3));
        nn.error_test(epoch)=measuremean_test(3);
        % Save results
        fprintf(opts.fid,'train_E \t %4f \t train_C \t %4f \t train_err \t %4f \t test_E \t %4f \t test_C \t %4f \t test_err \t %4f \n',...
            measuremean_train(1),measuremean_train(2),measuremean_train(3),measuremean_test(1),measuremean_test(2),measuremean_test(3));
     end  
end