%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FOR SNN plasticity consolidation with SBP
%%% Input: snn, opts, batch_index, train_y
%%% Output: snn
%%% Revision: 2020-5-28
%%% Author: Tielin Zhang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function nn = snnplasticitysbp(nn,opts,index,train_y)
    % Number of layers
    n=nn.n;
    % Save layer state
    nn.layers_new{1}=nn.layers{1};
    % Load configuration of iteration
    n_itreations=opts.n_it_pos;
    % Load epsilon
    eps=opts.epsilon;
    % Load beta
    beta=opts.beta;
    % Load alpha
    alpha=opts.alpha;
    % Load batchsize
    batchsize=opts.batchsize;
    % Load batch train_y
    test_y=train_y((index-1)*batchsize+1:index*batchsize,:);
    % Temporary neuron state
    for i=1:n
        nn.layers_positive_tmp{i}=nn.layers{i};
    end
    % State update
    for num=1:n_itreations
        % Only hidden layers
        for i=2:n-1
            % Number of batch samples
            [m,~]=size(nn.layers_positive_tmp{i});
            % Homeo-V update
            detHomeo=nn.layers_positive_tmp{i}-(nn.layers_positive_tmp{i-1}*nn.W{i-1}+nn.layers_positive_tmp{i+1}*nn.W{i}')-repmat(nn.b{i},m,1);            
            nn.layers_positive_tmp{i}=nn.layers_positive_tmp{i}-eps*detHomeo;
            % Normalization
            % TIPS: Some modes will be non-convergence, e.g., ReLU
            nn.layers_positive_tmp{i}=snn_sigmoid(nn.layers_positive_tmp{i},opts,1);
        end
        % Number of batch samples in output layer
        [m,~]=size(nn.layers_positive_tmp{n});
        % Homeo-V and error
        % TIPS: error = MSE = (y-y^)^2, det_error = (y-y^)
        detV=nn.layers_positive_tmp{n}-nn.layers_positive_tmp{n-1}*nn.W{n-1}-repmat(nn.b{n},m,1)+beta*(nn.layers_positive_tmp{n}-test_y);
        % Update of layers
        nn.layers_positive_tmp{n}=nn.layers_positive_tmp{n}-eps*detV;
        % Normalization
        % TIPS: Some modes will be non-convergence, e.g., Sigmoid
        nn.layers_positive_tmp{n} = snn_sigmoid(nn.layers_positive_tmp{n},opts,5); 
    end
	% Update bias
    for i=2:n
        % Only for hidden and output layers
        detbias=mean((nn.layers_positive_tmp{i}-nn.layers{i})/beta);
        % Update bias
        nn.b{i}=nn.b{i}+alpha(i-1)*detbias;
    end
    % Update synaptic weights with SBP
    % SBP(bp) LTP
    W_post_layer_LTP = {};
    % SBP(bp) LTD
    W_post_layer_LTD = {};
    % Inverse calculation from output to input layers
    for m=nn.n:-1:2
        % Selct plasticity type
        switch opts.stdp_type
            % differential STDP
        	case 'differentialSTDP'
            	change_W=(nn.layers_positive_tmp{m-1}'*nn.layers{m}-nn.layers{m-1}'*nn.layers{m})/(beta*batchsize);
            % symmetric STDP
            case 'symmetricSTDP'
                change_W=(nn.layers_positive_tmp{m-1}'*nn.layers_positive_tmp{m}-nn.layers{m-1}'*nn.layers{m})/(beta*batchsize);
            % SBP
            case 'SBP'
                % Initial detW
                change_W=(nn.layers{m-1}'*nn.layers_positive_tmp{m}-nn.layers{m-1}'*nn.layers{m})/(beta*batchsize);
                % DetW separation with LTP
                detW_plus = change_W; detW_plus(detW_plus<0)=0;
                % DetW seperation with LTD
                detW_minus = change_W; detW_minus(detW_minus>0)=0;
                % (1) Pre lateral spread, SBP(pre), LTP
                T_ltp_pre = 1+opts.lamda_sp_pre * opts.fraction * sbpnonlinear(opts.propagation_range * sum(detW_plus,2));
                % SBP(pre)_LTP propagation
                W_ltp_pre = diag(T_ltp_pre,0) * detW_plus;
                % Save SBP(pre)_LTP
                W_post_layer_LTP{m} = W_ltp_pre;
                % (2) Pre lateral spread, SBP(pre),  LTD
                T_ltd_pre = 1+opts.lamda_sp_pre * opts.fraction * sbpnonlinear(opts.propagation_range * abs(sum(detW_minus,2)));
                % SBP(pre)_LTD propagation
                W_ltd_pre = diag(T_ltd_pre,0) * detW_minus;
                % Save SBP(pre)_LTD
                W_post_layer_LTD{m} = W_ltd_pre;
                % (3) Post lateral spread, SBP(post), LTD (only)
                T_ltd_post = 1+opts.lamda_sp_post * opts.fraction * sbpnonlinear(opts.propagation_range * abs(sum(detW_minus,1)));
                % SBP(post)_LTD_only propagation
                W_ltd_post = detW_minus * diag(T_ltd_post,0);
                % Not include the output layer
                if(m<nn.n)
                    % (4) Back propagation SBP(bp), LTP
                    T_ltp_bp = 1+opts.lamda_sp_bp * opts.fraction * sbpnonlinear(opts.propagation_range * sum(W_post_layer_LTP{m+1},2));
                    % SBP(bp)_LTP propagation
                    W_ltp_bp = detW_plus * diag(T_ltp_bp,0);
                    %(5) Back propagation SBP(bp), LTD
                    T_ltd_bp = 1+opts.lamda_sp_bp * opts.fraction * sbpnonlinear(opts.propagation_range * sum(W_post_layer_LTD{m+1},2));
                    % SBP(bp)_LTD propagation
                    W_ltd_bp = detW_minus * diag(T_ltd_bp,0);
                    % Delete nan
                    % TIPS: Some caused Nan should be deleted or will be
                    % non-convergence especially for multi-layer SNNs
                    W_ltp_pre(isnan(W_ltp_pre))=0;
                    W_ltd_pre(isnan(W_ltd_pre))=0;
                    W_ltd_post(isnan(W_ltd_post))=0;
                    W_ltp_bp(isnan(W_ltp_bp))=0;
                    W_ltd_bp(isnan(W_ltd_bp))=0;
                    % SBP integration (5 items)
                    % TIPS: Normalization
                    change_W = (W_ltp_pre + W_ltp_bp)/2 + (W_ltd_pre + W_ltd_post + W_ltd_bp)/3;
                else
                    % For output layer, no SBP(bp)
                    change_W = W_ltp_pre + (W_ltd_pre + W_ltd_post)/2;
                end
            otherwise
                  % Error configuration
                  fprintf('stdp setting error !!!!\n');
        end
        % Add background noise (selection)
        switch opts.noise_type
            case 'none'
                % No noise
                change_W = change_W;
            case 'norm'
                % Norm noise
                sigma = 1.01;
                change_W = sigma*rand(size(change_W)) .* change_W;
            case 'poisson'
                % Poisson noise
                poisson = poisspdf(0:opts.numepochs,1);
                change_W = change_W.*(1+poisson(opts.epoch)*0.1);
            otherwise
                % Error configuration
                fprintf('background noise setting error !!!!\n');
        end
        % Dale's law
        if opts.set_EI
            % Only constrain output synapses of neurons
            if m>2
                % Each E/I neuron
                for j = 1:nn.size(m-1)
                    % Change of W will not change E/I type
                    alpha_change = snn_dalelaw(nn.W{m-1}(j,:),change_W(j,:),alpha(m-1),opts);
                    % Update of E/I weight
                    nn.W{m-1}(j,:) = nn.W{m-1}(j,:) + alpha_change.*change_W(j,:);
                end
            else
                % No Dale's law
                nn.W{m-1}=nn.W{m-1}+alpha(m-1)*change_W;
            end
        else
            % No Dale's law
            nn.W{m-1}=nn.W{m-1}+alpha(m-1)*change_W;
        end
    end

end

