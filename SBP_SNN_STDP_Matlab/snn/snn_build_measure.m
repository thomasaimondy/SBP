%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Calculate errors of SNNs
%%% Input: snn, datasets (train_x, train_y, test_x, test_y), opts
%%% Output: snn
%%% Revision: 2020-5-28
%%% Author: Tielin Zhang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [measure, y_predict,expected] =snn_build_measure(nn,index,train_y,opts)
    % Batchsize
    batchsize=opts.batchsize;
    % Measure initialization
    measure=zeros(1,3);
    % Calculate Hebb energy
    E=mean(snn_hebb_energy(nn));
    % SNN output y
    y=train_y((index-1)*batchsize+1:index*batchsize,:);
    % Calculate MSE
    C=mean(snn_mse_cost(nn,y));
    % Select task
    % TIPS: different calculation for different tasks
    switch opts.task
        % MNIST, with one-hot output
        case 'mnist'
            % Get y
            [~,y_predict]=max(nn.layers{end},[],2);
            % Get y^
            [~,expected]=max(y,[],2);
            % Misclassified samples
            bad=find(y_predict~=expected);
            % Calculate error
            error=numel(bad)/size(y,1);
        % Nettalk, with multi-targets
        case 'nettalk'
            % Get y
            y_predict = nn.layers{end};
            % Error list initialization
            errors=[];
            % Each sample
            for i =1:1:size(y,1)
                % Similarity of y and y^
                Composite_matrix = [y_predict(i,:);y(i,:)];
                % Error
                errors(i) = pdist(Composite_matrix,'cosine');
            end
            % Mean error omot Nan
            error = mean(errors,'omitnan');
            % Give output
            expected = y;
        % DvsGesture, one-hot output
        case 'guesture'
            % Get y
            [~,y_predict]=max(nn.layers{end},[],2);
            % Get y^
            [~,expected]=max(y,[],2);
            % Misclassified samples
            bad=find(y_predict~=expected);
            % Calculate error
            error=numel(bad)/size(y,1);
        otherwise
            fprintf('error in opts.task');  
            
    end
    % Return HebbEnergy, Cost, and Error to main
    measure(1)=E;
    measure(2)=C;
    measure(3)=error;
end