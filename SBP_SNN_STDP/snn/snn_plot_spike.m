%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Only for SNN analysis, plot spike trains
%%% Input: opts, index of neurons, recording matrixs, epoch
%%% Output: Save figs in docs
%%% Revision: 2020-5-28
%%% Author: Tielin Zhang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [] = snn_plot_spike(opts, neuronID, record2_mem, record1_spikes,record2_spikes,record3_spikes,targetspike, epoch)
    
    % Load save path
    save_path = opts.fname_figpath;
    % Time steps
    timestep = size(record2_mem,2);
    % Initialization figure
    h=figure('Position',[100,100,1024,1200]);
    subplot(4,1,1);
    layer1spike=record1_spikes'.*repmat(1:784,timestep,1);
    layer1spike(layer1spike==0)=nan;
    % Eliminate the non-spike points in final figure
    %layer1spike(layer1spike==0)=nan;
    plot(layer1spike,'k.','MarkerSize',1); title('The spikes in the input layer');
    xlabel('Time (ms)'); ylabel('Neuronal index');
    ylim([-100,900]);
    xlim([1,timestep+1]);
    hold on;
    subplot(4,1,2);
    layer2spike=record2_spikes'.*repmat(1:500,timestep,1);
    % Eliminate the non-spike points in final figure
    %layer2spike(layer2spike==0)=nan;
    layer2spike(layer2spike==0)=nan;
    plot(layer2spike,'k.','MarkerSize',1); title('The spikes in the hidden layer');
    xlabel('Time (ms)'); ylabel('Neuronal index');
    ylim([-100,600]);
    xlim([1,timestep+1]);
    hold on;
    subplot(4,1,3);
    layer2statespike = record2_spikes(neuronID,:) + record2_mem(neuronID,:);
    layer2statespike(layer2statespike==0)=nan;
    % Eliminate the non-spike points in final figure
    plot(layer2statespike,'k-','MarkerSize',5); title('The state of 100th neuron in the hidden layer');
    xlabel('Time (ms)'); ylabel('Neuronal state (mV)');
    %ylim([-100,600]);
    xlim([1,timestep+1]);
    hold on;
    subplot(4,1,4);
    layer3spike=record3_spikes'.*repmat(1:10,timestep,1);
    layer3spike(layer3spike==0)=nan;
    % Output signal
    p1 = plot(layer3spike,'k.','MarkerSize',5); title('The spikes in the output layer');
    hold on;
    % Target signal
    outtargetspike=targetspike'.*repmat(1:10,timestep,1);
    outtargetspike(outtargetspike==0)=nan;
    p2 = plot(outtargetspike,'r*','MarkerSize',5);
    hold off;
    legend([p1(1) p2(1)], {'Output spike trains','Teaching spike trains'});
    xlabel('Time (ms)'); ylabel('Neuronal index');
    ylim([0,11]);
    xlim([1,timestep+1]);
    hold on;
    
    % Check path
    if ~exist(save_path,'dir')
        mkdir(save_path);
    end
    % Save figures
    savefig(h,strcat(save_path,'train_',num2str(epoch),'_layer2'));
    saveas(gcf,strcat(save_path,'train_',num2str(epoch),'_layer2','.jpg'));
    %close(h);
end

