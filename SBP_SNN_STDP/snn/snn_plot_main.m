%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Main code for SNN plot spike trains
%%% Input: load saved variables containing opts, snn, index of neurons, recording matrixs, epoch
%%% Output: Save Figs
%%% Revision: 2020-5-28
%%% Author: Tielin Zhang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('save_spikes_start_1');
neuronID =100;
% plot 50%
record2_mem = record_inputs(:,200:299);
record1_spikes = record1_spikes(:,200:299);
record2_spikes = record2_spikes(:,200:299);
%record2_spikes = record2_mem>0.05;
record3_spikes = record3_spikes(:,200:299);
target_spikes = target_spikes(:,200:299);
snn_plot_spike(opts, neuronID, record2_mem, record1_spikes,record2_spikes,record3_spikes,target_spikes, 1);


