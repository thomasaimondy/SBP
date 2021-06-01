function [  ] = lif_demo()
%LIF Biology inspired Leaky and Integrated and Fire model

%% Inition Variables
n_neuron = 2;
sy_in = 1;
sy_out = 10;
V_E = 0.4;
V_I = 0;
V_leak = 0; % mV
V_th = 0.2; % mV
V_reset = 0; % mV
taum = 20; % ms
tau_syn_E = 2; % ms
%tau_syn_I = 5; % ms
g_leak = 20; % nS
dt = 0.1; % Integration  dt [ms]
T =100;  % ms

%% Inition Inputs
Iinj = zeros(sy_in,round(T/dt));
for ti = round(T/dt/3):round(T/dt/3*2)
    Iinj(:,ti )= 10*ones(sy_in,1);
end
%z = zeros(size(Iinj));
%ind = np.where(I>(V_th-vl)/(R/A))[0];
%z[ind] = 1000.*(tref + R*C*np.log((R/A*I[ind] + vl - vres)/(R/A*I[ind] + vl - vth)))**-1
%Xin = exp((t_p-t)/tau_m);
%Ai = exp((t_q-t)/tau_m);
%Vmp = sum(W*Xin)-Vth*Ai+Vmp;

%% Inition neurons
volt = V_leak*ones(n_neuron,1);
%spike_in = zeros(sy_in,1);
%spiking = zeros(n_neuron,1);
gE = ones(n_neuron,1)*0.1;

record_spikes = zeros(n_neuron,round(T/dt));
record_vmem = zeros(n_neuron,round(T/dt));
record_gE = zeros(n_neuron,round(T/dt));
%% Inition Weights
W = ones(n_neuron,sy_in)*0.5;

%% calculating
for ti = dt:dt:T
	spike_in = Iinj(:,round(ti/dt));
	% Update voltage
	gE = gE -gE*dt/tau_syn_E + W*spike_in;
    volt = volt -dt/taum*(volt - V_leak*ones(n_neuron,1) + gE/g_leak.*(volt-V_E*ones(n_neuron,1)));
    % Spikes reset
    spiking = volt > V_th;
    volt(spiking) = V_reset;
    record_spikes(:,round(ti/dt)) = spiking;
    record_vmem(:,round(ti/dt)) = volt;
    record_gE(:,round(ti/dt)) = gE;
end
%% ploting
subplot(4,1,1);
plot(dt:dt:T,Iinj(1,:));
hold on;    
subplot(4,1,2);
plot(dt:dt:T,record_vmem(1,:));
hold on;
subplot(4,1,3);
plot(dt:dt:T,record_spikes(1,:));
hold on;
subplot(4,1,4);
plot(dt:dt:T,record_gE(1,:));
end

