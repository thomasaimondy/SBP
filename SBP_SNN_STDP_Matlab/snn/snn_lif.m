%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% LIF neuron in SNN
%%% Input: snn, layer, inputs, opts
%%% Output: snn
%%% Revision: 2020-5-28
%%% Author: Tielin Zhang
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function nn = snn_lif(nn,layer,inputs,opts)
    % Variables initialization
    batchsize = opts.batchsize;
    % Set configuration of LIF
    V_E = opts.V_E;
    V_I = opts.V_I;
    V_leak = opts.V_leak; % mV
    V_th = opts.V_th; % mV
    taum = opts.taum; % ms
    tau_syn_E = opts.tau_syn_E; % ms
    tau_syn_I = opts.tau_syn_I; % ms
    g_leak_E = opts.g_leak_E; % nS
    g_leak_I = opts.g_leak_I; % nS
    % Time slot
    dt = opts.dt;
    % E/I neurons
    n_neuronE = numel(nn.index_positive{layer});
    n_neuronI = numel(nn.index_negative{layer});
    % V
    volt = nn.lifvolt{layer};
    % g
    gE = nn.lifge{layer};
    % g_E
    gE_E = gE(:,nn.index_positive{layer});
    % v_E
    volt_E = volt(:,nn.index_positive{layer});
    % g_I
    gE_I = gE(:,nn.index_negative{layer});
    % v_I
    volt_I = volt(:,nn.index_negative{layer});
    % STP code from Luozheng Li in Beijing Normal University
    if opts.setSTP
        u = nn.lifu{layer};
        x = nn.lifx{layer};
        I = nn.lifI{layer};
        U = opts.U;
        tauf = opts.tauf;
        taud = opts.taud;
        taus = opts.taus;
        A = 1;
        u = u + ( - u/tauf + U * inputs .* (1-u));
        x = x + ((1-x) / taud - u .* x .* inputs);
        I = I + ( - I/taus + A*u.*x.*inputs);
        inputs2 = I;
    else
        inputs2 = inputs;
    end
    % Update E voltage
    gE_E = gE_E + dt/tau_syn_E*(-gE_E + g_leak_E + inputs2(:,nn.index_positive{layer}));
    volt_E = volt_E + dt/taum*(-(volt_E - V_leak*ones(batchsize,n_neuronE)) - gE_E/g_leak_E.*(volt_E-V_E*ones(batchsize,n_neuronE)));
    % Update I voltage
    gE_I = gE_I + dt/tau_syn_I*(-gE_I + g_leak_I + inputs2(:,nn.index_negative{layer}));
    volt_I = volt_I + dt/taum*(-(volt_I - V_leak*ones(batchsize,n_neuronI)) - gE_I/g_leak_I.*(volt_I-V_I*ones(batchsize,n_neuronI)));
    % Integration
    gE(:,nn.index_positive{layer}) = gE_E;
    gE(:,nn.index_negative{layer}) = gE_I;
    volt(:,nn.index_positive{layer}) = volt_E;
    volt(:,nn.index_negative{layer}) = volt_I;
    % Spikes reset
    spiking = volt > V_th;
    volt(spiking) = opts.reset;
    % Save back
    nn.lifvolt{layer} = volt;
    nn.lifge{layer} = gE;
    %nn.lifspike{layer} = spiking;
    nn.spikescount{layer} = sum(sum(spiking));
end

