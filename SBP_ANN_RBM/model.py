import pickle as cpickle
from external_world import External_World
import numpy as np
import os
import theano
import theano.tensor as T
import pdb
import xlwt

def rho(s):
    return T.clip(s,0.,1.)
    #return T.nnet.sigmoid(4.*s-2.)

class Network(object):

    def __init__(self, name, hyperparameters=dict()):

        # LOAD/INITIALIZE PARAMETERS
        self.modelpath = "model/" + hyperparameters['model_path']
        self.log_path = "log/" + hyperparameters['log_path']
        self.biases, self.weights, self.hyperparameters, self.training_curves = self.__load_params(hyperparameters)
        # pdb.set_trace()

        # LOAD EXTERNAL WORLD (=DATA)
        self.external_world = External_World(self.hyperparameters['tasktype'])

        # INITIALIZE PERSISTENT PARTICLES
        dataset_size = self.external_world.size_dataset
        layer_sizes = [self.hyperparameters["input_sizes"]] + self.hyperparameters["hidden_sizes"] + [self.hyperparameters["output_sizes"]]
        values = [np.zeros((dataset_size, layer_size), dtype=theano.config.floatX) for layer_size in layer_sizes[1:]]
        self.persistent_particles  = [theano.shared(value, borrow=True) for value in values]

        # LAYERS = MINI-BACTHES OF DATA + MINI-BACTHES OF PERSISTENT PARTICLES
        batch_size = self.hyperparameters["batch_size"]
        self.index = theano.shared(np.int32(0), name='index') # index of a mini-batch

        self.x_data = self.external_world.x[self.index * batch_size: (self.index + 1) * batch_size]
        self.y_data = self.external_world.y[self.index * batch_size: (self.index + 1) * batch_size]
        self.y_data_onehot = self.external_world.y_onehot[self.index * batch_size: (self.index + 1) * batch_size]
        self.layers = [self.x_data]+[particle[self.index * batch_size: (self.index + 1) * batch_size] for particle in self.persistent_particles]

        # BUILD THEANO FUNCTIONS
        self.change_mini_batch_index = self.__build_change_mini_batch_index()
        self.measure                 = self.__build_measure()
        self.sbp_phase              = self.__build_sbp_phase()
        self.abp_phase              = self.__build_abp_phase()

    def save_params(self):
        f = open(self.modelpath, 'wb')
        biases_values  = [b.get_value() for b in self.biases]
        weights_values = [W.get_value() for W in self.weights]
        to_dump        = biases_values, weights_values, self.hyperparameters, self.training_curves, self.hyperparameters
        cpickle.dump(to_dump, f, protocol=cpickle.HIGHEST_PROTOCOL)
        f.close()

    def save_logs(self):
        import json
        filename = open(self.log_path,'w')
        filename.write("hyperparameters:\n")

        filename.write(str(self.hyperparameters))
        filename.write("\n\n")

        filename.write("training error:\n")
        filename.write(str(self.training_curves["training error"]))

        filename.write("\n\n")
        filename.write("test error:\n")
        filename.write(str(self.training_curves["validation error"]))
        filename.close()



    def __load_params(self, hyperparameters):

        hyper = hyperparameters

        # Glorot/Bengio weight initialization
        def initialize_layer(n_in, n_out):
            rng = np.random.RandomState()
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            return W_values

        if os.path.isfile(self.modelpath):
            f = open(self.modelpath, 'rb')
            biases_values, weights_values, hyperparameters, training_curves = cpickle.load(f)
            f.close()
            for k,v in hyper.iteritems():
                hyperparameters[k]=v
        else:
            layer_sizes = [hyperparameters["input_sizes"]] + hyperparameters["hidden_sizes"] + [hyperparameters["output_sizes"]]
            biases_values  = [np.zeros((size,), dtype=theano.config.floatX) for size in layer_sizes]
            weights_values = [initialize_layer(size_pre,size_post) for size_pre,size_post in zip(layer_sizes[:-1],layer_sizes[1:])]
            training_curves = dict()
            training_curves["training error"]   = list()
            training_curves["validation error"] = list()

        biases  = [theano.shared(value=value, borrow=True) for value in biases_values]
        weights = [theano.shared(value=value, borrow=True) for value in weights_values]

        return biases, weights, hyperparameters, training_curves

    # SET INDEX OF THE MINI BATCH
    def __build_change_mini_batch_index(self):

        index_new = T.iscalar("index_new")

        change_mini_batch_index = theano.function(
            inputs=[index_new],
            outputs=[],
            updates=[(self.index,index_new)]
        )

        return change_mini_batch_index

    # ENERGY FUNCTION, DENOTED BY E
    def __sbp_cost(self, layers):

        if self.hyperparameters["sbp"]:
            #############################################################################################thomas add
            pre = layers[:-1]
            post = layers[1:]
            W = self.weights
            # What is bias here? we give it a new definition as the proportion of weights for propagation.
            bss = self.biases
            # SP(bp) part:
            sbp_bp_term = 0
            # pdb.set_trace()
            for i in range(len(pre)-1):
                # back propagation of LTP/LTD with network state in scalar2.
                theta_sbp = self.hyperparameters["theta_sbp"]
                scalar1 = T.batched_dot(T.dot(rho(pre[i]),W[i]),rho(post[i])) + T.batched_dot(rho(pre[i]),rho(pre[i]))
                scalar2 = T.batched_dot(T.dot(rho(pre[i+1]),W[i+1]),rho(post[i+1])) + T.batched_dot(rho(post[i]),rho(post[i]))
                sbp_bp_term = sbp_bp_term + (scalar1 + theta_sbp * scalar2)

            # SP(pre) part:(Attention, this procedure is only one sub part of the SP(pre), another part is in detW)
            sbp_pre_term = 0
            for i in range(len(pre)):
                # Pre-synaptic lateral propagation of LTP/LTD, connect pre layers state with pre layer weights
                sbp_pre_term   = sbp_pre_term + T.dot(rho(pre[i]),bss[i])

            # SP(post) part: (Atttention, this procedure is only one sub part of the SP(post), another part is in detW)
            sbp_post_term = 0
            for i in range(len(post)):
            #     # Post-synaptic lateral propagation of LTD (only), connect post lyers state with post layer weights
                sbp_post_term   = sbp_post_term + T.dot(rho(post[i]),bss[i+1])

            return sbp_bp_term + sbp_pre_term + sbp_post_term
        else:
            squared_norm    =   sum( [T.batched_dot(rho(layer),rho(layer))       for layer      in layers] ) / 2.
            linear_terms    = - sum( [T.dot(rho(layer),b)                        for layer,b    in zip(layers,self.biases)] )
            quadratic_terms = - sum( [T.batched_dot(T.dot(rho(pre),W),rho(post)) for pre,W,post in zip(layers[:-1],self.weights,layers[1:])] )
            return squared_norm + linear_terms + quadratic_terms
        ################################################################################################thomas add end

    # COST FUNCTION, DENOTED BY C
    def __abp_cost(self, layers):
        return ((layers[-1] - self.y_data_onehot) ** 2).sum(axis=1)

    # TOTAL ENERGY FUNCTION, DENOTED BY F
    def __total_cost(self, layers, beta):
        return self.__sbp_cost(layers) + beta * self.__abp_cost(layers)

    # MEASURES THE ENERGY, THE COST AND THE MISCLASSIFICATION ERROR FOR THE CURRENT STATE OF THE NETWORK
    def __build_measure(self):

        # pdb.set_trace()
        E = T.mean(self.__sbp_cost(self.layers))
        C = T.mean(self.__abp_cost(self.layers))
        if self.hyperparameters['tasktype'] is 'mnist':
            y_prediction = T.argmax(self.layers[-1], axis=1)
            error        = T.mean(T.neq(y_prediction, self.y_data))
        if self.hyperparameters['tasktype'] is 'nettalk':
            # y_prediction = T.argmax(self.layers[-1], axis=1)
            y_prediction = self.layers[-1]
            error        = T.mean(T.neq(y_prediction, self.y_data))
        if self.hyperparameters['tasktype'] is 'gesture':
            y_prediction = self.layers[-1]
            error        = T.mean(T.neq(y_prediction, self.y_data))
        measure = theano.function(
            inputs=[],
            outputs=[E, C, error]
        )

        return measure

    def __build_sbp_phase(self):

        n_iterations = T.iscalar('n_iterations')
        epsilon  = T.fscalar('epsilon')

        def step(*layers):
            E_sum = T.sum(self.__sbp_cost(layers))
            layers_dot = T.grad(-E_sum, list(layers)) # temporal derivative of the state (free trajectory)
            layers_new = [layers[0]]+[T.clip(layer+epsilon*dot,0.,1.) for layer,dot in zip(layers,layers_dot)][1:]
            return layers_new

        ( layers, updates ) = theano.scan(
            step,
            outputs_info=self.layers,
            n_steps=n_iterations
        )
        layers_end = [layer[-1] for layer in layers]

        for particles,layer,layer_end in zip(self.persistent_particles,self.layers[1:],layers_end[1:]):
            updates[particles] = T.set_subtensor(layer,layer_end)

        sbp_phase = theano.function(
            inputs=[n_iterations,epsilon],
            outputs=[],
            updates=updates
        )

        return sbp_phase

    def __build_abp_phase(self):

        n_iterations = T.iscalar('n_iterations')
        epsilon  = T.fscalar('epsilon')
        beta = T.fscalar('beta')
        alphas = [T.fscalar("alpha_W"+str(r+1)) for r in range(len(self.weights))]

        def step(*layers):
            # test 1 Thomas
            F_sum = T.sum(self.__total_cost(layers, beta)) # giving the supervised labels
            # F_sum = T.sum(self.__sbp_cost(layers))
            layers_dot = T.grad(-F_sum, list(layers))
            layers_new = [layers[0]]+[T.clip(layer+epsilon*dot,0.,1.) for layer,dot in zip(layers,layers_dot)][1:]
            return layers_new

        ( layers, updates ) = theano.scan(
            step,
            outputs_info=self.layers,
            n_steps=n_iterations
        )

        # knowledge consolidation from layers into synaptic weights

        layers_weakly_clamped = [layer[-1] for layer in layers]
        E_mean_sbp           = T.mean(self.__sbp_cost(self.layers))
        E_mean_abp             = T.mean(self.__sbp_cost(layers_weakly_clamped))
        # E_mean_abp             = T.mean(self.__abp_cost(layers_weakly_clamped))
        # E_mean_abp             = T.mean(self.__abp_cost(self.layers))
        biases_dot            = T.grad( (E_mean_abp-E_mean_sbp) / beta, self.biases,  consider_constant=layers_weakly_clamped)
        weights_dot           = T.grad( (E_mean_abp-E_mean_sbp) / beta, self.weights, consider_constant=layers_weakly_clamped)

        # the calculation of SP(pre) and SP(post)
        # pdb.set_trace()

        biases_new  = [b - alpha * dot for b,alpha,dot in zip(self.biases[1:],alphas,biases_dot[1:])]
        weights_new = [W - alpha * dot for W,alpha,dot in zip(self.weights,   alphas,weights_dot)]

        Delta_log = [T.sqrt( ((W_new - W) ** 2).mean() ) / T.sqrt( (W ** 2).mean() ) for W,W_new in zip(self.weights,weights_new)]

        for bias, bias_new in zip(self.biases[1:],biases_new):
            updates[bias]=bias_new
        for weight, weight_new in zip(self.weights,weights_new):
            updates[weight]=weight_new

        abp_phase = theano.function(
            inputs=[n_iterations, epsilon, beta]+alphas,
            outputs=Delta_log,
            updates=updates
        )

        return abp_phase
