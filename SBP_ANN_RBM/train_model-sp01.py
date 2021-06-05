from model import Network
import numpy as np
import sys
from sys import stdout
import time
import pdb
import argparse

log = True
def train_net(net):
    
    modelpath         = net.modelpath
    input_sizes  = net.hyperparameters["input_sizes"]
    hidden_sizes = net.hyperparameters["hidden_sizes"]
    output_sizes = net.hyperparameters["output_sizes"]
    n_epochs     = net.hyperparameters["n_epochs"]
    batch_size   = net.hyperparameters["batch_size"]
    n_it_neg     = net.hyperparameters["n_it_neg"]
    n_it_pos     = net.hyperparameters["n_it_pos"]
    epsilon      = net.hyperparameters["epsilon"]
    beta         = net.hyperparameters["beta"]
    alphas       = net.hyperparameters["alphas"]
    train_samples = net.hyperparameters["train_samples"]
    valid_samples = net.hyperparameters["valid_samples"]

    print("name = %s" % (modelpath))
    print("architecture = "+ str(input_sizes) + '-' + str(hidden_sizes) + '-' + str(output_sizes))
    print("number of epochs = %i" % (n_epochs))
    print("batch_size = %i" % (batch_size))
    print("n_it_neg = %i"   % (n_it_neg))
    print("n_it_pos = %i"   % (n_it_pos))
    print("epsilon = %.1f" % (epsilon))
    print("beta = %.1f" % (beta))
    print("learning rates: "+" ".join(["alpha_W%i=%.3f" % (k+1,alpha) for k,alpha in enumerate(alphas)])+"\n")
    print("training samples = %d" % (train_samples))
    print("valid samples = %d" % (valid_samples))

    # net.save_logs()

    n_batches_train = int(train_samples / batch_size)
    n_batches_valid = int(valid_samples / batch_size)

    start_time = time.time()

    for epoch in range(n_epochs):

        ### TRAINING ###

        # CUMULATIVE SUM OF TRAINING ENERGY, TRAINING COST AND TRAINING ERROR
        measures_sum = [0.,0.,0.]
        gW = [0.] * len(alphas)

        for index in range(n_batches_train):

            # CHANGE THE INDEX OF THE MINI BATCH (= CLAMP X AND INITIALIZE THE HIDDEN AND OUTPUT LAYERS WITH THE PERSISTENT PARTICLES)
            net.change_mini_batch_index(index)
            net.sbp_phase(n_it_neg, epsilon)
            # pdb.set_trace()
            measures = net.measure()
            measures_sum = [measure_sum + measure for measure_sum,measure in zip(measures_sum,measures)]
            measures_avg = [measure_sum / (index+1) for measure_sum in measures_sum]
            measures_avg[-1] *= 100. # measures_avg[-1] corresponds to the error rate, which we want in percentage
            if log:
                stdout.write("\r%2i-train-%5i E=%.1f C=%.5f error=%.4f" % (epoch, (index+1)*batch_size, measures_avg[0], measures_avg[1], measures_avg[2]))
                stdout.flush()

            # ABP PHASE
            sign = 2*np.random.randint(0,2)-1 # random sign +1 or -1
            beta = np.float32(sign*beta) # choose the sign of beta at random

            # pdb.set_trace()
            Delta_logW = net.abp_phase(n_it_pos, epsilon, beta, *alphas)
            gW = [gW1 + Delta_logW1 for gW1,Delta_logW1 in zip(gW,Delta_logW)]
            # print(" ".join("gw="))


        if log: stdout.write("\n")
        dlogW = [100. * gW1 / n_batches_train for gW1 in gW]
        print("   "+" ".join(["dlogW%i=%.3f%%" % (k+1,dlogW1) for k,dlogW1 in enumerate(dlogW)]))
        # pdb.set_trace()
        net.training_curves["training error"].append(measures_avg[-1])

        ### VALIDATION ###
        
        # CUMULATIVE SUM OF VALIDATION ENERGY, VALIDATION COST AND VALIDATION ERROR
        measures_sum = [0.,0.,0.]

        for index in range(n_batches_valid):

            # CHANGE THE INDEX OF THE MINI BATCH (= CLAMP X AND INITIALIZE THE HIDDEN AND OUTPUT LAYERS WITH THE PERSISTENT PARTICLES)
            net.change_mini_batch_index(n_batches_train+index)

            # FREE PHASE
            net.sbp_phase(n_it_neg, epsilon)
            
            # MEASURE THE ENERGY, COST AND ERROR AT THE END OF THE FREE PHASE RELAXATION
            measures = net.measure()
            measures_sum = [measure_sum + measure for measure_sum,measure in zip(measures_sum,measures)]
            measures_avg = [measure_sum / (index+1) for measure_sum in measures_sum]
            measures_avg[-1] *= 100. # measures_avg[-1] corresponds to the error rate, which we want in percentage
            if log:
                stdout.write("\r   valid-%5i E=%.1f C=%.5f error=%.4f" % ((index+1)*batch_size, measures_avg[0], measures_avg[1], measures_avg[2]))
                stdout.flush()


        if log: stdout.write("\n")
        # print('test error = ' + str(measures_avg[-1]))
        net.training_curves["validation error"].append(measures_avg[-1])

        duration = (time.time() - start_time) / 60.
        # print("   duration=%.1f min" % (duration))

        # SAVE THE PARAMETERS OF THE NETWORK AT THE END OF THE EPOCH
    
    net.save_params()
    net.save_logs()
    print("training error output:")
    print(net.training_curves["training error"])
    print("validation error output:")
    print(net.training_curves["validation error"])

# # HYPERPARAMETERS FOR A NETWORK WITH 1 HIDDEN LAYER
# nowtime = time.strftime('-%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
# net1 = "net1"+str(nowtime), {
# "hidden_sizes" : [100],
# "n_epochs"     : 25, #25
# "batch_size"   : 20,
# "n_it_neg"     : 20, # 20
# "n_it_pos"     : 4, #4
# "epsilon"      : np.float32(.5), #.5
# "beta"         : np.float32(.5), #.5
# "alphas"       : [np.float32(.1), np.float32(.05)],
# "sbp"          : True,
# "theta_sbp"     : 0.25
# }

# # HYPERPARAMETERS FOR A NETWORK WITH 2 HIDDEN LAYERS
# net2 = "net2", {
# "hidden_sizes" : [500,500],
# "n_epochs"     : 60,
# "batch_size"   : 20,
# "n_it_neg"     : 150,
# "n_it_pos"     : 6,
# "epsilon"      : np.float32(.5),
# "beta"         : np.float32(1.),
# "alphas"       : [np.float32(.4), np.float32(.1), np.float32(.01)]
# }

# # HYPERPARAMETERS FOR A NETWORK WITH 3 HIDDEN LAYERS
# net3 = "net3", {
# "hidden_sizes" : [500,500,500],
# "n_epochs"     : 500,
# "batch_size"   : 20,
# "n_it_neg"     : 500,
# "n_it_pos"     : 8,
# "epsilon"      : np.float32(.5),
# "beta"         : np.float32(1.),
# "alphas"       : [np.float32(.128), np.float32(.032), np.float32(.008), np.float32(.002)]
# }


def train_mnist(sleep,wake,sp):
    # TRAIN A NETWORK WITH 1 HIDDEN LAYER
    # MNIST
    # for theta in range(21):
    nowtime = time.strftime('-%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
    # theta_sbp = theta*5.0/100.0
    theta_sbp = 0.5
    tasktype = 'mnist'
    netT = "net-"+str(theta_sbp)+str(nowtime), {
    "model_path"    : "model_"+str(nowtime)+"_"+tasktype+".save",
    "log_path"      : "log_"+str(nowtime)+"_"+tasktype+".txt",
    "tasktype"     : tasktype,
    "input_sizes"  : 784,
    "hidden_sizes" : [500],
    "output_sizes" : 10,
    "n_epochs"     : 10, #25
    "batch_size"   : 20,
    "n_it_neg"     : sleep, # 20
    "n_it_pos"     : wake, #4
    "epsilon"      : np.float32(1), #.5
    "beta"         : np.float32(0.2), #.5
    "alphas"       : [np.float32(.1), np.float32(.05)],
    "sbp"          : sp,
    "theta_sbp"     : theta_sbp,
    "train_samples"     : 50000,
    "valid_samples"     : 10000
    }
    train_net(Network(*netT)) 


def train_nettalk(sleep,wake,sp):
    nowtime = time.strftime('-%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
    theta_sbp = 0.5
    tasktype='nettalk'
    netT = "net-"+str(theta_sbp)+str(nowtime), {
    "model_path"    : "model_"+str(nowtime)+"_"+tasktype+".save",
    "log_path"      : "log_"+str(nowtime)+"_"+tasktype+".txt",
    "tasktype"     : tasktype,
    "input_sizes"   : 189,
    "hidden_sizes"  : [500],
    "output_sizes"  : 26,
    "n_epochs"      : 10, #25
    "batch_size"    : 1,
    "n_it_neg"      : sleep, # 20
    "n_it_pos"      : wake, #4
    "epsilon"       : np.float32(1), #.5
    "beta"          : np.float32(1), #.5
    "alphas"        : [np.float32(.1), np.float32(.05)],
    "sbp"           : sp,
    "theta_sbp"     : theta_sbp,
    "train_samples"     : 5033,
    "valid_samples"     : 500
    }
    train_net(Network(*netT))

def train_gesture(sleep,wake,sp):
    # TRAIN A NETWORK WITH 1 HIDDEN LAYER
    nowtime = time.strftime('-%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
    theta_sbp = 0.5
    tasktype = "gesture"
    netT = "net-"+str(theta_sbp)+str(nowtime), {
    "model_path"    : "model_"+str(nowtime)+"_"+tasktype+".save",
    "log_path"      : "log_"+str(nowtime)+"_"+tasktype+".txt",
    "tasktype"     : tasktype,
    "input_sizes"  : 102400,
    "hidden_sizes" : [500],
    "output_sizes" : 11,
    "n_epochs"     : 10, #25
    "batch_size"   : 1,
    "n_it_neg"     : sleep, # 20
    "n_it_pos"     : wake, #4
    "epsilon"      : np.float32(1), #.5
    "beta"         : np.float32(0.2), #.5
    "alphas"       : [np.float32(.1), np.float32(.05)],
    "sbp"          : sp,
    "theta_sbp"     : theta_sbp,
    "train_samples"     : 1752, # 1176+288+288
    "valid_samples"     : 288
    }
    train_net(Network(*netT)) 



parser = argparse.ArgumentParser()
parser.add_argument("--taskid", type=int, default=0) # 1 is mnist, 2 is nettalk, 3 is gesture
parser.add_argument("--sleep", type=int, default=10)
parser.add_argument("--wake", type=int, default=10)
parser.add_argument("--sp", type=int, default=0)
args = parser.parse_args()
taskid = args.taskid
sleep = args.sleep
wake = args.wake
sp = args.sp

time1 = time.time()
if taskid==1:
    train_mnist(sleep,wake,sp)
elif taskid==2:
    train_nettalk(sleep,wake,sp)
elif taskid==3:
    train_gesture(sleep,wake,sp)
elif taskid==4:
    for sleep in range(5):
        for wake in range(5):
            train_mnist(sleep*5+1,wake*5+1,sp)
else:
    print('no task')

print("time cost is : %f" %(time.time()-time1))
