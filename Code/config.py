# config.py

# ==============================================================================
#                               Data Settings
# ==============================================================================
# General data settings
DATASET = 'cifar100'  # Dataset to use: 'cifar10', 'cifar100', 'shake'

# Client data distribution settings
N_CLIENTS = 100  # Total number of clients (N)
SAMPLES_PER_CLIENT = 400  # Number of training samples per client
IS_IID = False  # True for IID client data distribution, False for non-IID
ALPHA = 0.5  # Dirichlet distribution parameter for client non-IID degree (α)

# Server data distribution settings
SERVER_IID = False  # True for IID server data, False for non-IID
BETA = 0.5  # Dirichlet distribution parameter for server non-IID degree (β)
SERVER_DATA_RATIO = 0.1  # Ratio of the total training data to be used by the server
SERVER_FILL = True  # Whether to fill up the server dataset to the specified ratio if sampling falls short

# ==============================================================================
#                               Model Settings
# ==============================================================================
# Model architecture and training settings
ORIGIN_MODEL = 'vgg'  # Model to use: 'resnet', 'vgg', 'lstm'
NUM_CLASSES = 20  # Number of classes in the dataset (10 for CIFAR-10, 20 for CIFAR-100 coarse, 80 for Shakespeare)
MOMENTUM = 0.5  # Momentum for the SGD optimizer
WEIGHT_DECAY = 0  # Weight decay for the optimizer
BC_SIZE = 50  # Batch size for local training
TEST_BC_SIZE = 128  # Batch size for testing

# ==============================================================================
#                           Federated Learning Settings
# ==============================================================================
# General FL hyperparameters
GLOBAL_RANDOM_FIX = True  # Whether to fix the randomness for the entire training process
SEED = 2  # Global random seed
GPU = 0  # GPU device ID to use
VERBOSE = False  # Set to True to print intermediate debugging information

# Federated training round settings
T_ROUNDS = 100  # Total number of global training rounds (T)
M_CLIENTS = 10  # Number of clients sampled in each round (M)
K_EPOCHS = 5  # Number of local training epochs on clients (K)
E_EPOCHS = 1  # Number of local training epochs on the server (E)

# Learning rates
ETA = 0.01  # Client-side learning rate (η)
ETA_0 = 0.01  # Server-side learning rate (η₀)

# ==============================================================================
#                         Hyperparameters
# ==============================================================================
# FedAT (Adaptive Server Training) module
MU = 5  # Scaling constant for the adaptive factor (μ)

# FedMV (Model Variation) module
RHO = 4.0  # Base magnitude for model variation (ρ)
THETA = 0.3  # Scaling ratio for adapting variation magnitude (θ)

# FedDU baseline hyperparameter
DECAY_RATE = 0.99 # Decay rate for the alpha calculation in FedDU

# FedMut baseline hyperparameters
MUT_ACC_RATE = 0.5  # Initial mutation acceptance rate (β₀) for FedMut
MUT_BOUND = 50  # Round threshold (Tb) for decaying mutation rate in FedMut