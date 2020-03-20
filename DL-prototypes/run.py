import argparse
from src.train import train_MNIST
from src.helper import str2bool

# Global parameters for device and reproducibility
PARSER = argparse.ArgumentParser()
PARSER.add_argument('--seed', type=int, default=42,
                    help='seed for reproduction')
PARSER.add_argument("--hier", type=str2bool, nargs='?', const=True, default=True,
                    help='Hierarchical mode turned on')
PARSER.add_argument("--n_prototypes", type=int, default=10,
                    help='Number of prototypes/superprototypes')
PARSER.add_argument("--n_sub_prototypes", type=int, default=20,
                    help='Number of sub-prototypes')

PARSER.add_argument("--lambda_class", type=int, default=20,
                    help='lambda value for standard classification loss')
PARSER.add_argument("--lambda_class_sub", type=int, default=20,
                    help='lambda value for sub-prototype layer classification loss')
PARSER.add_argument("--lambda_class_sup", type=int, default=20,
                    help='lambda value for super-prototype layer classification loss')
PARSER.add_argument("--lambda_ae", type=int, default=1,
                    help='lambda value for reconstruction loss')
PARSER.add_argument("--lambda_r1", type=int, default=1,
                    help='lambda value for r1 loss')
PARSER.add_argument("--lambda_r2", type=int, default=1,
                    help='lambda value for r2 loss')
PARSER.add_argument("--lambda_r3", type=int, default=1,
                    help='lambda value for r3 loss')
PARSER.add_argument("--lambda_r4", type=int, default=1,
                    help='lambda value for r4 loss')

PARSER.add_argument("--learning_rate", type=int, default=0.0001,
                    help='Training learning rate')
PARSER.add_argument("--n_epochs", type=int, default=900,
                    help='Number of epochs for training')
PARSER.add_argument("--batch_size", type=int, default=250,
                    help='Training batch size')
PARSER.add_argument("--save_every", type=int, default=1,
                    help='Save the results after every x epoch')

PARSER.add_argument("--sigma", type=int, default=4, 
                    help='Standard deviation for the random elastic distortions')
PARSER.add_argument("--alpha", type=int, default=20, 
                    help='Scaling factor for the random elastic distortions')

PARSER.add_argument("--underrepresented_class", type=int, default=-1,
                    help='Class number representing the underrepresented, downsampled class. \
                        When set to -1, all classes will be represented according to the original distribution.')
PARSER.add_argument('--dir', type=str, default='my_own_model',
                    help='main directory to save intermediate results')
ARGS = PARSER.parse_args()

LAMBDA_DICT = {
    'lambda_class' : ARGS.lambda_class,
    'lambda_class_sup' : ARGS.lambda_class_sup,
    'lambda_class_sub' : ARGS.lambda_class_sub,
    'lambda_ae' : ARGS.lambda_ae,
    'lambda_r1' : ARGS.lambda_r1,
    'lambda_r2' : ARGS.lambda_r2,
    'lambda_r3' : ARGS.lambda_r3,
    'lambda_r4' : ARGS.lambda_r4
}

train_MNIST(
    hierarchical=ARGS.hier,
    n_prototypes=ARGS.n_prototypes,
    n_sub_prototypes=ARGS.n_sub_prototypes,
    latent_size=40,
    n_classes=10,
    lambda_dict=LAMBDA_DICT,
    learning_rate=ARGS.learning_rate,
    training_epochs=ARGS.n_epochs,
    batch_size=ARGS.batch_size,
    save_every=ARGS.save_every,
    sigma=ARGS.sigma,
    alpha=ARGS.alpha,
    seed=ARGS.seed,
    directory=ARGS.dir,
    underrepresented_class=ARGS.underrepresented_class)
