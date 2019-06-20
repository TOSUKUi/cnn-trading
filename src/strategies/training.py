import chainer 
from  chainer import training
import argparse
from chainer import links as L
from chainer.training import triggers, extensions
from chainer.datasets import split_dataset
from chainer.functions import mean_squared_error
from pipeline import  Procedure
from sklearn.model_selection import train_test_split



class TrainProcedureKeras(Procedure):

    def __init__(self, model, **kwargs): 
        self.model = model
        self.kwargs = kwargs
    
    def run(self, data):
        return self.train_model(*data)

    def train_model(self, data):
        return self.model.train(*data, **self.kwargs)
    

class TrainProcedureChainer(Procedure):

    def __init__(self, model):
        self.model = model 

    def run(self, x):
        return self.train_model(x)

    def train_model(self, datasets):
        parser = argparse.ArgumentParser(description='Chainer CIFAR example:')
        parser.add_argument('--dataset', '-d', default='cifar10',
                            help='The dataset to use: cifar10 or cifar100')
        parser.add_argument('--batchsize', '-b', type=int, default=10,
                            help='Number of images in each mini-batch')
        parser.add_argument('--learnrate', '-l', type=float, default=0.05,
                            help='Learning rate for SGD')
        parser.add_argument('--epoch', '-e', type=int, default=300,
                            help='Number of sweeps over the dataset to train')
        parser.add_argument('--gpu', '-g', type=int, default=-1,
                            help='GPU ID (negative value indicates CPU)')
        parser.add_argument('--out', '-o', default='result',
                            help='Directory to output the result')
        parser.add_argument('--resume', '-r', default='',
                            help='Resume the training from snapshot')
        parser.add_argument('--early-stopping', type=str,
                            help='Metric to watch for early stopping')
        args = parser.parse_args()


        print('GPU: {}'.format(args.gpu))
        print('# Minibatch-size: {}'.format(args.batchsize))
        print('# epoch: {}'.format(args.epoch))


        if args.gpu >= 0:
            chainer.backends.cuda.get_device_from_id(args.gpu).use()
            self.model.to_gpu()

        optimizer = chainer.optimizers.Adam(args.learnrate)
        optimizer.setup(self.model)
        optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(5e-4))

        train, test = split_dataset(datasets, 80)

        train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
        test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

        stop_trigger = (args.epoch, 'epoch')
        # Early stopping option
        if args.early_stopping:
            stop_trigger = triggers.EarlyStoppingTrigger(
                monitor=args.early_stopping, verbose=True,
                max_trigger=(args.epoch, 'epoch'))

        # Set up a trainer
        updater = training.updaters.StandardUpdater(
            train_iter, optimizer, device=args.gpu, loss_func=mean_squared_error)
        trainer = training.Trainer(updater, stop_trigger, out=args.out)

        # Evaluate the model with the test dataset for each epoch
        trainer.extend(extensions.Evaluator(test_iter, self.model, device=args.gpu))

        # Reduce the learning rate by half every 25 epochs.
        trainer.extend(extensions.ExponentialShift('lr', 0.5),
                    trigger=(25, 'epoch'))

        # Dump a computational graph from 'loss' variable at the first iteration
        # The "main" refers to the target link of the "main" optimizer.
        trainer.extend(extensions.dump_graph('main/loss'))

        # Take a snapshot at each epoch
        trainer.extend(extensions.snapshot(
            filename='snaphot_epoch_{.updater.epoch}'))

        # Write a log of evaluation statistics for each epoch
        trainer.extend(extensions.LogReport())

        # Print selected entries of the log to stdout
        # Here "main" refers to the target link of the "main" optimizer again, and
        # "validation" refers to the default name of the Evaluator extension.
        # Entries other than 'epoch' are reported by the Classifier link, called by
        # either the updater or the evaluator.
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
            'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

        # Print a progress bar to stdout
        trainer.extend(extensions.ProgressBar())

        if args.resume:
            # Resume from a snapshot
            chainer.serializers.load_npz(args.resume, trainer)

        print(train[:1])

        # Run the training
        trainer.run()

        return self.model