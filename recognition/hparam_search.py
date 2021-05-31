import optuna
from argparse import Namespace


def objective(trial: optuna.Trial, hparams: Namespace):
    """Return the objective loss for an optuna trial.

    Args:
        trial (optuna.Trial): The optuna trial.
        hparams (Namespace): The argparse `Namespace` that will be passed to
        the `LightningModule`.
    """
    hparams.loss = trial.suggest_categorical('loss', ['CrossEntropy',
                                                      'TripletMargin',
                                                      'ArcFace'])

    if hparams.loss == 'ArcFace':
        hparams.loss_margin = trial.suggest_uniform('loss_margin', 0.3, 0.9)
        hparams.sampler = None
    elif hparams.loss == 'TripletMargin':
        hparams.loss_margin = trial.suggest_uniform('loss_margin', 0, 0.2)
        hparams.sampler = 'MPerClass'
        hparams.m_per_class = 5
        hparams.miner = trial.suggest_categorical('miner', ['BatchHard', None])

    hparams.optim = 'SGD'
    hparams.lr = trial.suggest_loguniform('lr', 1e-8, 1e0)

    max_steps = 1e4

    hparams.lr_sched = trial.suggest_categorical('lr_sched', [None,
                                                              'OneCycleLr'])
    if hparams.lr_sched == 'OneCycleLR':
        hparams.lr_sched_total_steps = max_steps
        hparams.lr_sched_max_lr = 10 * hparams.lr

    hparams.use_sample_data = True
