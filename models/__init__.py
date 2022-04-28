from models.acv import ACVNet
from models.Att import AttNet
from models.loss import model_loss_train, model_loss_test

__models__ = {
    "acvnet": ACVNet,
    "attnet": AttNet
}
