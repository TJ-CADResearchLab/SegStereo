from models.acv import ACVNet
from models.Att import AttNet
from models.MSMNet_costadd import MSMNet_cost as EDNet
from models.loss import model_loss_train, model_loss_test ,model_loss_train_scale

__models__ = {
    "acvnet": ACVNet,
    "attnet": AttNet,
    "ednet":EDNet
}
