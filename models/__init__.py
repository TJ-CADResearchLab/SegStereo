from models.acv import ACVNet
from models.Att import AttNet
from models.acvSG import ACVSGNet
from models.MSMNet_costadd import MSMNet_cost as EDNet
from models.DispNetC import DispNetC
from models.loss import model_loss_train, model_loss_test ,model_loss_train_scale

__models__ = {
    "acvnet": ACVNet,
    "attnet": AttNet,
    "ednet":EDNet,
    "dispnet":DispNetC,
    "acvsgnet":ACVSGNet
}
