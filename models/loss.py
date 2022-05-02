import torch.nn.functional as F
import torch


def model_loss_train(disp_ests, disp_gt,maxdisp):
    mask = (disp_gt < maxdisp) & (disp_gt > 0)
    weights = [0.7, 0.5, 0.7, 1.0]   #[0.5, 0.5, 0.7, 1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)

def model_loss_train_scale(disp_ests, disp_gt, maxdisp):
    weights = [0.6, 0.8, 0.8, 1.0,1.0]
    all_losses = []
    scales=[0,1,2,3,3]
    for disp_est, weight ,scale in zip(disp_ests, weights,scales):
        disp_gt_cur=F.avg_pool2d(disp_gt,(2**scale,2**scale))
        mask = (disp_gt_cur < maxdisp) & (disp_gt_cur > 0)
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt_cur[mask], size_average=True))
    return sum(all_losses)
    
def model_loss_test(disp_ests, disp_gt, mask):
    weights = [1.0] 
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)
