import torch.nn.functional as F
import torch
from models.submoduleEDNet import resample2d
import torch.nn as nn
from utils.experiment import make_nograd_func

config = {"weights": [0.5,0.5, 0.7, 1.0],
          "scales": [0,0, 0, 0],
          "refine_weights": [ 1.0],
          "refine_scales": [ 0],
          "occ_masks": [0.1, 0.1, 0.1,0.1],
          "refine_occ_masks": [0.01]
          }


def model_loss_train_self(disp_ests, imgL, imgR, refine_mode, occ_masks, only_train_refine=False, seg_features=None):
    weights = config["weights"]
    scales = config["scales"]
    refine_weights = config["refine_weights"]
    refine_scales = config["refine_scales"]

    if refine_mode:
        if only_train_refine:
            for i in range(len(weights)):
                weights[i] = 0.0
        disp_ests, refine_disp_ests = disp_ests[:len(weights)], disp_ests[len(weights):]
        occ_masks, refine_occ_masks = occ_masks[:len(weights)], occ_masks[len(weights):]

    all_losses = []
    for disp_est, weight, scale, occ_mask in zip(disp_ests, weights, scales, occ_masks):
        if weight == 0:
            continue
        imgR_cur = F.interpolate(imgR, scale_factor=1 / 2 ** scale)
        imgL_cur = F.interpolate(imgL, scale_factor=1 / 2 ** scale)
        left_rec = resample2d(imgR_cur, disp_est)
        mean_disp = disp_est.mean(1, True).mean(2, True)
        norm_disp = disp_est / (mean_disp + 1e-7)

        # origin disp loss: img-disp smooth , ph ,ssim
        smooth_loss = get_smooth_loss(norm_disp.unsqueeze(1), imgL_cur)
        loss_ph = F.smooth_l1_loss(left_rec[occ_mask], imgL_cur[occ_mask], size_average=True)
        loss_ssim = SSIM(left_rec, imgL_cur)[occ_mask].mean()
        all_losses.append(weight * (0.001 * smooth_loss + 0.15 * loss_ph + 0.85 * loss_ssim))

    if refine_mode:
        for disp_est, weight, scale, occ_mask, seg_feature in zip(refine_disp_ests, refine_weights, refine_scales,
                                                                  refine_occ_masks, seg_features):
            imgR_cur = F.interpolate(imgR, scale_factor=1 / 2 ** scale)
            imgL_cur = F.interpolate(imgL, scale_factor=1 / 2 ** scale)
            left_rec = resample2d(imgR_cur, disp_est)
            mean_disp = disp_est.mean(1, True).mean(2, True)
            norm_disp = disp_est / (mean_disp + 1e-7)

            # refine disp loss: seg-disp smooth , ph (occ 阈值不一样) , ssim
            smooth_loss = get_smooth_loss(norm_disp.unsqueeze(1), seg_feature)
            loss_ph = F.smooth_l1_loss(left_rec[occ_mask], imgL_cur[occ_mask], size_average=True)
            loss_ssim = SSIM(left_rec, imgL_cur)[occ_mask].mean()
            all_losses.append(weight * (0.01 * smooth_loss + 0.15 * loss_ph + 0.85 * loss_ssim))

        # L1 loss
        L1 = False
        if L1:
            for i in range(len(refine_disp_ests)):
                disp_gt = refine_disp_ests[-(i + 1)].detach()
                all_losses.append(0.1 * refine_weights[-(i + 1)] * F.smooth_l1_loss(disp_ests[-(i + 1)], disp_gt))

    return sum(all_losses)


def model_loss_train(disp_ests, maxdisp, refine_mode, disp_gt, only_train_refine=False, seg_features=None):
    weights = config["weights"]
    scales = config["scales"]
    refine_weights = config["refine_weights"]
    refine_scales = config["refine_scales"]

    all_losses = []
    if refine_mode:
        if only_train_refine:
            for i in range(len(weights)):
                weights[i] = 0.0
        weights = weights + refine_weights
        scales = scales + refine_scales

        # seg train with disp_gt for only once
        assert len(seg_features) == 1
        seg_feature = seg_features[0]
        seg_feature = F.interpolate(seg_feature, size=[disp_gt.size()[1], disp_gt.size()[2]], mode='bilinear',
                                    align_corners=True)  # [b, c, h, w]
        mean_disp = disp_gt.mean(1, True).mean(2, True)
        norm_disp = disp_gt / (mean_disp + 1e-7)
        all_losses.append(0.1 * get_smooth_loss(norm_disp.unsqueeze(1), seg_feature))

    for disp_est, weight, scale in zip(disp_ests, weights, scales):
        if weight == 0:
            continue
        gt_scale = F.interpolate(disp_gt.unsqueeze(1), scale_factor=1 / 2 ** scale) / 2 ** scale
        gt_scale = torch.squeeze(gt_scale, dim=1)
        mask = (gt_scale < (maxdisp) / 2 ** scale) & (gt_scale > 0)
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], gt_scale[mask], size_average=True))

    return sum(all_losses)


def model_loss_test(disp_ests, disp_gt, mask):
    all_losses = []
    for disp_est in disp_ests:
        all_losses.append(F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)


@make_nograd_func
def compute_occmask(model, imgL, imgR, disp_ests, refine_mode):
    occ_masks = []
    imgL_rev = imgL[:, :, :, torch.arange(imgL.size(3) - 1, -1, -1)]
    imgR_rev = imgR[:, :, :, torch.arange(imgR.size(3) - 1, -1, -1)]
    if refine_mode:
        disp_right, _ = model(imgR_rev, imgL_rev, refine_mode=True)
        occ_a = config["occ_masks"] + config["refine_occ_masks"]
    else:
        disp_right = model(imgR_rev, imgL_rev, refine_mode=False)
        occ_a = config["occ_masks"]
    disp_right = [i[:, :, torch.arange(i.size(2) - 1, -1, -1)] for i in disp_right]
    for i in range(len(disp_right)):
        disp_rec = resample2d(-disp_right[i], disp_ests[i])
        occ = (torch.abs(disp_rec + disp_ests[i]) > occ_a[i] * (torch.abs(disp_rec) + torch.abs(disp_ests[i])) + 0.5) | (
                disp_rec == 0)  # from occlusion aware
        occ = ~occ
        occ_masks.append(occ.unsqueeze(1).repeat(1, 3, 1, 1))
    return occ_masks


def SSIM(x, y):
    x = F.pad(x, (1, 1, 1, 1), mode='reflect')
    y = F.pad(y, (1, 1, 1, 1), mode='reflect')
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, 3, 1)
    mu_y = F.avg_pool2d(y, 3, 1)

    sigma_x = F.avg_pool2d(x ** 2, 3, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(y ** 2, 3, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class ContrastiveCorrelationLoss(nn.Module):

    def __init__(self):
        super(ContrastiveCorrelationLoss, self).__init__()
        self.pointwise = True
        self.zero_clamp = True
        self.use_salience = False
        self.stabalize = False
        self.feature_samples = 20
        self.pos_intra_shift = 0.18
        self.pos_inter_shift = 0.12
        self.neg_samples = 10
        self.neg_inter_shift = 0.46

    def norm(self, t):
        return F.normalize(t, dim=1, eps=1e-10)

    def sample_nonzero_locations(self, t, target_size):
        nonzeros = torch.nonzero(t)
        coords = torch.zeros(target_size, dtype=nonzeros.dtype, device=nonzeros.device)
        n = target_size[1] * target_size[2]
        for i in range(t.shape[0]):
            selected_nonzeros = nonzeros[nonzeros[:, 0] == i]
            if selected_nonzeros.shape[0] == 0:
                selected_coords = torch.randint(t.shape[1], size=(n, 2), device=nonzeros.device)
            else:
                selected_coords = selected_nonzeros[torch.randint(len(selected_nonzeros), size=(n,)), 1:]
            coords[i, :, :, :] = selected_coords.reshape(target_size[1], target_size[2], 2)
        coords = coords.to(torch.float32) / t.shape[1]
        coords = coords * 2 - 1
        return torch.flip(coords, dims=[-1])

    def tensor_correlation(self, a, b):
        return torch.einsum("nchw,ncij->nhwij", a, b)

    def sample(self, t: torch.Tensor, coords: torch.Tensor):
        if len(t.size()) == 3:
            return F.grid_sample(t.unsqueeze(1), coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)
        else:
            return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)

    def super_perm(self, size: int, device: torch.device):
        perm = torch.randperm(size, device=device, dtype=torch.long)
        perm[perm == torch.arange(size, device=device)] += 1
        return perm % size

    def standard_scale(self, t):
        t1 = t - t.mean()
        t2 = t1 / t1.std()
        return t2

    def helper(self, f1, f2, c1, c2, shift):
        with torch.no_grad():
            # Comes straight from backbone which is currently frozen. this saves mem.
            fd = self.tensor_correlation(self.norm(f1), self.norm(f2))

            if self.pointwise:
                old_mean = fd.mean()
                fd -= fd.mean([3, 4], keepdim=True)
                fd = fd - fd.mean() + old_mean

        cd = self.tensor_correlation(self.norm(c1), self.norm(c2))

        if self.zero_clamp:
            min_val = 0.0
        else:
            min_val = -9999.0

        if self.stabalize:
            loss = - cd.clamp(min_val, .8) * (fd - shift)
        else:
            loss = - cd.clamp(min_val) * (fd - shift)

        return loss, cd

    def forward(self,
                orig_feats: torch.Tensor, orig_feats_pos: torch.Tensor,
                orig_salience: torch.Tensor, orig_salience_pos: torch.Tensor,
                orig_code: torch.Tensor, orig_code_pos: torch.Tensor,
                ):

        coord_shape = [orig_feats.shape[0], self.feature_samples, self.feature_samples, 2]

        if self.use_salience:
            coords1_nonzero = self.sample_nonzero_locations(orig_salience, coord_shape)
            coords2_nonzero = self.sample_nonzero_locations(orig_salience_pos, coord_shape)
            coords1_reg = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
            coords2_reg = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
            mask = (torch.rand(coord_shape[:-1], device=orig_feats.device) > .1).unsqueeze(-1).to(torch.float32)
            coords1 = coords1_nonzero * mask + coords1_reg * (1 - mask)
            coords2 = coords2_nonzero * mask + coords2_reg * (1 - mask)
        else:
            coords1 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
            coords2 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1

        feats = self.sample(orig_feats, coords1)
        code = self.sample(orig_code, coords1)

        feats_pos = self.sample(orig_feats_pos, coords2)
        code_pos = self.sample(orig_code_pos, coords2)

        pos_intra_loss, pos_intra_cd = self.helper(
            feats, feats, code, code, self.pos_intra_shift)
        pos_inter_loss, pos_inter_cd = self.helper(
            feats, feats_pos, code, code_pos, self.pos_inter_shift)

        neg_losses = []
        neg_cds = []
        for i in range(self.neg_samples):
            perm_neg = self.super_perm(orig_feats.shape[0], orig_feats.device)
            feats_neg = self.sample(orig_feats[perm_neg], coords2)
            code_neg = self.sample(orig_code[perm_neg], coords2)
            neg_inter_loss, neg_inter_cd = self.helper(
                feats, feats_neg, code, code_neg, self.neg_inter_shift)
            neg_losses.append(neg_inter_loss)
            neg_cds.append(neg_inter_cd)
        neg_inter_loss = torch.cat(neg_losses, axis=0)
        neg_inter_cd = torch.cat(neg_cds, axis=0)

        return (pos_intra_loss.mean(),
                pos_intra_cd,
                pos_inter_loss.mean(),
                pos_inter_cd,
                neg_inter_loss,
                neg_inter_cd)

class ContrastiveCRFLoss(nn.Module):

    def __init__(self):
        super(ContrastiveCRFLoss, self).__init__()
        self.alpha = 0.5
        self.beta = 0.15
        self.gamma = 0.05
        self.w1 = 10.0
        self.w2 = 3.0
        self.n_samples = 1000
        self.shift = 0.00

    def forward(self, guidance, clusters):
        device = clusters.device
        assert (guidance.shape[0] == clusters.shape[0])
        assert (guidance.shape[2:] == clusters.shape[2:])
        h = guidance.shape[2]
        w = guidance.shape[3]

        coords = torch.cat([
            torch.randint(0, h, size=[1, self.n_samples], device=device),
            torch.randint(0, w, size=[1, self.n_samples], device=device)], 0)

        selected_guidance = guidance[:, :, coords[0, :], coords[1, :]]
        coord_diff = (coords.unsqueeze(-1) - coords.unsqueeze(1)).square().sum(0).unsqueeze(0)
        guidance_diff = (selected_guidance.unsqueeze(-1) - selected_guidance.unsqueeze(2)).square().sum(1)

        sim_kernel = self.w1 * torch.exp(- coord_diff / (2 * self.alpha) - guidance_diff / (2 * self.beta)) + \
                     self.w2 * torch.exp(- coord_diff / (2 * self.gamma)) - self.shift

        selected_clusters = clusters[:, :, coords[0, :], coords[1, :]]
        cluster_sims = torch.einsum("nka,nkb->nab", selected_clusters, selected_clusters)
        return -(cluster_sims * sim_kernel)
