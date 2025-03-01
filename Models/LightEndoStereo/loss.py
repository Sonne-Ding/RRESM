import torch.nn.functional as F

def single_loss(disp_ests, disp_gt, mask):
    disp_est = disp_ests[0]
    loss = F.smooth_l1_loss(disp_est[mask], disp_gt[mask], reduction='mean')
    return loss

def avg_loss(disp_ests, disp_gt, mask):
    # weights = [0.5, 0.5, 0.7, 1.0]
    all_losses = []
    for disp_est in disp_ests:
        all_losses.append(F.smooth_l1_loss(disp_est[mask], disp_gt[mask], reduction='mean'))
    return sum(all_losses)/len(disp_ests)

def mix_loss(disp_ests, disp_gt, mask):
    disploss = F.smooth_l1_loss(disp_ests[0][mask], disp_gt[mask], reduction="mean")
    return disploss+disp_ests[1]

def gwc_loss(disp_ests, disp_gt, mask):
    
    weights = [0.4, 0.4, 0.1, 0.1]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        # print(disp_est.shape)
        # print(disp_gt.shape)
        # assert False, "debug"
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], reduction='mean'))
    return sum(all_losses)

def cosine_loss(disp_ests, disp_gt, mask):
    weights = [0.5, 0.5, 0.7, 1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], reduction='mean'))
    return sum(all_losses)

