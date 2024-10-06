import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from models.model import Net

def compute_jacobian(model, image):
    '''
    :param model: the model to compute the Jacobian matrix
    :param image: the input image, 1*784
    :return: the Jacobian matrix, 10*784
    '''
    image_tmp = image.clone().detach().requires_grad_(True).cuda()
    output = model(image_tmp)

    # print(output) # 1*10
    num_features = int(np.prod(image_tmp.shape[1:])) # 784
    # print(var_input.shape) # 1*784
    # print(num_features) # 784
    # print(output.size()) # 1*10
    jacobian = torch.zeros([output.size()[1], num_features]) # 每个logit对每个像素的导数

    for i in range(output.size()[1]): # 遍历每个logit
        if image_tmp.grad is not None:
            image_tmp.grad.zero_()
        output[0][i].backward(retain_graph=True) # retain_graph=True，保留计算图，可以多次反向传播，否则反向传播一次后，计算图就被释放了，再次反向传播会报错
        # print(var_input.grad.shape) # 1*784
        jacobian[i] = image_tmp.grad.clone()

    # print(jacobian) # 10*784
    return jacobian.cuda()


# 计算显著图
def saliency_map(jacobian, target_index, increasing, search_space, nb_features):
    '''
    :param jacobian: the Jacobian matrix of forward derivative, 10*784
    :param target_index: the target class, 0-9
    :param increasing: whether to increase the prediction score, True or False
    :param search_space: the search domain, 1*784
    :param nb_features: the number of features, 784
    :return: the most significant two pixels, p and q
    '''

    domain = torch.eq(search_space, 1).float()  # True->1.0, False->0.0
    all_sum = torch.sum(jacobian, dim=0, keepdim=True) # 为每个像素的梯度求和，1*784
    target_grad = jacobian[target_index]  # target class 对每个像素的梯度，1*784
    others_grad = all_sum - target_grad  # 每个像素除了target class的梯度之和，1*784

    if increasing:
        increase_coef = 2 * (torch.eq(domain, 0)).float().cuda() # 1*784
    else:
        increase_coef = -1 * 2 * (torch.eq(domain, 0)).float().cuda() # 1*784
    
    target_tmp = target_grad.clone() # 1*784
    target_tmp -= increase_coef * torch.max(torch.abs(target_grad))

    alpha = target_tmp.view(-1, 1, nb_features) + target_tmp.view(-1, nb_features, 1)
    # print(alpha.shape) # 1*784*784, alpha[0][i][j] = target_tmp[0][i] + target_tmp[0][j]

    others_tmp = others_grad.clone()
    others_tmp += increase_coef * torch.max(torch.abs(others_grad))
    beta = others_tmp.view(-1, 1, nb_features) + others_tmp.view(-1, nb_features, 1)

    tmp = np.ones((nb_features, nb_features), int)
    np.fill_diagonal(tmp, 0)
    zero_diagonal = torch.from_numpy(tmp).byte().cuda()

    if increasing:
        mask1 = torch.gt(alpha, 0.0) # 逐元素比较alpha和0.0
        mask2 = torch.lt(beta, 0.0) # 逐元素比较beta和0.0
    else:
        mask1 = torch.lt(alpha, 0.0)
        mask2 = torch.gt(beta, 0.0)

    mask = torch.mul(torch.mul(mask1, mask2), zero_diagonal.view_as(mask1))
    saliency_map = torch.mul(torch.mul(alpha, torch.abs(beta)), mask.float())
    max_value, max_idx = torch.max(saliency_map.view(-1, nb_features * nb_features), dim=1)
    p = max_idx // nb_features
    q = max_idx % nb_features
    return p, q


def perturbation_single(image, ys_target, theta, gamma, model):
    '''
    :param image: the input image, 1*784, numpy array
    :param ys_target: the target class, 0-9, int
    :param theta: the perturbation value
    :param gamma: the ratio of the number of features to be perturbed
    :param model: the model to be attacked
    :return: the adversarial sample
    '''
    image_tmp = np.copy(image)
    var_sample = Variable(torch.from_numpy(image_tmp), requires_grad=True).cuda()
    var_target = Variable(torch.LongTensor([ys_target])).cuda()

    if theta > 0:
        increasing = True
    else:
        increasing = False

    num_features = int(np.prod(image_tmp.shape[1:]))
    shape = var_sample.size()

    # perturb two pixels in one iteration, thus max_iters is divided by 2.0
    max_iters = int(np.ceil(num_features * gamma / 2.0))

    # masked search domain, if the pixel has already reached the top or bottom, we don't bother to modify it.
    if increasing:
        search_domain = torch.lt(var_sample, 0.99) #逐元素比较var_sample和0.99
    else:
        search_domain = torch.gt(var_sample, 0.01)
    search_domain = search_domain.view(num_features)

    model.eval().cuda()
    output = model(var_sample)
    current = torch.max(output.data, 1)[1].cpu().numpy()

    iter = 0
    while (iter < max_iters) and (current[0] != ys_target) and (search_domain.sum() != 0):
        jacobian = compute_jacobian(model, var_sample)
        p1, p2 = saliency_map(jacobian, var_target, increasing, search_domain, num_features)
        # apply modifications
        var_sample_flatten = var_sample.view(-1, num_features).clone().detach_()
        var_sample_flatten[0, p1] += theta
        var_sample_flatten[0, p2] += theta

        new_sample = torch.clamp(var_sample_flatten, min=0.0, max=1.0)
        new_sample = new_sample.view(shape)
        search_domain[p1] = 0
        search_domain[p2] = 0
        var_sample = Variable(new_sample, requires_grad=True).cuda()

        output = model(var_sample)
        current = torch.max(output.data, 1)[1].cpu().numpy()
        iter += 1

    adv_samples = var_sample.data.cpu().numpy()
    return adv_samples
