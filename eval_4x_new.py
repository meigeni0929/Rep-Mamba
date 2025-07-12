#分类AID数据集上的
#这是加上的指标计算的，再训练一次
from __future__ import print_function
import argparse
import utils
import os
import torch
import cv2
import torch.backends.cudnn as cudnn
import torchvision.transforms as transform
from os import listdir
import math
# ---load model architecture---
from model.mambairv2_l import MambaIRv2Light as NET
import glob
import numpy as np
import socket
import time
import imageio
from PIL import Image
from torchvision.transforms import ToTensor



# Test settings   #实验过程中显示的参数初始值
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=3, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='training batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=1234, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')

#parser.add_argument('--data_dir', type=str, default='D:/SISR/Dataset/test/DIOR1000')  #数据集的位置
parser.add_argument('--data_dir', type=str, default='/root/lanyun-tmp/test/AID900') #这里直接进行测试
parser.add_argument('--model_type', type=str, default='mambairv2x4')
parser.add_argument('--pretrained_sr', default='/root/lanyun-tmp/SMSR/checkpoints/mambairv2x4/models/DID-Data_new/model_epoch_600.pth')
parser.add_argument('--save_folder', default='xr6/mambairv2x4', help='Location to save checkpoint models')   #保存位置在其实数据集的results里

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
cuda = opt.gpu_mode
print(opt)

current_time = time.strftime("%H-%M-%S")    #当前的时间
opt.save_folder = opt.save_folder + current_time + '/'

if not os.path.exists(opt.save_folder):
    os.makedirs(opt.save_folder)

transform = transform.Compose([transform.ToTensor(),])
def PSNR(pred, gt):       #计算指标
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

criterion_ssim = utils.SSIM().cuda()

import numpy as np
from scipy.ndimage import uniform_filter
import cv2

from fvcore.nn import FlopCountAnalysis
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    input = torch.rand(1, 3, 128, 128).cuda()
    #清理显存
    torch.cuda.empty_cache()
    #torch.cuda.reset_peak_memory_allocated()
    # 使用 FlopCountAnalysis 计算 FLOPs
    flops = FlopCountAnalysis(net, input)
    #with torch.no_grad():  # 禁用梯度计算以节省显存
    #    output = net(input)
    #peak_memory = torch.cuda.max_memory_allocated()
    peak_memory_bytes = torch.cuda.max_memory_allocated()
    # 打印总 FLOPs，以 G 为单位
    print(net)
    print("Total FLOPs: {:.2f} G".format(flops.total() / 1e9))
    print('Total number of parameters: %f M' % (num_params / 1e6))
    print("Peak GPU Memory Usage during inference: {:.4f} MB".format(peak_memory_bytes / (1024 ** 2)))
torch.cuda.manual_seed(opt.seed)
device = 'cuda:0'
print('===> Building model ', opt.model_type)
#model = NET(dim=36, n_blocks=8, ffn_scale=2.0, upscaling_factor=2)
model = NET()
model = torch.nn.DataParallel(model, device_ids=gpus_list)
print('---------- Networks architecture -------------')
print_network(model)
model = model.cuda(gpus_list[0])
def re_state(state):
    new_state = {}
    for key in state.keys():
        new_key ="module."+key[:]
        new_state[new_key] = state[key]
    return new_state
model_name = os.path.join(opt.pretrained_sr)
if os.path.exists(model_name):
    # model= torch.load(model_name, map_location=lambda storage, loc: storage)
    #model.load_state_dict(torch.load(model_name))
    old_state = torch.load(model_name, map_location=lambda storage, loc: storage)

    model.load_state_dict(re_state(old_state['state_dict']))
    #model.load_state_dict(old_state)
    print('Pre-trained SR model is loaded.')
else:
    print('No pre-trained model!!!!')
#写函数       
rewrite_print = print
def print_log2(*arg):  # 加上一个保存路径
    save_filder = opt.save_folder + '/test_log.txt'  # 这里应该改成test
    if not os.path.exists(save_filder):  # 没有文件，创建文件
        os.makedirs(save_filder)
    file_path = save_filder + '/test_log.txt'  # 这里应该改为test
    rewrite_print(*arg)
    # 保存到文件
    rewrite_print(*arg, file=open(file_path, "a", encoding="utf-8"))

#定义平均值
aver_panr = 0.0
aver_ssim = 0.0
t = 0.0
avg_t = 0.0
def eval(folder_name):      #测试过程
    global aver_panr, aver_ssim, t
    print('===> Loading val datasets')   #这里开始加载数据 一个类一个类的进行加载
    LR_filename = os.path.join(opt.data_dir, 'LR') + '/' + folder_name
    LR_image = sorted(glob.glob(os.path.join(LR_filename, '*')))  # LR图像路径列表
    GT_filename = os.path.join(opt.data_dir, 'GT') + '/' + folder_name
    GT_image = sorted(glob.glob(os.path.join(GT_filename, '*')))  # LR图像路径列表
    # test begin       #开始测试，预训练好的模型已经给了
    model.eval()
    count=0  #计算的数量
    avg_psnr_predicted=0.0
    avg_ssim_predicted = 0.0
    avg_t0 = 0.0
    #这里输出到结果文件中
    def print_log(*arg):  # 加上一个保存路径
        save_filder=opt.save_folder + folder_name+ '/test_log.txt'   #这里应该改成test
        if not os.path.exists(save_filder):  # 没有文件，创建文件
            os.makedirs(save_filder)
        file_path = save_filder + '/test_log.txt'                     #这里应该改为test
        rewrite_print(*arg)
        # 保存到文件
        rewrite_print(*arg, file=open(file_path, "a", encoding="utf-8"))
    #以上是一些初始化的变量
    t1=0
    t0=0
    for lr_img_path, gt_img_path in zip(LR_image,GT_image):

        # lr = imageio.v2.imread(img_path)
        # lr = np.ascontiguousarray(lr.transpose((2, 0, 1)))
        # lr = torch.from_numpy(lr).float().to(device).unsqueeze(0)

        lr = Image.open(lr_img_path).convert('RGB')
        lr = transform(lr).unsqueeze(0)
        gt = Image.open(gt_img_path).convert('RGB')    #新加入的标签数据
        with torch.no_grad():
            t0 = time.time()
            prediction = model(lr)
            t1 = time.time()
        gt_t = ToTensor()(gt).unsqueeze(0).cuda()  # (1, 3, 512, 512)
        ssim_pre = criterion_ssim(prediction, gt_t)
        prediction = prediction.cpu()
        prediction = prediction.data[0].numpy().astype(np.float32)
        prediction = prediction * 255.0
        prediction = prediction.clip(0, 255)
        # print(prediction.shape)
        prediction = prediction.transpose(1, 2, 0)
        #开始计算相应的PSNR,SSIM的指标
        # 计算相应的指标
        psnr_predicted = PSNR(prediction, gt)

        #ssim_pre=0
        # 输出结果，包括 PSNR 和 SSIM
        print_log("===> Processing image: %s || Timer: %.4f sec. || PSNR: %.4f dB || SSIM: %.4f" % (
        str(count), (t1 - t0), psnr_predicted, ssim_pre))
        print("===> Processing image: %s || Timer: %.4f sec." % (lr_img_path, (t1 - t0)))   #显示训练的数据
        save_name = os.path.splitext(os.path.basename(lr_img_path))[0]
        save_foler = opt.save_folder + folder_name
        if not os.path.exists(save_foler):
            os.makedirs(save_foler)
        save_fn = save_foler + save_name + '.png'
        print('save image to:', save_fn)
        Image.fromarray(np.uint8(prediction)).save(save_fn)   #保存结果的文件地址
        # print(prediction.shape)  # (512, 512, 3)
        # cv2.imwrite(save_fn, prediction, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        # cv2.imwrite(save_fn, cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB), [cv2.IMWRITE_PNG_COMPRESSION, 0])
        avg_psnr_predicted += psnr_predicted
        avg_ssim_predicted += ssim_pre
        avg_t0 += t1-t0
        count = count + 1

    #这里是进行的一些指标的计算，这里是进行平均值的计算
       #计算所有的总值
        #print(count)
    avg_test_psnr = avg_psnr_predicted / count
    avg_test_ssim = avg_ssim_predicted / count
    avg_t = avg_t0 / count
    aver_panr += avg_test_psnr
    aver_ssim += avg_test_ssim
    t += avg_t
    print_log2(
        "===> Processing image: %s || Timer: %.4f sec. || avg_PSNR: %.4f dB|| avg_SSIM: %.4f" % (str(count), t, avg_test_psnr,avg_test_ssim))

if __name__ == '__main__':
    AID_class_name = ['Airport/','BareLand/','BaseballField/','Beach/','Bridge/','Center/','Church/','Commercial/','DenseResidential/',
                     'Desert/','Farmland/','Forest/','Industrial/','Meadow/','MediumResidential/','Mountain/','Park/','Parking/','Playground/',
                      'Pond/','Port/','RailwayStation/','Resort/','River/','School/','SparseResidential/','Square/','Stadium/','StorageTanks/','Viaduct/']
    dota_class = ['']
    #WHU_RS19 = [
    #'Airport', 'Beach', 'Bridge', 'Commercial', 'Desert', 'Farmland', 'footballField','Forest',
    #'Industrial', 'Meadow', 'Mountain', 'Park', 'Parking', 'Pond', 'Port',
    #  'railwayStation', 'Residential', 'River', 'Viaduct'
    #]
    folder = ''
    
    for folder in AID_class_name:    #修改后的
        eval(folder_name=folder)
    print_log2(
        "===> Processing image: %s || Timer: %.4f sec. || avg_PSNR: %.4f dB|| avg_SSIM: %.4f" % (
        str(2800), (t), (aver_panr/30), (aver_ssim/30)))