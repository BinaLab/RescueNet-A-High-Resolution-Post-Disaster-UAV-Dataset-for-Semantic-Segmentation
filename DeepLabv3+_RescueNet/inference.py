import argparse
import scipy
import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy import ndimage
from tqdm import tqdm
from math import ceil
from glob import glob
from PIL import Image
import dataloaders
import models
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter #@sh: add
from utils import transforms as local_transforms # @sh: add
from utils import losses # @sh:add
import datetime # @sh: add
from torch.utils import tensorboard # @sh: add
from utils import logger # @sh: add
import logging # @sh: add

class Inference:
    def __init__(self, model, config, data_loader, loss, device):

        #def __init__(self, model, data_loader, criterion, metric, device):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.loss = loss
        self.config = config
        
        self.wrt_mode, self.wrt_step = 'test_', 0
        #self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.train_loader.batch_size)))
        #if config['trainer']['log_per_iter']: self.log_step = int(self.log_step / self.train_loader.batch_size) + 1

        self.num_classes = self.data_loader.dataset.num_classes
        self.logger = logging.getLogger(self.__class__.__name__)

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            local_transforms.DeNormalize(self.data_loader.MEAN, self.data_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])
        
        if self.device ==  torch.device('cpu'): prefetch = False
        start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
        cfg_trainer = self.config['trainer']
        writer_dir = os.path.join(cfg_trainer['log_dir'], self.config['name'], start_time)
        self.writer = tensorboard.SummaryWriter(writer_dir)

    def get_instance(self, module, name, config, *args):
        # GET THE CORRESPONDING CLASS / FCT 
        return getattr(module, config[name]['type'])(*args, **config[name]['args'])

    def pad_image(self, img, target_size):
        rows_to_pad = max(target_size[0] - img.shape[2], 0)
        cols_to_pad = max(target_size[1] - img.shape[3], 0)
        padded_img = F.pad(img, (0, cols_to_pad, 0, rows_to_pad), "constant", 0)
        return padded_img

    def sliding_predict(self, model, image, num_classes, flip=True):
        image_size = image.shape
        tile_size = (int(image_size[2]//2.5), int(image_size[3]//2.5))
        overlap = 1/3

        stride = ceil(tile_size[0] * (1 - overlap))
        
        num_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)
        num_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
        total_predictions = np.zeros((num_classes, image_size[2], image_size[3]))
        count_predictions = np.zeros((image_size[2], image_size[3]))
        tile_counter = 0

        for row in range(num_rows):
            for col in range(num_cols):
                x_min, y_min = int(col * stride), int(row * stride)
                x_max = min(x_min + tile_size[1], image_size[3])
                y_max = min(y_min + tile_size[0], image_size[2])

                img = image[:, :, y_min:y_max, x_min:x_max]
                padded_img = pad_image(img, tile_size)
                tile_counter += 1
                padded_prediction = model(padded_img)
                if flip:
                    fliped_img = padded_img.flip(-1)
                    fliped_predictions = model(padded_img.flip(-1))
                    padded_prediction = 0.5 * (fliped_predictions.flip(-1) + padded_prediction)
                predictions = padded_prediction[:, :, :img.shape[2], :img.shape[3]]
                count_predictions[y_min:y_max, x_min:x_max] += 1
                total_predictions[:, y_min:y_max, x_min:x_max] += predictions.data.cpu().numpy().squeeze(0)

        total_predictions /= count_predictions
        return total_predictions


    def multi_scale_predict(self, model, image, scales, num_classes, device, flip=False):
        input_size = (image.size(2), image.size(3))
        upsample = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
        total_predictions = np.zeros((num_classes, image.size(2), image.size(3)))

        image = image.data.data.cpu().numpy()
        for scale in scales:
            scaled_img = ndimage.zoom(image, (1.0, 1.0, float(scale), float(scale)), order=1, prefilter=False)
            scaled_img = torch.from_numpy(scaled_img).to(device)
            scaled_prediction = upsample(model(scaled_img).cpu())

            if flip:
                fliped_img = scaled_img.flip(-1).to(device)
                fliped_predictions = upsample(model(fliped_img).cpu())
                scaled_prediction = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
            total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)

        total_predictions /= len(scales)
        return total_predictions


    def save_images(self, image, mask, output_path, image_file, palette):
        # Saves the image, the model output and the results after the post processing
        w, h = image.size
        image_file = os.path.basename(image_file).split('.')[0]
        colorized_mask = colorize_mask(mask, palette)
        colorized_mask.save(os.path.join(output_path, image_file+'.png'))
        # output_im = Image.new('RGB', (w*2, h))
        # output_im.paste(image, (0,0))
        # output_im.paste(colorized_mask, (w,0))
        # output_im.save(os.path.join(output_path, image_file+'_colorized.png'))
        # mask_img = Image.fromarray(mask, 'L')
        # mask_img.save(os.path.join(output_path, image_file+'.png'))

    def _test_epoch(self, arg):
        args = arg
        self.model.to(self.device)
        self.model.eval()
        self.wrt_mode = 'test'

        self._reset_metrics()
        tbar = tqdm(self.data_loader, ncols=130)
        self.valMiou = 0 #@sh: add
        with torch.no_grad():
            val_visual = []
            for batch_idx, (data, target) in enumerate(tbar):
                #data, target = data.to(self.device), target.to(self.device)
                # LOSS
                output = self.model(data)
                loss = self.loss(output.to(self.device), target.to(self.device))
                if isinstance(self.loss, torch.nn.DataParallel):
                    loss = loss.mean()
                self.total_loss.update(loss.item())

                seg_metrics = eval_metrics(output.to(self.device), target.to(self.device), self.num_classes)
                self._update_seg_metrics(*seg_metrics)

                # LIST OF IMAGE TO VIZ (15 images)
                if len(val_visual) < 15:
                    target_np = target.data.cpu().numpy()
                    output_np = output.data.max(1)[1].cpu().numpy()
                    val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])

                # PRINT INFO
                pixAcc, mIoU, _ = self._get_seg_metrics().values()
                self.valMiou = mIoU
                tbar.set_description('EVAL | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f} |'.format(
                                                self.total_loss.average,
                                                pixAcc, mIoU))

            # METRICS TO TENSORBOARD
            #self.wrt_step = (epoch) * len(self.val_loader)
            self.wrt_step = (1) * len(self.data_loader)
            self.writer.add_scalar(f'{self.wrt_mode}/loss', self.total_loss.average, self.wrt_step)
            seg_metrics = self._get_seg_metrics()
            for k, v in list(seg_metrics.items())[:-1]: 
                self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)

            log = {
                'test_loss': self.total_loss.average,
                'test_miou': self.valMiou,
                **seg_metrics
            }

            for k, v in seg_metrics.items():
                    self.logger.info(f'         {str(k):15s}: {v}')

        #return log

        if self.config['save_output']:
            if not os.path.exists('outputs'):
                os.makedirs('outputs')

            to_tensor = transforms.ToTensor()
            normalize = transforms.Normalize(self.data_loader.MEAN, self.data_loader.STD)

            image_files = sorted(glob(os.path.join(args.images, f'*.{args.extension}')))
            with torch.no_grad():
                tbar = tqdm(image_files, ncols=100)
                for img_file in tbar:
                    image = Image.open(img_file).convert('RGB')
                    input = normalize(to_tensor(image)).unsqueeze(0)
                    ''' #@sh: commented out
                    if args.mode == 'multiscale':
                        prediction = multi_scale_predict(model, input, scales, num_classes, device)
                    elif args.mode == 'sliding':
                        prediction = sliding_predict(model, input, num_classes)
                    else:
                        prediction = model(input.to(device))
                        prediction = prediction.squeeze(0).cpu().numpy()'''

                    prediction = self.model(input.to(self.device))
                    prediction = prediction.squeeze(0).cpu().numpy()
                    prediction = F.softmax(torch.from_numpy(prediction), dim=0).argmax(0).cpu().numpy()
                    self.save_images(image, prediction, args.output, img_file, self.data_loader.dataset.palette)

    def _reset_metrics(self):
            self.batch_time = AverageMeter()
            self.data_time = AverageMeter()
            self.total_loss = AverageMeter()
            self.total_inter, self.total_union = 0, 0
            self.total_correct, self.total_label = 0, 0

    def _update_seg_metrics(self, correct, labeled, inter, union):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def _get_seg_metrics(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
        }


def main():
    args = parse_arguments()
    config = json.load(open(args.config))

    # Dataset used for training the model
    dataset_type = config['train_loader']['type']
    assert dataset_type in ['VOC', 'COCO', 'CityScapes', 'ADE20K', 'UAV', 'Michael', 'Harvey']
    if dataset_type == 'CityScapes': 
        scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25] 
    else:
        scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    loader = getattr(dataloaders, config['test_loader']['type'])(**config['test_loader']['args'])
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(loader.MEAN, loader.STD)
    num_classes = loader.dataset.num_classes
    palette = loader.dataset.palette

    # Model
    model = getattr(models, config['arch']['type'])(num_classes, **config['arch']['args'])
    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')
    print(device)

    loss = getattr(losses, config['loss'])(ignore_index = config['ignore_index'])

    checkpoint = torch.load(args.model)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint)

    inference = Inference(model, config, loader, loss, device)
    inference._test_epoch(args)
    
   
def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='VOC',type=str,
                        help='The config used to train the model')
    parser.add_argument('-mo', '--mode', default='multiscale', type=str,
                        help='Mode used for prediction: either [multiscale, sliding]')
    parser.add_argument('-m', '--model', default='model_weights.pth', type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-i', '--images', default=None, type=str,
                        help='Path to the images to be segmented')
    parser.add_argument('-o', '--output', default='outputs', type=str,  
                        help='Output Path')
    parser.add_argument('-e', '--extension', default='jpg', type=str,
                        help='The extension of the images to be segmented')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
