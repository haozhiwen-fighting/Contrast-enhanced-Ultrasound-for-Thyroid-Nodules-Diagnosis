import cv2
import numpy as np
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from networks.cla_pro import cla_p
from networks.cla_3D_family.R3D_model import R3DClassifier
from networks.cla_3D_family.C3D_model import C3D
from networks.seg_sub.textnet import TextNet
import scipy

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compatibility with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x,x1):
        self.gradients = []
        self.activations = []
        return self.model(x,x1)

    def release(self):
        for handle in self.handles:
            handle.remove()


class GradCAM:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=(2, 3), keepdims=True)

    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self, activations, grads):
        # activations=activations.squeeze(dim=2)
        # grads=grads.squeeze(dim=2)
        weights = self.get_cam_weights(grads)
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)

        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.squeeze(dim=2).cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.squeeze(dim=2).cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def __call__(self, input_tensor,label_tensor, target_category=None):

        if self.cuda:
            input_tensor = input_tensor.cuda()
            label_tensor=label_tensor.cuda()

        # 正向传播得到网络输出logits(未经过softmax)
        output = self.activations_and_grads(input_tensor,label_tensor)
        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"category id: {target_category}")
        else:
            assert (len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        # cam_per_layer=cam_per_layer[0]
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    # mask=mask[:,:,4:]
    # a=[]
    # a=img.shape
    mask = cv2.resize(mask, (400,600), dst=None, fx=None, fy=None, interpolation=None)
    # temperature_array_extented = scipy.ndimage.zoom(mask, (400,400), order=0)

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = 0.5*heatmap + img[:,400:,:]
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def center_crop_img(img: np.ndarray, size: int):
    h, w, c = img.shape

    if w == h == size:
        return img

    if w < h:
        ratio = size / w
        new_w = size
        new_h = int(h * ratio)
    else:
        ratio = size / h
        new_h = size
        new_w = int(w * ratio)

    img = cv2.resize(img, dsize=(new_w, new_h))

    if new_w == size:
        h = (new_h - size) // 2
        img = img[h: h+size]
    else:
        w = (new_w - size) // 2
        img = img[:, w: w+size]

    return img

def main():
    model = cla_p()
    # model=R3DClassifier(2)
    # model=TextNet()
    # model=C3D(2)
    weights_path = "D:/HHJ/second/CEUS/classification/pretrained_models/model_cla.pt"
    # weights_path = "D:/HHJ/second/TextPMs/model\CEUS/TextPMs_vnet_98.pth"
    checkpoint=torch.load(weights_path) 
    # model.load_state_dict(checkpoint['model'])
    model.load_state_dict(checkpoint['state_dict'])
    target_layers = [model.cla3.pool2]
    # target_layers = [model.rrgn.SepareConv1[-1]]
    # target_layers = [model.pool5]
    data_transform = transforms.Compose([
                                        transforms.Resize((224,448)),
                                        # transforms.Resize((224,224)),
                                        # transforms.Resize((112,224)),
                                        # transforms.Resize((112,112)),
                                        transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    data_transform1 = transforms.Compose([
                                        transforms.Resize((224,448)),
                                        # transforms.Resize((224,224)),
                                        # transforms.Resize((112,224)),
                                        # transforms.Resize((112,112)),
                                        transforms.ToTensor(),
                                         ])
    # load image
    img_path = "D:/data/thyoid video/data_aug1/train/1/0027_1/keyframes/"
    img_list=[]
    #######
    imagelabel=Image.open('D:/data/thyoid video/data_aug1/train/1/0027_1/mask.png').convert('L')
    # lable=cv2.imread('D:/data/thyoid video/data_aug1/train/0/0096_2/mask.png',1)
    # gray = cv2.cvtColor(lable,cv2.COLOR_BGR2GRAY)
    # ret,binary = cv2.threshold (gray,127,255, cv2.THRESH_BINARY)
    # input,contours,hierarchy = cv2.findContours (binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    # #一—-------------构造矩形边界-------------
    # x,y,w,h = cv2.boundingRect (contours[0])
    # leng=max(w,h)
    # if leng==w:
    #     cv2.rectangle(lable, (x-20, y-20-int((w-h)/2)), (x+leng+20, y-int((w-h)/2)+leng+20),(255,255,255),2)
    #     cropped = lable[(y-20-int((w-h)/2)):( y-int((w-h)/2)+leng+20),(x-20):(x+leng+20)]
    #     imagelabel=Image.fromarray(cropped)  
    # else:
    #     cv2.rectangle(lable, (x-20-int((h-w)/2), y-20), (x+leng+20-int((h-w)/2), y+leng+20),(255,255,255),2)
    #     cropped = lable[(y-20):( y+leng+20),(x-20-int((h-w)/2)):(x+leng+20-int((h-w)/2))]
    #     imagelabel=Image.fromarray(cropped) 
    # imagelabel.show()

    ########
    for i in os.listdir(img_path):
        path=img_path+i
        img = Image.open(path).convert('RGB')
        # if leng==w:
        #         cv2.rectangle(lable, (x-20, y-20-int((w-h)/2)), (x+leng+20, y-int((w-h)/2)+leng+20),(255,255,255),2)
        #         box = (x-20, y-20-int((w-h)/2), x+leng+20, y-int((w-h)/2)+leng+20)
        #         img = img.crop(box)        
        # else:
        #     cv2.rectangle(lable, (x-20-int((h-w)/2), y-20), (x+leng+20-int((h-w)/2), y+leng+20),(255,255,255),2)
        #     box = (x-20-int((h-w)/2), y-20, x+leng+20-int((h-w)/2), y+leng+20)
        #     img = img.crop(box)
        img_tensor = data_transform(img).unsqueeze(dim=0)
        img_tensor=img_tensor[:,:,:,224:]
        # img_tensor=img_tensor[:,:,:,112:]
        img_list.append(img_tensor)
    img_t=torch.cat(img_list,0)
    # label=Image.open('D:/data/thyoid video/data_aug1/train/0/0066_1/mask.png').convert('L')

    label_tensor = data_transform1(imagelabel).unsqueeze(dim=0)
    label_tensor=label_tensor[:,:,:,224:]
    # label_tensor=label_tensor[:,:,:,112:]
    input_tensor = torch.unsqueeze(img_t, dim=0)
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    target_category = 1  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor,label_tensor=label_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    img = np.array(img, dtype=np.uint8)
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()
    plt.imsave('C:/Users/hp/Desktop/ours/27.png',visualization)
    a=1


if __name__ == '__main__':
    main()