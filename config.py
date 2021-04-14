class Config(object):
    CLASSES_NAME = ('face')

    down_stride = 4
    slug = 'r50'
    fpn = False
    freeze_bn = False
    resize_size = (512, 512)
    
    bn_momentum = 0.1

    head_channel = 64

    num_classes = 1

    score_th = 0.1
    lr = 1.25e-4
    train_batch = 8
    valid_batch = 16

    gamma = 0.1
    milestones = [15, 25, 30]
    epochs = 35

    train_path = '../FaceDetection-CenterNet/dataset/train'
    valid_path = '../FaceDetection-CenterNet/dataset/validation'
    save_path = '../FaceDetection-CenterNet/save_model'
    save_model = 'resnet18.pth'

    pretrained_path = '../FaceDetection-CenterNet/pretrained_model/'
    pretrained_model = 'ctdet_pascal_resdcn18_512.pth'

    test_path = '../FaceDetection-CenterNet/test_img'

    heads = {'hm':1, 'wh':2, 'reg':2}
    head_conv = 64
    model_name = 'resnet18'
    pretrained = True
    device_inference = 'cuda'

    custom_net = False