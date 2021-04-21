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
    train_batch = 16
    valid_batch = 32

    gamma = 0.1
    milestones = [15, 25, 30]
    epochs = 35

    # Human faces from google dataset
    train_path = '../FaceDetection-CenterNet/GoogleDataset/train'
    valid_path = '../FaceDetection-CenterNet/GoogleDataset/validation'

    # WiderFace dataset
    wider_train_path = '../FaceDetection-CenterNet/WiderFaceDataset/WIDER_train'
    wider_valid_path = '../FaceDetection-CenterNet/WiderFaceDataset/WIDER_val'
    train_label_path = '../FaceDetection-CenterNet/WiderFaceDataset/wider_face_split/wider_face_train_bbx_gt.txt'
    valid_label_path = '../FaceDetection-CenterNet/WiderFaceDataset/wider_face_split/wider_face_val_bbx_gt.txt'
    json_path = '../FaceDetection-CenterNet/WiderFaceDataset/Label'

    use_wider = False


    save_path = '../FaceDetection-CenterNet/save_model'
    save_model = 'resnet18_google_data.pth'

    pretrained_path = '../FaceDetection-CenterNet/pretrained_model/'
    pretrained_model = 'ctdet_pascal_resdcn18_512.pth'

    test_path = '../FaceDetection-CenterNet/test_img'
    save_img_path = '../FaceDetection-CenterNet/images'

    heads = {'hm':1, 'wh':2, 'reg':2}
    head_conv = 64
    model_name = 'resnet18'
    pretrained = True
    device_inference = 'cuda'

    custom_net = False
    resume = True

    # expand bouding boxes to cover the head
    expand_bbox = True
    expend_percent = 0.05