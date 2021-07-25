from inference import *
import yaml
import cv2

def show_heatmap(config_path, test_path):
    config = yaml.safe_load(open(config_path))
    device = torch.device('cpu' if config['gpu'] < 0 else 'cuda:%s' % config['gpu'])
    net = load_model(config, device)
    net.eval()

    img_list = os.listdir(test_path)
    K = len(img_list)
    fig = plt.figure(figsize=(20, 10*K))
    for idx, img_name in enumerate(img_list):
        img_path = os.path.join(test_path, img_name)
        img = get_img(img_path)

        img_paded, info = preprocess_img(img, (config['img_size'], config['img_size']))
        imgs = [img]
        infos = [info]

        input = transforms.ToTensor()(img_paded)
        input = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(input)
        inputs = input.unsqueeze(0).to(device)

        detects = inference(config, net, inputs, infos, topK = 50, return_hm = True, th=0.2)
        detect = detects[0]
        hm = detect[3].permute(1, 2, 0).cpu().detach().numpy()[:, :, 0]
        hm = cv2.resize(hm, (config['img_size'], config['img_size']))
        ax = plt.subplot(K, 1, idx+1)
        plt.xticks([])
        plt.yticks([])
        # plt.tight_layout(pad = 1.0)

        plt.imshow(img_paded)
        plt.imshow(hm, alpha=0.3)
    plt.show()

