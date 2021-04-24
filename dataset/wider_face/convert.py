'''
    Convert .txt file to .json file
'''
import json
import os
from config import Config

def convertToJson(annotationsPath, targetDir, nameJson, validOnly):
    infos = []
    num = 0
    with open(annotationsPath) as f:
        if not os.path.exists(targetDir):
            os.makedirs(targetDir)
        line = f.readline().strip()
        while line:
            info = {}
            imFilename = line
            info['path'] = imFilename
            bboxes = []
            nbBndboxes = f.readline().strip()
            # print(line, nbBndboxes)
            i = 0
            isBbox = False
            while i < int(nbBndboxes):
                i = i + 1
                x1, y1, w, h, blur, expr, illum, invalid, occl, pose = [int(k) for k in f.readline().split()]
                x2 = x1 + w
                y2 = y1 + h

                if (x2 <= x1 or y2 <= y1):
                    print('[WARNING] Invalid box dimensions in image "{}" x1 y1 w h: {} {} {} {}'.format(imFilename,x1,y1,w,h))
                    continue

                if validOnly and invalid == 1:
                    continue

                bboxes.append((x1, y1, x2, y2))
                isBbox = True
            if isBbox:
                num += 1
                info['bbox'] = bboxes
                info['id'] = num
                infos.append(info)
            
            if int(nbBndboxes) == 0:
                line = f.readline().strip() # read blank

            line = f.readline().strip()

    with open(os.path.join(targetDir, nameJson + '.json'), 'w') as outfile:
        json.dump(infos, outfile)

if __name__ == '__main__':
    cfg = Config()
    convertToJson(cfg.wider_train_path, cfg.json_path, "train_label", True)
    convertToJson(cfg.wider_valid_path, cfg.json_path, "valid_label", True)
