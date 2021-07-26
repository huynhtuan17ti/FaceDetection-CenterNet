'''
    Convert .txt file to .json file
'''
import json
import os

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
                
                if (x2-x1)*(y2 - y1) <= 1000:
                    print('[WARNING] Remove small objects')
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
    convertToJson('dataset/wider_face/wider_face_split/wider_face_train_bbx_gt.txt', 'dataset/wider_face/WIDER_train', "label", True)
    convertToJson('dataset/wider_face/wider_face_split/wider_face_val_bbx_gt.txt', 'dataset/wider_face/WIDER_val', "label", True)
