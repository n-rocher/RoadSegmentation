import numpy as np
from tqdm import tqdm
import multiprocessing

from utils.dataset import MultiDataset, MapillaryVistasDataset

IMG_SIZE = (384, 384)

A2D2_FOLDER = r"F:\\A2D2 Camera Semantic\\"
VISTAS_FOLDER = r"F:\\Mapillary Vistas\\"

def returnData(x):
    return x

if __name__ == '__main__':

    dataset_type = "train"

    dataset = MultiDataset(100, IMG_SIZE, dataset_type, A2D2_FOLDER, VISTAS_FOLDER)

    data_image = None
    data_mask = None

    with multiprocessing.Pool(4) as pool:
        results = pool.imap_unordered(returnData, dataset)

        id = 0

        for imgs, masks in tqdm(results, total=len(dataset)):

            if data_image is None:
                data_image = np.array(imgs * 255., np.uint8)
            else:
                data_image = np.concatenate((data_image, imgs), axis=0)

            if data_mask is None:
                data_mask = np.array(masks, np.uint8)
            else:
                data_mask = np.concatenate((data_mask, masks), axis=0)

            id += 1
          
            if id % 10 == 0:
                print("")
                print("saving the file")
                print(data_image.shape, data_mask.shape)
                filename = "DATASET_" + dataset.name() + "_" + dataset_type + "_" + str(data_image.shape[0]) + "-" + str(IMG_SIZE[0]) + "-" + str(IMG_SIZE[1]) + "_CAT-" + str(dataset.classes()) + "_" + str(id % 10)
                np.savez_compressed(filename, data_image=data_image, data_mask=data_mask)
                print("file saved :", filename)
                data_image = None
                data_mask = None