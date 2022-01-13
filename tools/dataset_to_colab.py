from utils.dataset import MultiDataset
import numpy as np
import multiprocessing
import time
from tqdm import tqdm

IMG_SIZE = (512, 512)
BATCH_SIZE = 4


def returnData(x):
    return x

if __name__ == '__main__':

    start_time = time.time()

    # Generating datasets
    print("\n> Generating datasets")
    dataset_type = "training"
    dataset = MultiDataset(BATCH_SIZE, IMG_SIZE, dataset_type)

    train_x = []
    train_y = []

    INDEX_FILE = 0

    p = multiprocessing.Pool(8)
    results = p.imap_unordered(returnData, dataset)

    def saveAtTheEnd():

        global INDEX_FILE
        global train_x
        global train_y

        size = str(len(train_x))

        train_x = np.array(train_x)
        train_y = np.array(train_y)

        np.savez_compressed(size + "-" + dataset_type + "_x_" + str(INDEX_FILE), train_x)
        np.savez_compressed(size + "-" + dataset_type + "_y_" + str(INDEX_FILE), train_y)

        train_x = []
        train_y = []

        INDEX_FILE += 1


    for x, y in tqdm(results, total=len(dataset)):

        train_x.append(x)
        train_y.append(y)

        if len(train_x) >= 100:
            saveAtTheEnd()
            

    saveAtTheEnd()

    print("saved")

    print("--- %s seconds ---" % (time.time() - start_time))