<p align="center">

![Demo of the road segmentation](image/demo.gif)

</p>

# Road Segmentation
The goal of this project is to create an AI able to segment in real time some categories of objects on the road.
All images are segmented independently.

# Models
I used those models to segment the video :
1. BiSeNet V2 - [paper](https://arxiv.org/abs/2004.02147) - [pretrained model](https://github.com/n-rocher/RoadSegmentation/blob/main/models/BiSeNet-V2_MultiDataset_512-512_epoch-13_loss-0.23_miou_0.54.h5)<br/>
    Result : Mean Intersection Over Union = 54% Loss = 0.23

2. Attention R2U-Net - [pretrained model](https://github.com/n-rocher/RoadSegmentation/blob/main/models/AttentionResUNet-F16_MultiDataset_512-512_epoch-26_loss-0.21_miou_0.55.h5)<br/>
    Result : Mean Intersection Over Union = 55% Loss = 0.21
3. BiSeNet V2 - [paper](https://arxiv.org/abs/2101.06085)
4. TMANet - [paper](https://arxiv.org/abs/2102.08643)

I trained each of these models for about 48 hours with an I7-7700K, a 6GB GTX 1060 and 28GB of RAM.


# Usage

## Testing Segmentation

To test the models on a video, you can use the UI.

First, install required packages : 
> pip install -r requirements.txt

Then, start the UI :
> python segmentation.py [Video Folder Path]

![Ui for testing](image/ui.png)

## Training

To train a model, you first need to download the A2D2 and Mappillary Vistas dataset.

Then, install required packages : 
> pip install -r requirements.txt

After that, you might need to change some constant (dataset folders, epochs, lr, WanDB, ...) in the file `train.py` : 
> code train.py

Finally, start the learning : 
> python train.py


# Categories
Those are the categories trained to be segmented by the AI.

<table class="categories">
    <thead>
        <tr>
            <th>#</th>
            <th>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Name&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>
            <th>Color</th>
        </tr>
    </thead>
    <tbody>
        <tr><td>1</td><td>Road</td><td><img src="https://via.placeholder.com/70x25/4b4b4b/000000?text=+"/></td></tr>
        <tr><td>2</td><td>Lane</td><td><img src="https://via.placeholder.com/70x25/ffffff/000000?text=+"/></td></tr>
        <tr><td>3</td><td>Crosswalk</td><td><img src="https://via.placeholder.com/70x25/c88080/000000?text=+"/></td></tr>
        <tr><td>4</td><td>Curb</td><td><img src="https://via.placeholder.com/70x25/969696/000000?text=+"/></td></tr>
        <tr><td>5</td><td>Sidewalk</td><td><img src="https://via.placeholder.com/70x25/f423e8/000000?text=+"/></td></tr>
        <tr><td>6</td><td>Traffic Light</td><td><img src="https://via.placeholder.com/70x25/faaa1e/000000?text=+"/></td></tr>
        <tr><td>7</td><td>Traffic Sign</td><td><img src="https://via.placeholder.com/70x25/ffff00/000000?text=+"/></td></tr>
        <tr><td>8</td><td>Person</td><td><img src="https://via.placeholder.com/70x25/ff0000/000000?text=+"/></td></tr>
        <tr><td>9</td><td>Bicycle</td><td><img src="https://via.placeholder.com/70x25/582900/000000?text=+"/></td></tr>
        <tr><td>10</td><td>Bus</td><td><img src="https://via.placeholder.com/70x25/ff0f93/000000?text=+"/></td></tr>
        <tr><td>11</td><td>Car</td><td><img src="https://via.placeholder.com/70x25/00ff8e/000000?text=+"/></td></tr>
        <tr><td>12</td><td>Motorcycle</td><td><img src="https://via.placeholder.com/70x25/0000e6/000000?text=+"/></td></tr>
        <tr><td>13</td><td>Truck</td><td><img src="https://via.placeholder.com/70x25/4b0aaa/000000?text=+"/></td></tr>
        <tr><td>14</td><td>Sky</td><td><img src="https://via.placeholder.com/70x25/87ceff/000000?text=+"/></td></tr>
        <tr><td>15</td><td>Nature</td><td><img src="https://via.placeholder.com/70x25/6b8e23/000000?text=+"/></td></tr>
    </tbody>
</table>

# Datasets
The AI was trained using a mix of those two datasets :
1. [A2D2 of Audi](https://www.a2d2.audi/a2d2/en.html) 
2. [Mapillary Vistas](https://www.mapillary.com/dataset/vistas)


# Tools
List of tools I used :
1. [Keras](https://keras.io/)
2. [OpenCV](https://opencv.org/)
3. [Weights & Biases](https://wandb.ai/)