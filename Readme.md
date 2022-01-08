<p align="center">

![Demo of BiSeNet V2](image/bisenetv2.gif)

</p>

# Road Image Segmentation

The purpose of this project is to create a real-time capable AI to detect a few categories of object on the road.
All images are segmented independantly.

# Models
I used those models to segment the video :
1. BiSeNet V2 - [paper](https://arxiv.org/abs/2004.02147)
2. Attention U-net - [paper](https://arxiv.org/abs/1802.06955)
3. R2U-Net - [paper](https://arxiv.org/abs/1802.06955)
4. Attention R2U-Net

I've trained those models using I7-7700K with one GTX 1060 and 28 Go of ram.

# Categories

Those are the categories trained to be segmented by the AI.

<style>
    .categories th:nth-child(1), .categories td:nth-child(1) {
        text-align: center;
        width: 50px
    }
    .categories th:nth-child(3), .categories td:nth-child(3) {
        text-align: center;
        width: 50px
    }
</style>

<table class="categories">
    <thead>
        <tr>
            <th>#</th>
            <th style="width:250px">Name</th>
            <th>Color</th>
        </tr>
    </thead>
    <tbody>
        <tr><td>1</td><td>Road</td><td style="background: #4b4b4b"></td></tr>
        <tr><td>2</td><td>Lane</td><td style="background: #ffffff"></td></tr>
        <tr><td>3</td><td>Crosswalk</td><td style="background: #c88080"></td></tr>
        <tr><td>4</td><td>Curb</td><td style="background: #969696"></td></tr>
        <tr><td>5</td><td>Sidewalk</td><td style="background: #f423e8"></td></tr>
        <tr><td>6</td><td>Traffic Light</td><td style="background: #faaa1e"></td></tr>
        <tr><td>7</td><td>Traffic Sign</td><td style="background: #ffff00"></td></tr>
        <tr><td>8</td><td>Person</td><td style="background: #ff0000"></td></tr>
        <tr><td>9</td><td>Bicyclist</td><td style="background: #969664"></td></tr>
        <tr><td>10</td><td>Motorcyclist</td><td style="background: #143264"></td></tr>
        <tr><td>11</td><td>Bicycle</td><td style="background: #770b20"></td></tr>
        <tr><td>12</td><td>Bus</td><td style="background: #ff0f93"></td></tr>
        <tr><td>13</td><td>Car</td><td style="background: #00ff8e"></td></tr>
        <tr><td>14</td><td>Motorcycle</td><td style="background: #0000e6"></td></tr>
        <tr><td>15</td><td>Truck</td><td style="background: #4b0aaa"></td></tr>
    </tbody>
</table>

# Datasets
The AI has been trained using a mix of those two datasets :
1. [A2D2 of Audi](https://www.a2d2.audi/a2d2/en.html) 
2. [Mapillary Vistas](https://www.mapillary.com/dataset/vistas)


# Tools
List of tools I used :
1. [Keras](https://keras.io/)
2. [OpenCV](https://opencv.org/)
3. [Weights & Biases](https://wandb.ai/)