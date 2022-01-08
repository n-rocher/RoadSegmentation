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

<table class="categories">
    <thead>
        <tr>
            <th>#</th>
            <th>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Name&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>
            <th>Color</th>
        </tr>
    </thead>
    <tbody>
        <tr><td>1</td><td>Road</td><td><img src="https://via.placeholder.com/70x35/4b4b4b/000000?text=+"/></td></tr>
        <tr><td>2</td><td>Lane</td><td><img src="https://via.placeholder.com/70x35/ffffff/000000?text=+"/></td></tr>
        <tr><td>3</td><td>Crosswalk</td><td><img src="https://via.placeholder.com/70x35/c88080/000000?text=+"/></td></tr>
        <tr><td>4</td><td>Curb</td><td><img src="https://via.placeholder.com/70x35/969696/000000?text=+"/></td></tr>
        <tr><td>5</td><td>Sidewalk</td><td><img src="https://via.placeholder.com/70x35/f423e8/000000?text=+"/></td></tr>
        <tr><td>6</td><td>Traffic Light</td><td><img src="https://via.placeholder.com/70x35/faaa1e/000000?text=+"/></td></tr>
        <tr><td>7</td><td>Traffic Sign</td><td><img src="https://via.placeholder.com/70x35/ffff00/000000?text=+"/></td></tr>
        <tr><td>8</td><td>Person</td><td><img src="https://via.placeholder.com/70x35/ff0000/000000?text=+"/></td></tr>
        <tr><td>9</td><td>Bicycle</td><td><img src="https://via.placeholder.com/70x35/582900/000000?text=+"/></td></tr>
        <tr><td>10</td><td>Bus</td><td><img src="https://via.placeholder.com/70x35/ff0f93/000000?text=+"/></td></tr>
        <tr><td>11</td><td>Car</td><td><img src="https://via.placeholder.com/70x35/00ff8e/000000?text=+"/></td></tr>
        <tr><td>12</td><td>Motorcycle</td><td><img src="https://via.placeholder.com/70x35/0000e6/000000?text=+"/></td></tr>
        <tr><td>13</td><td>Truck</td><td><img src="https://via.placeholder.com/70x35/4b0aaa/000000?text=+"/></td></tr>
        <tr><td>14</td><td>Sky</td><td><img src="https://via.placeholder.com/70x35/87ceff/000000?text=+"/></td></tr>
        <tr><td>15</td><td>Nature</td><td><img src="https://via.placeholder.com/70x35/6b8e23/000000?text=+"/></td></tr>
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