## Multi-object tracking based on PP-YOLO Tiny and DSST algorithm

### Summary

This project takes the object detection data of manually labeled [HeLa cells](https://aistudio.baidu.com/aistudio/datasetdetail/107056) (origin: http://celltrackingchallenge.net/2d-datasets/) as an example. By training the [pp-yolo tiny object detector](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo) of the cell, the [DSST algorithm](http://dx.doi.org/10.5244/C.28.65) general object tracker is constructed, use the Hungarian algorithm to realize the association of the same object in the front and rear frames to realize the multi-object tracking of the cell.

If necessary, you can replace your own object detector to achieve multi-object tracking.

### Notebook

- AI Studio Project: [https://aistudio.baidu.com/aistudio/projectdetail/2607915](https://aistudio.baidu.com/aistudio/projectdetail/2607915)

### Reference

1. [PaddleX: PaddlePaddle End-to-End Development Toolkit](https://github.com/PaddlePaddle/PaddleX/tree/release/2.0.0)
2. [DLib: A toolkit for making real world machine learning and data analysis applications in C++](https://github.com/davisking/dlib/releases/tag/v19.22)
