# FastGeodis: Fast Generalised Geodesic Distance Transform

CPU (OpenMP) and GPU (CUDA) implementation of Generalised Geodesic Distance Transform in PyTorch for 2D and 3D input data based on parallelisable raster scan ideas from [1, 2].


| 2D images, 1 of 4 passes | 3D volumes, 1 of 6 passes  |
|-------------------|-------------------------|
| <img src="figures/FastGeodis2D.png?raw=true" width="300" /> | <img src="figures/FastGeodis3D.png?raw=true" width="300" /> |


The above raster scan method can be parallelised for each row/plane on an available device (CPU or GPU). This leads to significant speed up as compared to existing non-parallelised raster scan implementations (e.g. [https://github.com/taigw/GeodisTK](https://github.com/taigw/GeodisTK)). 

## Installation instructions
The provided package can be installed using:

`pip install git+https://github.com/masadcv/FastGeodis`

or

TODO:
`pip install FastGeodis`

If you use this code, then please cite our paper: TODO

## Unit Tests
A number of unittests are provided, which can be run as:

`python -m unittest`


## Example usage

### Fast Geodesic Distance Transform
The following demonstrates a simple example showing FastGeodis usage:
```
device = "cuda" if torch.cuda.is_available else "cpu"
image = np.asarray(Image.open("data/img2d.png"), np.float32)

image_pt = torch.from_numpy(image).unsqueeze_(0).unsqueeze_(0)
image_pt = image_pt.to(device)
mask_pt = torch.ones_like(image_pt)
mask_pt[..., 100, 100] = 0

v = 1e10
lamb = 1.0
iterations = 2
geodesic_dist = FastGeodis.generalised_geodesic2d(
    image_pt, mask_pt, v, lamb, 1 - lamb, iterations
)
geodesic_dist = np.squeeze(geodesic_dist.cpu().numpy())
```

For more usage examples see:
- **Simple 2D Geodesic Distance example**: [`samples/simpledemo2d.py`](./samples/simpledemo2d.py) 
- **2D Geodesic Distance**: [`samples/demo2d.py`](./samples/demo2d.py) 
- **3D Geodesic Distance**: [`samples/demo3d.py`](./samples/demo3d.py)
- **2D GSF Segmentation Smoothing**: [`samples/demoGSF2d_SmoothingSegExample.ipynb`](./samples/demoGSF2d_SmoothingSegExample.ipynb)
 

## Comparison of Execution Time and Accuracy
FastGeodis (CPU/GPU) is compared with existing GeodisTK ([https://github.com/taigw/GeodisTK](https://github.com/taigw/GeodisTK)) in terms of execution speed as well as accuracy.


### Execution Time
| 2D images | 3D volumes  |
|-------------------|-------------------------|
|<img src="figures/experiment_2d.png?raw=true" width="400" />|<img src="figures/experiment_3d.png?raw=true" width="400" />|

<br>

### Accuracy
2D case
| Qualitative Comparison | Quantitative (joint histogram)  |
|-------------------|-------------------------|
|<img src="figures/fast_marching_compare_2d.png?raw=true?raw=true" width="800" />  |<img src="figures/fast_marching_compare_2d_jointhist.png?raw=true" width="400" /> |

3D case

| Qualitative Comparison | Quantitative (joint histogram)  |
|-------------------|-------------------------|
| <img src="figures/fast_marching_compare_3d.png?raw=true" width="800" /> | <img src="figures/fast_marching_compare_3d_jointhist.png?raw=true" width="400" /> |

## References
- [1] Criminisi, Antonio, Toby Sharp, and Khan Siddiqui. "Interactive Geodesic Segmentation of n-Dimensional Medical Images on the Graphics Processor."

- [2] Weber, Ofir, et al. "Parallel algorithms for approximation of distance maps on parametric surfaces." ACM Transactions on Graphics (TOG), (2008).

- [3] GeodisTK: [https://github.com/taigw/GeodisTK](https://github.com/taigw/GeodisTK)
