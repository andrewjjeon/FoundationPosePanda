# Evaluating Foundation Model Robot Pose Estimation with Synthetic Data Generation
Please see my project page for a full description of what exactly I did: https://andrewjjeon.github.io/projects/fposepanda/

## FoundationPose Setup

Download Model Weights and Demo Data
1) Download all network weights from [here](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing) and put them under the folder `weights/`. For the refiner, you will need `2023-10-28-18-33-37`. For scorer, you will need `2024-01-11-20-02-45`.

1) If you want to run FoundationPose on some of the original demo data from the Nvidia team: [Download demo data](https://drive.google.com/drive/folders/1pRyFmxYXmAnpku7nGRioZaKrVJtIsroP?usp=sharing) and extract them under the folder `demo_data/`. 

Environment Setup: Docker
  ```
  cd docker/
  docker pull wenbowen123/foundationpose && docker tag wenbowen123/foundationpose foundationpose  # Or to build from scratch: docker build --network host -t foundationpose .
  bash docker/run_container.sh
  ```
If it's the first time you launch the container, you need to build extensions. Run this command *inside* the Docker container.
```
bash build_all.sh
```

Later you can execute into the container without re-build.
```
docker exec -it foundationpose bash
```

Demo

First, try running on the given demo data from the original team (mustard bottle and driller) to learn how to run FoundationPose.

```
python run_demo.py
```

<br>

## Synthetic Data Generation and Evaluation (What I did)

Inside the FoundationPose Docker Container, install Pybullet via pip or another module to run my code to generate synthetic robot data. You can switch out the (.urdf) and (.obj) files with other robots, and this code should work with minimal tuning.
```
pip install pybullet
```


Next, go to /demo_data and run...

If you want to render a whole robot urdf inside pybullet and get images(rgb, mask, depth) and annotations for that:
```
python urdf_render.py
```

Along with the synthetic data, urdf_render.py will also generate ground truth pose annotations in hand2cam_poses.npy inside your synth directory.
Now, run FoundationPose again, after you change the data paths in run_demo.py to our synth directory.
```
python run_demo.py
```

Finally, Take the predicted pose matrix from FoundationPose and put it into evaluation.py which will evaluate against the ground truth pose annotations we generated above to get translation and rotation error for foundationpose vs ground truth!
```
python evaluation.py
```