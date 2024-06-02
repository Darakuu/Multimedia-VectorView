# Multimedia VectorView

A simple python project that calculates and visualizes any kind of video's motion vectors.

## Motion estimation tab

![image](https://github.com/Darakuu/Multimedia-VectorView/assets/32675220/444d83f6-3a2d-48ca-a827-3c08f9bdc99b)

Load up a video, choose the algorithm to be used (and optionally, change their parameters), and watch the magic happen. You can stop the video at any time.

## Tracking tab

![image](https://github.com/Darakuu/Multimedia-VectorView/assets/32675220/4623797c-d0a8-420e-b855-885885a9b5d1)

The tracking tab is similar. Load a video, draw a bounding box over the area you wish to track, and the program will do its job.

## Implementation stages (aka makeshift todo list)

In no particular order:

- [x] Repository initialization
- [x] Open video from file system
- [x] UI
- [x] EBMA implementation
- [x] MV on-top visualization
- [x] Bounding box drawing
- [x] BB-Based Motion tracking
- [x] EBMA alternatives
- [ ] Documentation
- [ ] Final Report

Extra:

- [ ] FAST
- [ ] File restructure
- [x] Improve code readability
- [ ] better openCV2 usage


## Instructions

Subject to change.

0. Python version used: `3.12.x`
1. Create a .venv virtual environment and install the requirements using pip, with the constraints: `pip install -r .\requirements.txt`
2. Install the **Standard** K-Lite codecs: [K-Lite Codecs Download](https://www.codecguide.com/download_kl.htm)
3. Run `source/main_window.py` and you're  good to go!
