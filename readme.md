# Multimedia VectorView

A simple python project that calculates and visualizes any kind of video's motion vectors.

## Motion estimation tab

![image](https://github.com/Darakuu/Multimedia-VectorView/assets/32675220/6ada5b52-8cb0-4995-ab5d-29bd0f254c73)

Load up a video, choose the algorithm to be used (and optionally, change their parameters), and watch the magic happen. You can stop the video at any time.
Playback can be stopped / resumed.

## Tracking tab

![image](https://github.com/Darakuu/Multimedia-VectorView/assets/32675220/656bd849-84c6-4b16-ba6f-b024255d3704)

The tracking tab is similar. Load a video, draw a bounding box over the area you wish to track, and the program will do its job.
You can stop the tracking to redraw the bounding box in case it loses track of the target, and then resume the playback.

Press ESC to quit the application. Currently, this is the only way to load a new video.

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

- [ ] Anti-Jitter filter (FPS or MVI)
- [ ] File restructure
- [x] Improve code readability
- [ ] Add a method to reset the loaded video.
- [ ] better openCV2 usage

- [ ] ~~FAST~~ - won't do, already implemented in ORB

## Instructions

Subject to change.

0. Python version used: `3.12.x`
1. Create a .venv virtual environment and install the requirements using pip, with the constraints: `pip install -r .\requirements.txt`
2. Run `source/main_window.py` and you're  good to go!

### Troubleshooting

- Optional step: Install the **Standard** K-Lite codecs: [K-Lite Codecs Download](https://www.codecguide.com/download_kl.htm)

This step is only needed if you have trouble in loading or playing the video.
