# MRacing-Mapping

For MRacing people:
You will need an official YOLOv7 file hierarchy set up, then the directories in pipeline will make sense. All jsons are stored in runs, videos are output to arbitrarty directory declared in Plotter.
detect-store-bboxes.py works the same as regular detect.py for YOLOv7, but instead of outputting a video file it outputs a json for each frame storing bbox locations.
pipeline.py is the main thing you want to run
PnP transforms bbox into overhead projection
Plotter makes videos out of points, lines, boxes etc
Velocity is testing to see if integrating delta-position and delta-rotation is enough to build a map (probably not), in this, you will find circle fitting for track radius estimation (when combined with velocity, you can get absolute rotation of car, but over a track loop the drift is sometimes over 90 deg so not good. need IMU data)
Boundaries takes points and makes boundary lines
The rest of the stuff in here is experimental, it might not make sense to you because it doenst make sense to me.
