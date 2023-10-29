# Unmanned-Boat-Online-Verification-System
This is a platform for validating unmanned surface vehicle (USV) navigation algorithms.

## Requirements
A [conda](https://conda.io/) environment named `SimSys` can be created and activated with:

```bash
conda env create -f environment.yaml
conda activate SimSys
```

## Usage:  
1. install openai, numpy, matplotlib.  
2. replace the "config.json" file.  
3. python main.py  
4. Left click to set the starting point, right click to set the destination, and you can continuously set static obstacles in between. Press the "M" key to set dynamic obstacles. Press the "space" to start the pursuit.
5. The program automatically saves experimental result images. Images with the suffix "-F" indicate failure, while images with the suffix "-S" indicate success.

This is a beta version with many bugs. Feel free to commit and contribute.

## Acknowledgement
This project is kinda borrowed from [Wheeled-robot-path-planning-algorithm](https://github.com/Friedrich-M/Wheeled-robot-path-planning-algorithm). We sincerely thank the authors for open-sourcing!