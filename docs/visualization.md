## Visualization

### Visualization Script

Visualize the predictions and attention maps.

Before visualization, build directory named as `vis_output` under the project directory. Then run:

```
    python tools/visualize.py -c $PATH2CONFIG -w $PATH2WEIGHT
```

This will visualize both good and bad cases of the GazeFollow dataset. For each case, we visualize the original image with the person's head marked, 24 attention maps from each attention head, and the predictions.
