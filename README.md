An automatic solver for math word problems using the 2017 Transformer model. Research conducted at the University of Colorado Colorado Springs in Summer 2019.

To start, you'll have to generate the data binaries by running the following.

```
chmod a+x make-data && ./make-data
```

Use the build shell script to build a container and then run it with your current user. The current directory is a mounted volume so the checkpoints and training logs are saved to the host machine. Then use the run script to start the container back up if you exit after building.

Build the Docker image by:

```
chmod a+x docker-build && ./docker-build
```

You can re-run the docker image with GPUs using:

```
./run
```

The approach now is to use a vanilla Transformer model and see how accurate we can get it. It's the example found [here](https://www.tensorflow.org/beta/tutorials/text/transformer).

Data collected thus far is from [AI2 Arithmetic Questions](https://allenai.org/data/data-all.html), [Common Core](https://cogcomp.org/page/resource_view/98), and [Illinois](https://cogcomp.org/page/resource_view/98). Data has been compiled to respect only single variable algebra questions. Hopefully multi-var and more calculus-based work will follow. Further work to transform the data to generalize more consistently is being researched currently.

More datasets will be added to create a more diverse set compared to the data that has recently proven to be less-than-promising. So far, this repo has collected ~4000 math word problems. That doesn't count the generated problems.

All questions in the testing and training data are similar to the following:

```
There are 37 short bushes and 30 tall trees currently in the park . Park workers will plant 20 short bushes today . How many short bushes will the park have when the workers are finished ?
57
X = 37 + 20
```

My trainable model is found in the translator.py file.

First create a config JSON file like this:

```
{
  "dataset": "train_infix.pickle",
  "test": false,
  "seed": 365420,
  "model": false,
  "dff": 2048,
  "layers": 6,
  "d_model": 512,
  "heads": 16,
  "lr": 0.0001,
  "dropout": 0.1,
  "epochs": 1,
  "batch": 64,
  "beta_1": 0.98,
  "beta_2": 0.99
}

// output to myconfig.json
```

Then you can run the trial script to iterate over all of your configuration files.

To run a sinlge trial you can use:

```
python translator.py myconfig.json
```

TODO:
[] Unsupervised training on language in general
[] Train on generated equations with words instead of numbers
