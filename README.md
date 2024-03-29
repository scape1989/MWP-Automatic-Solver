![University of Colorado Colorado Springs](https://github.com/kadengriffith/MWP-Automatic-Solver/blob/master/publication/UCCS.svg)

# MWP Automatic Solver

---

This repo is the home to an automatic solver for math word problems using the 2017 Transformer model. This research was conducted at the University of Colorado Colorado Springs by Kaden Griffith and Jugal Kalita over the Summer in 2019. This research was a part of an undergraduate program through the National Science Foundation. This Transformer arrangement translates a word problem into a useable expression. We did not provide comprehensive code to solve the output, but you can use our _EquationConverter_ class to solve infix expressions via the sympy package. Our best model surpassed state-of-the-art, receiving an average of 86.7% accuracy among all test sets.

## Quickstart Guide

---

This quickstart is aimed to help you create your very own question to equation translator! Our configuration shown here is the model which performed the best among our tests. The translations are simple, so don't expect this to finish your calculus homework.

#### Step 1

---

For consistency, we provide a ready-to-use Docker image in this repo. To download all necessary packages and a version of TensorFlow that uses GPU configurations, follow this step. If you do not have Docker installed, you're going to want to install that before proceeding here.

Build the Docker image by:

```
chmod a+x docker-build && ./docker-build
```

This command builds the image, and start up a bash environment that we can train and test within. Use the following command after you exit the container and wish to start it again.

```
./run
```

The script above is set to mount the working directory and uses GPU0 on your system. Please alter the script to utilize more GPUs if you want to speed up the training process (i.e., _NVIDIA_VISIBLE_DEVICES=0,1_ to use 2 GPUs).

#### Step 2

---

So that we don't require a large download to use this software, compilers and generators exist that need to be used before training your model. Run the following command to both generate 50000 problems using our problem generator and compile the training and test sets we used in our work. We did not end up using the 50000 generated problems in our publication, but they're fun to train with and allow for custom applications if you want to use this code for something else.

```
./make-data
```

Data collected is from [AI2 Arithmetic Questions](https://allenai.org/data/data-all.html), [Common Core](https://cogcomp.org/page/resource_view/98), [Illinois](https://cogcomp.org/page/resource_view/98), and [MaWPS](http://lang.ee.washington.edu/MAWPS/). For more details on these sets, please refer to their authors.

All MWPs in the testing and training data are similar to the following:

```
There are 37 short bushes and 30 tall trees currently in the park. Park workers will plant 20 short bushes today. How many short bushes will the park have when the workers are finished?
57
X = 37 + 20
```

#### Step 3

---

Our approach has tested various vanilla Transformer models. Our most successful model is speedy and uses two Transformer layers.

The trainable model code lives in the translator.py file. This model is similar to the Transformer example discussed in the [TensorFlow online tutorial](https://www.tensorflow.org/beta/tutorials/text/transformer).

To use this model create a config JSON file like this:

```
{
 "dataset": "train_all_prefix.pickle",
 "test": "prefix",
 "pretrain": false,
 "seed": 6271996,
 "model": false,
 "layers": 2,
 "heads": 8,
 "d_model": 256,
 "dff": 1024,
 "lr": "scheduled",
 "dropout": 0.1,
 "epochs": 300,
 "batch": 128,
 "beta_1": 0.95,
 "beta_2": 0.99
}

// output to config.json
```

#### Step 4

---

From this point, you can run the following command in the container.

```
python translator.py config.json 0.0001

// 0.0001 <- Training will stop if loss is below this value.
// Set it low to complete your trial fully.
// Default (no argument) is 0.001
```

Alternatively, you can run the _trial_ script, which iterates over all of the config files found in the root directory and completes all epoch iterations you specify.

#### Step 5

---

After training, the program saves your model, and your configuration file is updated to refer to this model. You can train from the checkpoint generated by repeating step 4.

#### Tips

---

- Once trained, setting _epochs_ to 0 skips training if you wish to retest your model. This mode also enables the command line input testing. The translator works well when fed data from the training sets, but highly unique user input will most likely not result in the best output equations.
- To pre-train your model on IMDb reviews, enable the _pretrain_ setting before training on MWP data.

---

If you wish to see all of the numbers our models made, check out the publication folder. There are templated configuration files located there as well for examples of different configurations that were proven to work.

We hope that your interest in math word problems has increased, and encourage any suggestions for future work or bug fixes.

Happy solving!
