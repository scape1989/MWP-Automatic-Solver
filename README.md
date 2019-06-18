An automatic solver for math word problems. Research conducted at the University of Colorado Colorado Springs.

For journal updates check out [The Clipping Point Project](https://theclippingpointproject.com). Searching for the title "Solving Math Word Problems Automatically" will bring up all related articles. The site is still being prettified, but it's there.

Warning: This is in very early development.

The plan now is to use a vanilla Transformer model and see how accurate we can get. I'll probably make a problem generator that will allow for better training since Transformers need a lot of data to work. For an example run build the docker container here, and run the transformer_translation.py. Instructions for container building is below. I hope to bring back BERT... but I can only find a way to classify with the available model. Looking into it...

Data collected thus far is from [AI2 Arithmetic Questions](https://allenai.org/data/data-all.html), [Dolphin18k](https://www.microsoft.com/en-us/research/wp-content/uploads/2015/08/dolphin18k.pdf), and [MaWPS](http://lang.ee.washington.edu/MAWPS/). Data has been compiled to respect only single variable algebra questions. Hopefully multi-var and more calculus-based work will follow. Further work to transform the data to generalize more consistently is being researched currently.

More datasets will be added to create a more diverse set compared to the data that has recently proven to be less-than-promising. So far, this repo has collected 4197 math word problems.

All questions in the testing and training data are similar to the following:
[('question', 'suppose your club is selling candles to raise money. it costs $100 to rent a booth from which to sell the candles. if the candles cost your club $1 each and are sold for $5 each, how many candles must be sold to equal your expenses?\n'), ('answer', '25'), ('equation', 'unkn: m,0=5*m-(100+m)')]
-> Source: Dolphin18k

This repo contains a few examples for transformer general use. Looking into MT-DNN...

To run interactively with GPU support
Replace 'containername' with whatever you want
```
nvidia-docker build -t containername . && nvidia-docker run -it --rm --runtime=nvidia containername bash
```

To run the translation example do:
```
cd mwp && python transformer_translation.py
```
NOTE: If the container is not mounted to a volume, you will lose the checkpointed model after training.
Training the Transformer took around an hour for 20 epochs using 4 Nvidia GTX GeForce 1080 Ti cards.
My final running accuracy was 0.3422, which is worse than the recorded accuracy found [here](https://www.tensorflow.org/beta/tutorials/text/transformer).

To mount a volume after building do:
```
nvidia-docker build -t containername .
nvidia-docker run --rm -v ${PWD}:/mwp -it --runtime=nvidia containername bash
```
Then your checkpoint should be saved to the host machine.

I'm working to get the MWP data into the correct format for translation. For now this is just a straight up rip off of [Google's Colab](https://www.tensorflow.org/beta/tutorials/text/transformer). Gotta start somewhere...