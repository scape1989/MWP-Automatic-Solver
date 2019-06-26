An automatic solver for math word problems. Research conducted at the University of Colorado Colorado Springs.

For journal updates check out [The Clipping Point Project](https://theclippingpointproject.com). Searching for the title "Solving Math Word Problems Automatically" will bring up all related articles. The site is still being prettified and finalized, but it's there.

Warning: This is in very early development and the Transformer performs very badly right now.

To start, you'll have to generate the data binaries by running the following. They're too large or else I'd just include them in the repo.
```
python3 util/generator.py && python3 util/create_data.py
```
This will make the files described in data/statistics.txt

The plan now is to use a vanilla Transformer model and see how accurate we can get. For an example run build the docker container here, and run the transformer_translation.py in examples. Instructions for container building is below. I hope to bring back BERT...

Data collected thus far is from [AI2 Arithmetic Questions](https://allenai.org/data/data-all.html), [Dolphin18k](https://www.microsoft.com/en-us/research/wp-content/uploads/2015/08/dolphin18k.pdf), and [MaWPS](http://lang.ee.washington.edu/MAWPS/). Data has been compiled to respect only single variable algebra questions. Hopefully multi-var and more calculus-based work will follow. Further work to transform the data to generalize more consistently is being researched currently.

More datasets will be added to create a more diverse set compared to the data that has recently proven to be less-than-promising. So far, this repo has collected ~4000 math word problems. That doesn't count the generated problems.

All questions in the testing and training data are similar to the following:
[('question', 'suppose your club is selling candles to raise money. it costs $100 to rent a booth from which to sell the candles. if the candles cost your club $1 each and are sold for $5 each, how many candles must be sold to equal your expenses?\n'), ('answer', '25'), ('equation', 'unkn: m,0=5*m-(100+m)')]
-> Source: Dolphin18k

This repo contains a few examples for transformer general use. Looking into MT-DNN...

To run interactively with GPU support
Replace 'containername' with whatever you want
```
nvidia-docker build -t containername . && nvidia-docker run -it --runtime=nvidia containername bash
```
Or just use the build shell script...

To run the translation example do:
```
cd mwp && python translation_models/transformer_translation.py
```
NOTE: If the container is not mounted to a volume, you will lose the checkpointed model after training.
Training the Transformer took around an hour for 20 epochs using 4 Nvidia GTX GeForce 1080 Ti cards.
My final running accuracy was 0.3422, which is worse than the recorded accuracy found [here](https://www.tensorflow.org/beta/tutorials/text/transformer).

To mount a volume after building do:
```
nvidia-docker build -t containername .
nvidia-docker run -it --runtime=nvidia containername bash
```
Or to limit available GPUs in the container
```
nvidia-docker run -u $(id -u):$(id -g) -v $(PWD):home/mwp -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 containername bash
```
Then your checkpoints, tensorboard data, etc, should be saved to the host machine.

Useful commands I always forget
```
watch -n 0.5 nvidia-smi
```

```
tmux new -s sessionname
tmux a -t sessionname
```
