See statistics.txt file for what the binaries here contain.

Data collected thus far is from [AI2 Arithmetic Questions](https://allenai.org/data/data-all.html), [Dolphin18k](https://www.microsoft.com/en-us/research/wp-content/uploads/2015/08/dolphin18k.pdf), and [MaWPS](http://lang.ee.washington.edu/MAWPS/). Data has been compiled to respect only single variable algebra questions. Hopefully multi-var and more calculus-based work will follow. Further work to transform the data to generalize more consistently is being researched currently.

More datasets will be added to create a more diverse set compared to the data that has recently proven to be less-than-promising. So far, this repo has collected 3688 math word problems.

All questions in the testing and training data are similar to the following:
[('question', 'suppose your club is selling candles to raise money. it costs $100 to rent a booth from which to sell the candles. if the candles cost your club $1 each and are sold for $5 each, how many candles must be sold to equal your expenses?\n'), ('answer', '25'), ('equation', 'unkn: m,0=5*m-(100+m)')]
-> Source: Dolphin18k

To compile the data to binaries use
```
python3 util/create_data.py
```

To generate problems for training
```
python3 util/generator.py
```
