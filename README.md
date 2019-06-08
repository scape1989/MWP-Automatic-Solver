An automatic solver for math word problems. Research conducted at the University of Colorado Colorado Springs.

Warning: This is in very early development.

Models relying on LSTM/GRU/RNN architectures will be proposed and tested to perform automatic arithmetic. In addition, the [MathDQN](https://github.com/uestc-db/DQN_Word_Problem_Solver) Deep Reinforcement Learning approach will be taken. Data collected thus far is from [AI2 Arithmetic Questions](https://allenai.org/data/data-all.html), [Dolphin18k](https://www.microsoft.com/en-us/research/wp-content/uploads/2015/08/dolphin18k.pdf), and [MaWPS](http://lang.ee.washington.edu/MAWPS/). Data has been compiled to respect only single variable algebra questions. Hopefully multi-var and more calculus-based work will follow. Further work to transform the data to generalize more consistently is being researched currently.

More datasets will be added to create a more diverse set compared to the data that has recently proven to be less-than-promising. So far, this repo has collected 4197 math word problems.

All questions in the testing and training data are similar to the following:
[('question', 'suppose your club is selling candles to raise money. it costs $100 to rent a booth from which to sell the candles. if the candles cost your club $1 each and are sold for $5 each, how many candles must be sold to equal your expenses?\n'), ('answer', '25'), ('equation', 'unkn: m,0=5*m-(100+m)')]
-> Source: Dolphin18k

Embedding of the data will be from the [BERT](https://arxiv.org/abs/1810.04805) system utilizing the Google pretrained model. With the advancements of word vectorization and less eradic data, improvements in speed and performance are expected.