from ica.paraphraseator.T5 import T5

t5 = T5(accumulate_grad_batches=2, batch_size=4)
t5.train_model()
