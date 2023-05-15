import utils

model = utils.creat_model('show_alexnet', in_ch=1, pretrained=False)

trainer = utils.Trainer(model, dataset='fashion-mnist', batch_size=128, seed=0, lr=0.01, use_tb=True)

trainer.train(10)