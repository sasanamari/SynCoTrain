
Engine run is terminating due to exception: Boolean value of Tensor with more than one value is ambiguous             
Engine run is terminating due to exception: Boolean value of Tensor with more than one value is ambiguous
Traceback (most recent call last):
  File "/home/samariam/projects/chemheuristics/alignn/PU_alignn.py", line 85, in <module>
    train_for_folder(
  File "/home/samariam/projects/chemheuristics/alignn/alignn_setup.py", line 209, in train_for_folder
    train_dgl(
  File "/home/samariam/projects/chemheuristics/alignn/myTrain.py", line 912, in train_dgl
    trainer.run(train_loader, max_epochs=config.epochs)
  File "/home/samariam/anaconda3/envs/alignn/lib/python3.8/site-packages/pytorch_ignite-0.5.0.dev20221024-py3.8.egg/ignite/engine/engine.py", line 900, in run
    return self._internal_run()
  File "/home/samariam/anaconda3/envs/alignn/lib/python3.8/site-packages/pytorch_ignite-0.5.0.dev20221024-py3.8.egg/ignite/engine/engine.py", line 943, in _internal_run
    return next(self._internal_run_generator)
  File "/home/samariam/anaconda3/envs/alignn/lib/python3.8/site-packages/pytorch_ignite-0.5.0.dev20221024-py3.8.egg/ignite/engine/engine.py", line 1001, in _internal_run_as_gen
    self._handle_exception(e)
  File "/home/samariam/anaconda3/envs/alignn/lib/python3.8/site-packages/pytorch_ignite-0.5.0.dev20221024-py3.8.egg/ignite/engine/engine.py", line 639, in _handle_exception
    raise e
  File "/home/samariam/anaconda3/envs/alignn/lib/python3.8/site-packages/pytorch_ignite-0.5.0.dev20221024-py3.8.egg/ignite/engine/engine.py", line 973, in _internal_run_as_gen
    self._fire_event(Events.EPOCH_COMPLETED)
  File "/home/samariam/anaconda3/envs/alignn/lib/python3.8/site-packages/pytorch_ignite-0.5.0.dev20221024-py3.8.egg/ignite/engine/engine.py", line 426, in _fire_event
    func(*first, *(event_args + others), **kwargs)
  File "/home/samariam/projects/chemheuristics/alignn/myTrain.py", line 830, in log_results
    evaluator.run(val_loader)
  File "/home/samariam/anaconda3/envs/alignn/lib/python3.8/site-packages/pytorch_ignite-0.5.0.dev20221024-py3.8.egg/ignite/engine/engine.py", line 900, in run
    return self._internal_run()
  File "/home/samariam/anaconda3/envs/alignn/lib/python3.8/site-packages/pytorch_ignite-0.5.0.dev20221024-py3.8.egg/ignite/engine/engine.py", line 943, in _internal_run
    return next(self._internal_run_generator)
  File "/home/samariam/anaconda3/envs/alignn/lib/python3.8/site-packages/pytorch_ignite-0.5.0.dev20221024-py3.8.egg/ignite/engine/engine.py", line 1001, in _internal_run_as_gen
    self._handle_exception(e)
  File "/home/samariam/anaconda3/envs/alignn/lib/python3.8/site-packages/pytorch_ignite-0.5.0.dev20221024-py3.8.egg/ignite/engine/engine.py", line 639, in _handle_exception
    raise e
  File "/home/samariam/anaconda3/envs/alignn/lib/python3.8/site-packages/pytorch_ignite-0.5.0.dev20221024-py3.8.egg/ignite/engine/engine.py", line 973, in _internal_run_as_gen
    self._fire_event(Events.EPOCH_COMPLETED)
  File "/home/samariam/anaconda3/envs/alignn/lib/python3.8/site-packages/pytorch_ignite-0.5.0.dev20221024-py3.8.egg/ignite/engine/engine.py", line 426, in _fire_event
    func(*first, *(event_args + others), **kwargs)
  File "/home/samariam/anaconda3/envs/alignn/lib/python3.8/site-packages/pytorch_ignite-0.5.0.dev20221024-py3.8.egg/ignite/handlers/early_stopping.py", line 80, in __call__
    elif score <= self.best_score + self.min_delta:
RuntimeError: Boolean value of Tensor with more than one value is ambiguous
(alignn) [samariam@sv2112 chemheuristics]$ 
=================================================================

Best score in now  tensor([0., 1.], dtype=torch.float64)

y = tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0,\n        0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0,\n        1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0,\n        1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1,\n        0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0,\n        0, 1, 1, 1, 0, 0, 1, 1], device='cuda:0')
y_pred = tensor([[0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.],\n        [0., 1.]], device='cuda:0')
y_pred_before = tensor([[-1.0710, -0.4196],\n        [-1.1816, -0.3664],\n        [-1.0679, -0.4212],\n        [-0.9744, -0.4739],\n        [-1.1906, -0.3624],\n        [-1.1205, -0.3947],\n        [-1.1034, -0.4031],\n        [-1.1517, -0.3799],\n        [-1.2079, -0.3550],\n        [-1.1474, -0.3819],\n        [-1.0812, -0.4143],\n        [-1.1180, -0.3959],\n        [-1.2516, -0.3369],\n        [-1.1972, -0.3596],\n        [-1.2142, -0.3523],\n        [-1.2327, -0.3446],\n        [-1.2074, -0.3552],\n        [-1.2135, -0.3526],\n        [-1.2627, -0.3325],\n        [-1.0922, -0.4087],\n        [-1.2830, -0.3246],\n        [-1.3457, -0.3016],\n        [-1.4336, -0.2724],\n        [-1.2112, -0.3536],\n        [-1.0707, -0.4197],\n        [-0.9325, -0.5002],\n        [-1.0155, -0.4498],\n        [-1.1880, -0.3636],\n        [-1.2396, -0.3418],\n        [-1.0018, -0.4576],\n        [-1.1959, -0.3602],\n        [-1.1191, -0.3954],\n        [-1.2871, -0.3231],\n        [-1.1058, -0.4019],\n        [-1.1928, -0.3615],\n        [-1.1425, -0.3842],\n        [-1.0018, -0.4576],\n        [-1.0983, -0.4056],\n        [-1.0625, -0.4240],\n        [-1.0832, -0.4132],\n        [-1.2550, -0.3356],\n        [-1.2096, -0.3543],\n        [-1.3096, -0.3146],\n        [-1.1192, -0.3953],\n        [-0.9947, -0.4618],\n        [-1.3454, -0.3017],\n        [-0.9775, -0.4720],\n        [-0.9550, -0.4858],\n        [-1.0312, -0.4409],\n        [-1.2066, -0.3556],\n        [-1.1089, -0.4003],\n        [-1.0178, -0.4485],\n        [-0.8339, -0.5698],\n        [-0.8681, -0.5443],\n        [-1.1314, -0.3895],\n        [-1.1880, -0.3636],\n        [-1.0988, -0.4054],\n        [-1.0439, -0.4340],\n        [-1.2121, -0.3532],\n        [-1.0074, -0.4544],\n        [-1.0208, -0.4468],\n        [-1.1492, -0.3811],\n        [-1.1756, -0.3691],\n        [-1.1080, -0.4008],\n        [-1.1214, -0.3943],\n        [-1.1902, -0.3626],\n        [-1.0141, -0.4505],\n        [-0.9641, -0.4802],\n        [-1.2190, -0.3503],\n        [-1.3657, -0.2947],\n        [-1.1918, -0.3620],\n        [-1.3540, -0.2987],\n        [-0.9506, -0.4886],\n        [-1.1251, -0.3925],\n        [-1.0480, -0.4318],\n        [-1.3055, -0.3161],\n        [-1.1042, -0.4027],\n        [-1.1754, -0.3692],\n        [-1.4332, -0.2725],\n        [-1.2163, -0.3514],\n        [-1.0341, -0.4393],\n        [-1.0636, -0.4235],\n        [-1.5139, -0.2485],\n        [-1.2091, -0.3545],\n        [-1.0604, -0.4252],\n        [-1.1403, -0.3853],\n        [-1.2740, -0.3281],\n        [-1.0723, -0.4189],\n        [-1.0825, -0.4136],\n        [-1.1985, -0.3590],\n        [-1.2281, -0.3465],\n        [-1.2710, -0.3293],\n        [-1.1521, -0.3798],\n        [-1.1693, -0.3719],\n        [-1.2476, -0.3385],\n        [-1.2195, -0.3501],\n        [-1.2005, -0.3582],\n        [-1.1961, -0.3601],\n        [-1.1393, -0.3857],\n        [-1.2817, -0.3251],\n        [-1.2388, -0.3421],\n        [-1.0607, -0.4250],\n        [-1.0796, -0.4151],\n        [-1.1491, -0.3811],\n        [-1.4367, -0.2714],\n        [-1.1114, -0.3991],\n        [-0.9852, -0.4674],\n        [-1.6919, -0.2035],\n        [-1.0428, -0.4346],\n        [-0.9861, -0.4669],\n        [-1.1525, -0.3796],\n        [-0.9978, -0.4600],\n        [-1.1013, -0.4041],\n        [-1.3100, -0.3145],\n        [-1.1165, -0.3967],\n        [-1.0879, -0.4108],\n        [-1.0380, -0.4372],\n        [-1.1299, -0.3902],\n        [-1.2879, -0.3227],\n        [-1.0457, -0.4330],\n        [-0.9960, -0.4610],\n        [-1.0696, -0.4203],\n        [-0.9836, -0.4683],\n        [-1.2112, -0.3536],\n        [-1.1346, -0.3880],\n        [-1.0737, -0.4182],\n        [-1.0161, -0.4494],\n        [-1.2068, -0.3555]], device='cuda:0')
torch.round(torch.exp(p))

my_metrics
'accuracy'
engine.state.metrics["accuracy"]
0.55078125
engine.state.metrics["recall"]
tensor([0.1641, 0.9375], dtype=torch.float64)
# So for some reason, recall gives us the recall for both classes.
# The class labels seem to go with index.

Even after setting up recall[1] as the score function, earlystopping is always triggiered
after patiance number of epochs, even when results keep improving.