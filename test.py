a = '''
 2023-03-07 23:23:53,388 | INFO | root   curriculum

 2023-03-07 23:23:54,427 | INFO | root   [['mammal', 'bird', 'device', 'container'], ['ungulate', 'rodent', 'primate', 'feline', 'canine'], ['game_bird', 'finch', 'wading_bird', 'other_oscine', 'other_aquatic_bird'], ['instrument', 'restraint', 'mechanism', 'musical_instrument', 'machine'], ['vessel', 'box', 'bag', 'self-propelled_vehicle', 'other_wheeled_vehicle'], ['hippopotamus', 'ox', 'hartebeest', 'impala', 'zebra'], ['guinea_pig', 'marmot', 'porcupine', 'hamster', 'beaver'], ['titi', 'capuchin', 'howler_monkey', 'patas', 'gibbon'], ['tiger_cat', 'tiger', 'persian_cat', 'cheetah', 'lion'], ['hyena', 'dhole', 'mexican_hairless', 'arctic_fox', 'timber_wolf'], ['ruffed_grouse', 'peacock', 'ptarmigan', 'partridge', 'quail'], ['goldfinch', 'junco', 'brambling', 'indigo_bunting', 'house_finch'], ['bustard', 'ruddy_turnstone', 'little_blue_heron', 'limpkin', 'spoonbill'], ['bulbul', 'jay', 'magpie', 'chickadee', 'water_ouzel'], ['goose', 'black_swan', 'european_gallinule', 'king_penguin', 'albatross'], ['sunglasses', 'cannon', 'rule', 'radio_telescope', 'guillotine'], ['buckle', 'padlock', 'hair_slide', 'safety_pin', 'muzzle'], ['paddlewheel', 'potters_wheel', 'puck', 'car_wheel', 'switch'], ['harp', 'sax', 'trombone', 'oboe', 'cornet'], ['chain_saw', 'cash_machine', 'abacus', 'harvester', 'desktop_computer'], ['mortar', 'ladle', 'tub', 'pitcher', 'beaker'], ['safe', 'pencil_box', 'mailbox', 'crate', 'chest'], ['backpack', 'sleeping_bag', 'mailbag', 'purse', 'plastic_bag'], ['streetcar', 'forklift', 'tank', 'tractor', 'recreational_vehicle'], ['barrow', 'freight_car', 'jinrikisha', 'motor_scooter', 'unicycle']]
[-53, -54, -81, -84]

 2023-03-07 23:23:59,546 | INFO | root   Begin task 0

 2023-03-07 23:24:00,130 | INFO | root   Now [500, 500, 500, 500] examplars per class.

 2023-03-07 23:24:02,985 | INFO | root   Step 0 weight decay 0.00050

 2023-03-07 23:24:05,263 | INFO | root   Begin evaluation: eval_after_decouple
[-84 -81 -54 -53]
[1250 1250 1250 1250]
/opt/app-root/lib64/python3.8/site-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /pytorch/c10/core/TensorImpl.h:1156.)
  return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)

 2023-03-07 23:24:14,654 | INFO | root   Evaluation eval_after_decouple, avg total: 0.858, finest avg: None with 0 classes, coarse avg: 0.858 with 4 classes, avg top0 total: None, aux_acc: 0
[-185, -186, -190, -285, -286]

 2023-03-07 23:24:26,464 | INFO | root   Begin task 1

 2023-03-07 23:24:27,544 | INFO | root   Now [250, 250, 250, 250, 250, 250, 250, 250] examplars per class.

 2023-03-07 23:24:28,207 | INFO | root   Step 1 weight decay 0.00050

 2023-03-07 23:24:28,704 | INFO | root   Begin evaluation: eval_after_decouple
[-286 -285 -190 -186 -185  -84  -81  -54]
[ 250  250  250  250  250 1250 1250 1250]

 2023-03-07 23:24:36,391 | INFO | root   Evaluation eval_after_decouple, avg total: 0.83, finest avg: None with 0 classes, coarse avg: 0.83 with 8 classes, avg top0 total: None, aux_acc: 0.7846000002861023
[-98, -192, -196, -450, -451]

 2023-03-07 23:24:55,102 | INFO | root   Begin task 2

 2023-03-07 23:24:55,469 | INFO | root   Now [166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166, 166] examplars per class.

 2023-03-07 23:24:56,496 | INFO | root   Step 2 weight decay 0.00050

 2023-03-07 23:24:57,007 | INFO | root   Begin evaluation: eval_after_decouple
[-451 -450 -286 -285 -196 -192 -190 -186 -185  -98  -84  -81]
[ 250  250  250  250  250  250  250  250  250  250 1250 1250]

 2023-03-07 23:25:05,270 | INFO | root   Evaluation eval_after_decouple, avg total: 0.812, finest avg: None with 0 classes, coarse avg: 0.812 with 12 classes, avg top0 total: None, aux_acc: 0.8322
[-143, -146, -149, -155, -156]

 2023-03-07 23:25:35,601 | INFO | root   Begin task 3

 2023-03-07 23:25:53,028 | INFO | root   Now [125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125] examplars per class.

 2023-03-07 23:25:59,855 | INFO | root   Step 3 weight decay 0.00050

 2023-03-07 23:26:02,257 | INFO | root   Begin evaluation: eval_after_decouple
[-451 -450 -286 -285 -196 -192 -190 -186 -185 -156 -155 -149 -146 -143
  -98  -84]
[ 250  250  250  250  250  250  250  250  250  250  250  250  250  250
  250 1250]

 2023-03-07 23:26:11,014 | INFO | root   Evaluation eval_after_decouple, avg total: 0.776, finest avg: None with 0 classes, coarse avg: 0.776 with 16 classes, avg top0 total: None, aux_acc: 0.6287999996185303
[-165, -166, -171, -278, -452]

 2023-03-07 23:26:38,326 | INFO | root   Begin task 4

 2023-03-07 23:26:39,036 | INFO | root   Now [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100] examplars per class.

 2023-03-07 23:26:40,902 | INFO | root   Step 4 weight decay 0.00050

 2023-03-07 23:26:41,477 | INFO | root   Begin evaluation: eval_after_decouple
[-452 -451 -450 -286 -285 -278 -196 -192 -190 -186 -185 -171 -166 -165
 -156 -155 -149 -146 -143  -98]
[250 250 250 250 250 250 250 250 250 250 250 250 250 250 250 250 250 250
 250 250]

 2023-03-07 23:26:51,480 | INFO | root   Evaluation eval_after_decouple, avg total: 0.768, finest avg: None with 0 classes, coarse avg: 0.768 with 20 classes, avg top0 total: None, aux_acc: 0.6883999996185303
[344, 345, 351, 352, 340]

 2023-03-07 23:26:57,105 | INFO | root   Begin task 5

 2023-03-07 23:26:57,508 | INFO | root   Now [83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83] examplars per class.

 2023-03-07 23:26:57,963 | INFO | root   Step 5 weight decay 0.00050

 2023-03-07 23:26:58,580 | INFO | root   Begin evaluation: eval_after_decouple
[-452 -451 -450 -286 -285 -278 -196 -192 -190 -186 -171 -166 -165 -156
 -155 -149 -146 -143  -98  340  344  345  351  352]
[250 250 250 250 250 250 250 250 250 250 250 250 250 250 250 250 250 250
 250  50  50  50  50  50]

 2023-03-07 23:27:09,765 | INFO | root   Evaluation eval_after_decouple, avg total: 0.764, finest avg: 0.864 with 5 classes, coarse avg: 0.759 with 19 classes, avg top0 total: None, aux_acc: 0.884
[338, 336, 334, 333, 337]

 2023-03-07 23:27:20,410 | INFO | root   Begin task 6

 2023-03-07 23:27:20,702 | INFO | root   Now [71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71] examplars per class.

 2023-03-07 23:27:21,670 | INFO | root   Step 6 weight decay 0.00050

 2023-03-07 23:27:22,470 | INFO | root   Begin evaluation: eval_after_decouple
[-452 -451 -450 -286 -285 -278 -196 -192 -190 -171 -166 -165 -156 -155
 -149 -146 -143  -98  333  334  336  337  338  340  344  345  351  352]
[250 250 250 250 250 250 250 250 250 250 250 250 250 250 250 250 250 250
  50  50  50  50  50  50  50  50  50  50]

 2023-03-07 23:27:34,673 | INFO | root   Evaluation eval_after_decouple, avg total: 0.761, finest avg: 0.85 with 10 classes, coarse avg: 0.751 with 18 classes, avg top0 total: None, aux_acc: 0.8907999993324279
[380, 378, 379, 371, 368]

 2023-03-07 23:27:54,230 | INFO | root   Begin task 7

 2023-03-07 23:27:54,602 | INFO | root   Now [62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62] examplars per class.

 2023-03-07 23:27:55,029 | INFO | root   Step 7 weight decay 0.00050

 2023-03-07 23:28:31,346 | INFO | root   Begin evaluation: eval_after_decouple
[-452 -451 -450 -286 -285 -278 -196 -192 -171 -166 -165 -156 -155 -149
 -146 -143  -98  333  334  336  337  338  340  344  345  351  352  368
  371  378  379  380]
[250 250 250 250 250 250 250 250 250 250 250 250 250 250 250 250 250  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50]

 2023-03-07 23:28:44,383 | INFO | root   Evaluation eval_after_decouple, avg total: 0.754, finest avg: 0.8227 with 15 classes, coarse avg: 0.742 with 17 classes, avg top0 total: None, aux_acc: 0.9316000002861023
[282, 292, 283, 293, 291]

 2023-03-07 23:28:51,270 | INFO | root   Begin task 8

 2023-03-07 23:28:51,624 | INFO | root   Now [55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55] examplars per class.

 2023-03-07 23:28:52,078 | INFO | root   Step 8 weight decay 0.00050

 2023-03-07 23:29:29,839 | INFO | root   Begin evaluation: eval_after_decouple
[-452 -451 -450 -286 -278 -196 -192 -171 -166 -165 -156 -155 -149 -146
 -143  -98  282  283  291  292  293  333  334  336  337  338  340  344
  345  351  352  368  371  378  379  380]
[250 250 250 250 250 250 250 250 250 250 250 250 250 250 250 250  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50]

 2023-03-07 23:29:43,938 | INFO | root   Evaluation eval_after_decouple, avg total: 0.752, finest avg: 0.815 with 20 classes, coarse avg: 0.736 with 16 classes, avg top0 total: None, aux_acc: 0.9358000008583068
[276, 274, 268, 279, 269]

 2023-03-07 23:29:54,533 | INFO | root   Begin task 9

 2023-03-07 23:29:55,282 | INFO | root   Now [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50] examplars per class.

 2023-03-07 23:29:55,715 | INFO | root   Step 9 weight decay 0.00050

 2023-03-07 23:30:30,331 | INFO | root   Begin evaluation: eval_after_decouple
[-452 -451 -450 -278 -196 -192 -171 -166 -165 -156 -155 -149 -146 -143
  -98  268  269  274  276  279  282  283  291  292  293  333  334  336
  337  338  340  344  345  351  352  368  371  378  379  380]
[250 250 250 250 250 250 250 250 250 250 250 250 250 250 250  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50]

 2023-03-07 23:30:45,284 | INFO | root   Evaluation eval_after_decouple, avg total: 0.752, finest avg: 0.8104 with 25 classes, coarse avg: 0.733 with 15 classes, avg top0 total: None, aux_acc: 0.9323999999046325
[82, 84, 81, 86, 85]

 2023-03-07 23:30:56,062 | INFO | root   Begin task 10

 2023-03-07 23:30:56,515 | INFO | root   Now [45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45] examplars per class.

 2023-03-07 23:30:56,962 | INFO | root   Step 10 weight decay 0.00050

 2023-03-07 23:31:40,674 | INFO | root   Begin evaluation: eval_after_decouple
[-452 -451 -450 -278 -196 -192 -171 -166 -165 -156 -155 -149 -146 -143
   81   82   84   85   86  268  269  274  276  279  282  283  291  292
  293  333  334  336  337  338  340  344  345  351  352  368  371  378
  379  380]
[250 250 250 250 250 250 250 250 250 250 250 250 250 250  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50]

 2023-03-07 23:32:01,753 | INFO | root   Evaluation eval_after_decouple, avg total: 0.747, finest avg: 0.8027 with 30 classes, coarse avg: 0.723 with 14 classes, avg top0 total: None, aux_acc: 0.8566000004768372
[11, 13, 10, 14, 12]

 2023-03-07 23:32:20,328 | INFO | root   Begin task 11

 2023-03-07 23:32:20,329 | INFO | root   Now [41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41] examplars per class.

 2023-03-07 23:32:20,432 | INFO | root   Step 11 weight decay 0.00050

 2023-03-07 23:33:10,117 | INFO | root   Begin evaluation: eval_after_decouple
[-452 -451 -450 -278 -196 -171 -166 -165 -156 -155 -149 -146 -143   10
   11   12   13   14   81   82   84   85   86  268  269  274  276  279
  282  283  291  292  293  333  334  336  337  338  340  344  345  351
  352  368  371  378  379  380]
[250 250 250 250 250 250 250 250 250 250 250 250 250  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50]

 2023-03-07 23:33:27,053 | INFO | root   Evaluation eval_after_decouple, avg total: 0.745, finest avg: 0.816 with 35 classes, coarse avg: 0.707 with 13 classes, avg top0 total: None, aux_acc: 0.9356000001907349
[138, 139, 131, 135, 129]

 2023-03-07 23:33:52,972 | INFO | root   Begin task 12

 2023-03-07 23:33:52,973 | INFO | root   Now [38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38] examplars per class.

 2023-03-07 23:33:53,084 | INFO | root   Step 12 weight decay 0.00050

 2023-03-07 23:34:47,557 | INFO | root   Begin evaluation: eval_after_decouple
[-452 -451 -450 -278 -171 -166 -165 -156 -155 -149 -146 -143   10   11
   12   13   14   81   82   84   85   86  129  131  135  138  139  268
  269  274  276  279  282  283  291  292  293  333  334  336  337  338
  340  344  345  351  352  368  371  378  379  380]
[250 250 250 250 250 250 250 250 250 250 250 250  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50]

 2023-03-07 23:35:05,480 | INFO | root   Evaluation eval_after_decouple, avg total: 0.744, finest avg: 0.8235 with 40 classes, coarse avg: 0.69 with 12 classes, avg top0 total: None, aux_acc: 0.9281999997138977
[16, 17, 18, 19, 20]

 2023-03-07 23:35:15,318 | INFO | root   Begin task 13

 2023-03-07 23:35:16,602 | INFO | root   Now [35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35] examplars per class.

 2023-03-07 23:35:18,174 | INFO | root   Step 13 weight decay 0.00050

 2023-03-07 23:36:17,536 | INFO | root   Begin evaluation: eval_after_decouple
[-452 -451 -278 -171 -166 -165 -156 -155 -149 -146 -143   10   11   12
   13   14   16   17   18   19   20   81   82   84   85   86  129  131
  135  138  139  268  269  274  276  279  282  283  291  292  293  333
  334  336  337  338  340  344  345  351  352  368  371  378  379  380]
[250 250 250 250 250 250 250 250 250 250 250  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50]

 2023-03-07 23:36:38,354 | INFO | root   Evaluation eval_after_decouple, avg total: 0.74, finest avg: 0.8124 with 45 classes, coarse avg: 0.68 with 11 classes, avg top0 total: None, aux_acc: 0.8899999997138978
[99, 100, 136, 145, 146]

 2023-03-07 23:36:53,957 | INFO | root   Begin task 14

 2023-03-07 23:36:54,429 | INFO | root   Now [33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33] examplars per class.

 2023-03-07 23:36:57,522 | INFO | root   Step 14 weight decay 0.00050

 2023-03-07 23:38:05,530 | INFO | root   Begin evaluation: eval_after_decouple
[-452 -278 -171 -166 -165 -156 -155 -149 -146 -143   10   11   12   13
   14   16   17   18   19   20   81   82   84   85   86   99  100  129
  131  135  136  138  139  145  146  268  269  274  276  279  282  283
  291  292  293  333  334  336  337  338  340  344  345  351  352  368
  371  378  379  380]
[250 250 250 250 250 250 250 250 250 250  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50]

 2023-03-07 23:38:26,836 | INFO | root   Evaluation eval_after_decouple, avg total: 0.737, finest avg: 0.8096 with 50 classes, coarse avg: 0.665 with 10 classes, avg top0 total: None, aux_acc: 0.9258000004768372
[837, 471, 769, 755, 583]

 2023-03-07 23:38:37,390 | INFO | root   Begin task 15

 2023-03-07 23:38:38,176 | INFO | root   Now [31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31] examplars per class.

 2023-03-07 23:38:39,390 | INFO | root   Step 15 weight decay 0.00050

 2023-03-07 23:39:44,756 | INFO | root   Begin evaluation: eval_after_decouple
[-452 -278 -171 -166 -165 -156 -155 -149 -146   10   11   12   13   14
   16   17   18   19   20   81   82   84   85   86   99  100  129  131
  135  136  138  139  145  146  268  269  274  276  279  282  283  291
  292  293  333  334  336  337  338  340  344  345  351  352  368  371
  378  379  380  471  583  755  769  837]
[250 250 250 250 250 250 250 250 250  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50]

 2023-03-07 23:40:09,066 | INFO | root   Evaluation eval_after_decouple, avg total: 0.73, finest avg: 0.7873 with 55 classes, coarse avg: 0.66 with 9 classes, avg top0 total: None, aux_acc: 0.6670000003814698
[464, 695, 584, 772, 676]

 2023-03-07 23:40:39,652 | INFO | root   Begin task 16

 2023-03-07 23:40:42,361 | INFO | root   Now [29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29] examplars per class.

 2023-03-07 23:40:43,472 | INFO | root   Step 16 weight decay 0.00050

 2023-03-07 23:42:06,338 | INFO | root   Begin evaluation: eval_after_decouple
[-452 -278 -171 -166 -165 -156 -155 -149   10   11   12   13   14   16
   17   18   19   20   81   82   84   85   86   99  100  129  131  135
  136  138  139  145  146  268  269  274  276  279  282  283  291  292
  293  333  334  336  337  338  340  344  345  351  352  368  371  378
  379  380  464  471  583  584  676  695  755  769  772  837]
[250 250 250 250 250 250 250 250  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50]

 2023-03-07 23:42:29,851 | INFO | root   Evaluation eval_after_decouple, avg total: 0.725, finest avg: 0.752 with 60 classes, coarse avg: 0.684 with 8 classes, avg top0 total: None, aux_acc: 0.7515999995231628
[694, 739, 746, 479, 844]

 2023-03-07 23:49:32,761 | INFO | root   Step 17 weight decay 0.00050

 2023-03-07 23:50:55,997 | INFO | root   Begin evaluation: eval_after_decouple
[-452 -278 -171 -166 -165 -156 -155   10   11   12   13   14   16   17
   18   19   20   81   82   84   85   86   99  100  129  131  135  136
  138  139  145  146  268  269  274  276  279  282  283  291  292  293
  333  334  336  337  338  340  344  345  351  352  368  371  378  379
  380  464  471  479  583  584  676  694  695  739  746  755  769  772
  837  844]
[250 250 250 250 250 250 250  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50]
/opt/app-root/lib64/python3.8/site-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /pytorch/c10/core/TensorImpl.h:1156.)
  return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)

 2023-03-07 23:51:19,371 | INFO | root   Evaluation eval_after_decouple, avg total: 0.724, finest avg: 0.7468 with 65 classes, coarse avg: 0.681 with 7 classes, avg top0 total: None, aux_acc: 0.7953999994277954
[594, 776, 875, 683, 513]

 2023-03-07 23:51:27,573 | INFO | root   Begin task 18

 2023-03-07 23:51:27,573 | INFO | root   Now [26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26] examplars per class.

 2023-03-07 23:51:27,695 | INFO | root   Step 18 weight decay 0.00050

 2023-03-07 23:52:40,541 | INFO | root   Begin evaluation: eval_after_decouple
[-452 -278 -171 -166 -165 -156   10   11   12   13   14   16   17   18
   19   20   81   82   84   85   86   99  100  129  131  135  136  138
  139  145  146  268  269  274  276  279  282  283  291  292  293  333
  334  336  337  338  340  344  345  351  352  368  371  378  379  380
  464  471  479  513  583  584  594  676  683  694  695  739  746  755
  769  772  776  837  844  875]
[250 250 250 250 250 250  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50]

 2023-03-07 23:53:05,249 | INFO | root   Evaluation eval_after_decouple, avg total: 0.715, finest avg: 0.7343 with 70 classes, coarse avg: 0.669 with 6 classes, avg top0 total: None, aux_acc: 0.8389999997138977
[491, 480, 398, 595, 527]

 2023-03-07 23:53:13,232 | INFO | root   Begin task 19

 2023-03-07 23:53:13,233 | INFO | root   Now [25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25] examplars per class.

 2023-03-07 23:53:13,352 | INFO | root   Step 19 weight decay 0.00050

 2023-03-07 23:54:24,998 | INFO | root   Begin evaluation: eval_after_decouple
[-452 -278 -171 -166 -165   10   11   12   13   14   16   17   18   19
   20   81   82   84   85   86   99  100  129  131  135  136  138  139
  145  146  268  269  274  276  279  282  283  291  292  293  333  334
  336  337  338  340  344  345  351  352  368  371  378  379  380  398
  464  471  479  480  491  513  527  583  584  594  595  676  683  694
  695  739  746  755  769  772  776  837  844  875]
[250 250 250 250 250  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50]

 2023-03-07 23:54:50,736 | INFO | root   Evaluation eval_after_decouple, avg total: 0.714, finest avg: 0.7312 with 75 classes, coarse avg: 0.661 with 5 classes, avg top0 total: None, aux_acc: 0.8091999995231628
[666, 618, 876, 725, 438]

 2023-03-07 23:55:25,069 | INFO | root   Begin task 20

 2023-03-07 23:55:25,069 | INFO | root   Now [23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23] examplars per class.

 2023-03-07 23:55:25,204 | INFO | root   Step 20 weight decay 0.00050

 2023-03-07 23:56:43,920 | INFO | root   Begin evaluation: eval_after_decouple
[-452 -278 -171 -166   10   11   12   13   14   16   17   18   19   20
   81   82   84   85   86   99  100  129  131  135  136  138  139  145
  146  268  269  274  276  279  282  283  291  292  293  333  334  336
  337  338  340  344  345  351  352  368  371  378  379  380  398  438
  464  471  479  480  491  513  527  583  584  594  595  618  666  676
  683  694  695  725  739  746  755  769  772  776  837  844  875  876]
[250 250 250 250  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50]

 2023-03-07 23:57:10,532 | INFO | root   Evaluation eval_after_decouple, avg total: 0.707, finest avg: 0.7148 with 80 classes, coarse avg: 0.677 with 4 classes, avg top0 total: None, aux_acc: 0.8452000004768372
[771, 709, 637, 519, 492]

 2023-03-08 00:03:14,491 | INFO | root   Begin task 21

 2023-03-08 00:03:14,492 | INFO | root   Now [22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22] examplars per class.

 2023-03-08 00:03:14,615 | INFO | root   Step 21 weight decay 0.00050

 2023-03-08 00:04:46,805 | INFO | root   Begin evaluation: eval_after_decouple
[-452 -278 -171   10   11   12   13   14   16   17   18   19   20   81
   82   84   85   86   99  100  129  131  135  136  138  139  145  146
  268  269  274  276  279  282  283  291  292  293  333  334  336  337
  338  340  344  345  351  352  368  371  378  379  380  398  438  464
  471  479  480  491  492  513  519  527  583  584  594  595  618  637
  666  676  683  694  695  709  725  739  746  755  769  771  772  776
  837  844  875  876]
[250 250 250  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50]
/opt/app-root/lib64/python3.8/site-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /pytorch/c10/core/TensorImpl.h:1156.)
  return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)

 2023-03-08 00:05:14,585 | INFO | root   Evaluation eval_after_decouple, avg total: 0.706, finest avg: 0.7071 with 85 classes, coarse avg: 0.7 with 3 classes, avg top0 total: None, aux_acc: 0.843400000667572
[414, 797, 636, 748, 728]

 2023-03-08 00:05:23,134 | INFO | root   Begin task 22

 2023-03-08 00:05:23,135 | INFO | root   Now [21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21] examplars per class.

 2023-03-08 00:05:23,279 | INFO | root   Step 22 weight decay 0.00050

 2023-03-08 00:06:54,703 | INFO | root   Begin evaluation: eval_after_decouple
[-452 -278   10   11   12   13   14   16   17   18   19   20   81   82
   84   85   86   99  100  129  131  135  136  138  139  145  146  268
  269  274  276  279  282  283  291  292  293  333  334  336  337  338
  340  344  345  351  352  368  371  378  379  380  398  414  438  464
  471  479  480  491  492  513  519  527  583  584  594  595  618  636
  637  666  676  683  694  695  709  725  728  739  746  748  755  769
  771  772  776  797  837  844  875  876]
[250 250  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50]

 2023-03-08 00:07:23,775 | INFO | root   Evaluation eval_after_decouple, avg total: 0.699, finest avg: 0.6896 with 90 classes, coarse avg: 0.784 with 2 classes, avg top0 total: None, aux_acc: 0.8278000008583069
[829, 561, 847, 866, 757]

 2023-03-08 00:08:05,251 | INFO | root   Begin task 23

 2023-03-08 00:08:05,251 | INFO | root   Now [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20] examplars per class.

 2023-03-08 00:08:05,386 | INFO | root   Step 23 weight decay 0.00050

 2023-03-08 00:09:41,003 | INFO | root   Begin evaluation: eval_after_decouple
[-452   10   11   12   13   14   16   17   18   19   20   81   82   84
   85   86   99  100  129  131  135  136  138  139  145  146  268  269
  274  276  279  282  283  291  292  293  333  334  336  337  338  340
  344  345  351  352  368  371  378  379  380  398  414  438  464  471
  479  480  491  492  513  519  527  561  583  584  594  595  618  636
  637  666  676  683  694  695  709  725  728  739  746  748  755  757
  769  771  772  776  797  829  837  844  847  866  875  876]
[250  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50  50
  50  50  50  50  50  50]

 2023-03-08 00:10:10,982 | INFO | root   Evaluation eval_after_decouple, avg total: 0.696, finest avg: 0.6943 with 95 classes, coarse avg: 0.736 with 1 classes, avg top0 total: None, aux_acc: 0.8787999998092652
[428, 565, 612, 670, 880]

 2023-03-08 00:10:26,386 | INFO | root   Begin task 24

 2023-03-08 00:10:26,387 | INFO | root   Now [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20] examplars per class.

 2023-03-08 00:10:26,521 | INFO | root   Step 24 weight decay 0.00050

 2023-03-08 00:12:03,558 | INFO | root   Begin evaluation: eval_after_decouple
[ 10  11  12  13  14  16  17  18  19  20  81  82  84  85  86  99 100 129
 131 135 136 138 139 145 146 268 269 274 276 279 282 283 291 292 293 333
 334 336 337 338 340 344 345 351 352 368 371 378 379 380 398 414 428 438
 464 471 479 480 491 492 513 519 527 561 565 583 584 594 595 612 618 636
 637 666 670 676 683 694 695 709 725 728 739 746 748 755 757 769 771 772
 776 797 829 837 844 847 866 875 876 880]
[50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50
 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50
 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50
 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50
 50 50 50 50]

 2023-03-08 00:12:34,603 | INFO | root   Evaluation eval_after_decouple, avg total: 0.696, finest avg: 0.6956 with 100 classes, coarse avg: None with 0 classes, avg top0 total: None, aux_acc: 0.8401999996185303
 


'''

import os
import numpy as np
def load_logs(info, restart=False):

    info_filtered = []
    record_list = []
    for i in info.split('\n'):
        if i != '\n' and i != '':
            info_filtered.append(i)
            if 'Evaluation eval_after_decouple' in i:
                finest_info = i.split('finest avg: ')[1].split(' classes, coarse avg:')[0].split(' with ')
                if int(finest_info[1]) != 0:
                    record_list.append(float(finest_info[0]))
    if restart:
        record_list = record_list[1:]
    return record_list

b = load_logs(a)
# # b.append(0.883)
# # b.append(0.715)
# b.append(0.663)
print(len(b))
print(np.mean(b))
#
# a = '''cp -r /datasets/codes/ctl_imagenet100_wo_connect_only_ance_fs512_512_512_more_data_Mar3/ctl /datasets/codes/ctl_imagenet100_wo_connect_only_ance_fs512_512_512_more_data_Mar3_eval
# cp ~/config_folder_eval/ctl_imagenet100_wo_connect_only_ance_fs512_512_512_more_data_Mar3_ind64/ctl2_gpu_imagenet100.yaml /datasets/codes/ctl_imagenet100_wo_connect_only_ance_fs512_512_512_more_data_Mar3_eval/ctl/codes/base/configs
# cp ~/addition_update_eval/non_baseline_imagenet_main/main.py /datasets/codes/ctl_imagenet100_wo_connect_only_ance_fs512_512_512_more_data_Mar3_eval/ctl/codes/base
# cp ~/addition_update_eval/non_baseline_imagenet_dataset/dataset.py /datasets/codes/ctl_imagenet100_wo_connect_only_ance_fs512_512_512_more_data_Mar3_eval/ctl/inclearn/datasets
# cp ~/addition_update_eval/non_baseline_imagenet_data/data.py /datasets/codes/ctl_imagenet100_wo_connect_only_ance_fs512_512_512_more_data_Mar3_eval/ctl/inclearn/datasets
# cp ~/addition_update_eval/non_baseline_imagenet_metrics/metrics.py /datasets/codes/ctl_imagenet100_wo_connect_only_ance_fs512_512_512_more_data_Mar3_eval/ctl/inclearn/deeprtc'''
# for i in a.split('\n'):
#     if '/datasets/codes/ctl_imagenet100_wo_connect_only_ance_fs512_512_512_more_data_Mar3_eval' in i:
#         i = i.replace('/datasets/codes/ctl_imagenet100_wo_connect_only_ance_fs512_512_512_more_data_Mar3_eval', '~')
#     print(i)

# for i in list(range(65, 72)):
#     if i != 70:
#         print(f'kubectl create -f /Users/chenyuzhao/Desktop/UCSD项目/server/job/gen_job_info_eval/job_{i}_eval.yaml')