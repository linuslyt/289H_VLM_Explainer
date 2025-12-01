batched sub_item:
{'img_id': 'COCO_train2014_000000283451', 
 'instruction': '\nProvide a one-sentence caption for the provided image.', 
 'response': 'A young boy holding a long haired dog in a pink feather costume.', 
 'image': '/media/data/ytllam/coco/train2014/COCO_train2014_000000283451.jpg', 
 'targets': 'A young boy holding a long haired dog in a pink feather costume.$$A little boy holds a small dog while he sits on a bench$$a boy is holding up a small dog$$A child is holding a pomeranian dog in his hands.$$A boy in blue jersey holding a brown dog.', 
 'text': '\nProvide a one-sentence caption for the provided image.', 
 'model_output': tensor([[ ... ]], device='cuda:0'), 
  'model_generated_output': tensor([[ ... ]], device='cuda:0'), 
  'model_predictions': ['A young boy is holding a dog that is wearing a pink feather boa.']}

batched hook_data:
{'img_id': ['COCO_train2014_000000209139'], 
 'instruction': ['\nProvide a one-sentence caption for the provided image.'], 
 'response': ['A dog in a pool swimming with a frisbee in its mouth.'], 
 'image': ['/media/data/ytllam/coco/train2014/COCO_train2014_000000209139.jpg'], 
 'targets': ['A dog in a pool swimming with a frisbee in its mouth.$$A dog in a pool with some sort of white ring in its mouth.$$a dog swimming in a pool holding a circular thing$$A dog catches a disc in its mouth while swimming.$$a dog in the water with a frisbee in its mouth'], 
 'text': ['\nProvide a one-sentence caption for the provided image.'], 
 'model_output': [tensor([[    1,  3148,  1001, 29901, 29871, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 29871,    13,    13,  1184, 29894,   680,   263,   697, 29899,
         18616,   663,  5777,   683,   363,   278,  4944,  1967, 29889,   319,
          1799,  9047, 13566, 29901,   319, 11203,   411,   263,  4796,  1424,
           275,   915, 29872,   297,   967, 13394,   338,  2381, 25217,   297,
           263, 11565, 29889,     2]], device='cuda:0')], 
  'model_generated_output': [tensor([[  319, 11203,   411,   263,  4796,  1424,   275,   915, 29872,   297,
           967, 13394,   338,  2381, 25217,   297,   263, 11565, 29889,     2]],
       device='cuda:0')], 
  'model_predictions': [['A dog with a white frisbee in its mouth is swimming in a pool.']], 
  'token_of_interest_mask': [tensor([True], device='cuda:0')], 
  'hidden_states': [
    {
      'language_model.model.layers.30': tensor([[[ 0.2148,  0.5879,  3.9453,  ..., -1.1348,  0.0571,  1.7344]], [[-0.3384, -0.3345,  4.3438,  ..., -1.9551, -1.3643, -1.2568]]], dtype=torch.float16), 
      'language_model.model.layers.31': tensor([[[-0.3828, -2.3379,  4.4297,  ...,  0.1582,  1.3984,  1.2402]], [[ 0.0134, -2.1602,  3.8555,  ..., -0.9424,  1.4551, -2.4883]]], dtype=torch.float16), 
      'language_model.model.norm': tensor([[[-0.2075, -1.2676,  2.3203,  ...,  0.0789,  0.7422,  0.5767]], [[ 0.0067, -1.0703,  1.8467,  ..., -0.4302,  0.7061, -1.0576]]], dtype=torch.float16)
    }
  ]}

item.items()
dict_items([
  ('img_id', ['COCO_train2014_000000209139', 'COCO_train2014_000000302661']), 
  ('instruction', ['\nProvide a one-sentence caption for the provided image.', '\nProvide a one-sentence caption for the provided image.']), 
  ('response', ['A dog in a pool swimming with a frisbee in its mouth.', 'a few sheep are outside in a field with a dog']), 
  ('image', ['/media/data/ytllam/coco/train2014/COCO_train2014_000000209139.jpg', '/media/data/ytllam/coco/train2014/COCO_train2014_000000302661.jpg']), 
  ('targets', ['A dog in a pool swimming with a frisbee in its mouth.$$A dog in a pool with some sort of white ring in its mouth.$$a dog swimming in a pool holding a circular thing$$A dog catches a disc in its mouth while swimming.$$a dog in the water with a frisbee in its mouth', 'a few sheep are outside in a field with a dog$$Sheep stand in a field while some drink water.$$A few sheep wander around and drink water$$Sheep eat from a blue bowl with the sheepdog behind them.$$two large sheep are and a dog and one of the sheep is eating out of a bowl.']), ('text', ['\nProvide a one-sentence caption for the provided image.', '\nProvide a one-sentence caption for the provided image.'])
])

FIXED:
batched sub_item:
{'img_id': ['COCO_train2014_000000209139'], 
 'instruction': ['\nProvide a one-sentence caption for the provided image.'], 
 'response': ['A dog in a pool swimming with a frisbee in its mouth.'], 
 'image': ['/media/data/ytllam/coco/train2014/COCO_train2014_000000209139.jpg'], 
 'targets': ['A dog in a pool swimming with a frisbee in its mouth.$$A dog in a pool with some sort of white ring in its mouth.$$a dog swimming in a pool holding a circular thing$$A dog catches a disc in its mouth while swimming.$$a dog in the water with a frisbee in its mouth'], 
 'text': ['\nProvide a one-sentence caption for the provided image.'], 
 'model_output': tensor([[    1,  3148,  1001, 29901, 29871, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 29871,    13,    13,  1184, 29894,   680,   263,   697, 29899,
         18616,   663,  5777,   683,   363,   278,  4944,  1967, 29889,   319,
          1799,  9047, 13566, 29901,   319, 11203,   411,   263,  4796,  1424,
           275,   915, 29872,   297,   967, 13394,   338,  2381, 25217,   297,
           263, 11565, 29889,     2]], device='cuda:0'), 
  'model_generated_output': tensor([[  319, 11203,   411,   263,  4796,  1424,   275,   915, 29872,   297,
           967, 13394,   338,  2381, 25217,   297,   263, 11565, 29889,     2]],
       device='cuda:0'), 
  'model_predictions': ['A dog with a white frisbee in its mouth is swimming in a pool.']}


  batched hook_data:
{'img_id': [['COCO_train2014_000000209139']], 
 'instruction': [['\nProvide a one-sentence caption for the provided image.']], 
 'response': [['A dog in a pool swimming with a frisbee in its mouth.']], 
 'image': [['/media/data/ytllam/coco/train2014/COCO_train2014_000000209139.jpg']], 
 'targets': [['A dog in a pool swimming with a frisbee in its mouth.$$A dog in a pool with some sort of white ring in its mouth.$$a dog swimming in a pool holding a circular thing$$A dog catches a disc in its mouth while swimming.$$a dog in the water with a frisbee in its mouth']], 
 'text': [['\nProvide a one-sentence caption for the provided image.']], 
 'model_output': [tensor([[    1,  3148,  1001, 29901, 29871, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000, 32000,
         32000, 29871,    13,    13,  1184, 29894,   680,   263,   697, 29899,
         18616,   663,  5777,   683,   363,   278,  4944,  1967, 29889,   319,
          1799,  9047, 13566, 29901,   319, 11203,   411,   263,  4796,  1424,
           275,   915, 29872,   297,   967, 13394,   338,  2381, 25217,   297,
           263, 11565, 29889,     2]], device='cuda:0')],
  'model_generated_output': [tensor([[  319, 11203,   411,   263,  4796,  1424,   275,   915, 29872,   297,
           967, 13394,   338,  2381, 25217,   297,   263, 11565, 29889,     2]],
       device='cuda:0')],
  'model_predictions': [['A dog with a white frisbee in its mouth is swimming in a pool.']],
  'token_of_interest_mask': [tensor([True], device='cuda:0')], 
  'hidden_states': [{
    'language_model.model.layers.30': tensor([[[ 0.2148,  0.5879,  3.9453,  ..., -1.1348,  0.0571,  1.7344]], [[-0.3384, -0.3345,  4.3438,  ..., -1.9551, -1.3643, -1.2568]]], dtype=torch.float16),
    'language_model.model.layers.31': tensor([[[-0.3828, -2.3379,  4.4297,  ...,  0.1582,  1.3984,  1.2402]], [[ 0.0134, -2.1602,  3.8555,  ..., -0.9424,  1.4551, -2.4883]]], dtype=torch.float16),
    'language_model.model.norm': tensor([[[-0.2075, -1.2676,  2.3203,  ...,  0.0789,  0.7422,  0.5767]], [[ 0.0067, -1.0703,  1.8467,  ..., -0.4302,  0.7061, -1.0576]]], dtype=torch.float16)}]}

hidden_states = [{
    'language_model.model.layers.30': [1,2,3],
    'language_model.model.layers.31': [1,2,3],
    'language_model.model.norm': [1,2,3]
    }]

hs = {k: v[b : b + 1] for k, v in hidden_states.items()}


[{'language_model.model.layers.30': tensor([[[-1.5371,  0.7300,  0.0119,  ...,  0.6328, -1.4775,  0.3359]]], dtype=torch.float16), 
  'language_model.model.layers.31': tensor([[[-1.4092, -0.3088, -0.9326,  ...,  2.3848,  1.1406, -1.6641]]], dtype=torch.float16), 
  'language_model.model.norm': tensor([[[-0.7378, -0.1617, -0.4719,  ...,  1.1494,  0.5845, -0.7471]]], dtype=torch.float16)
  },
  {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, 
  {'language_model.model.layers.30': tensor([[[-2.0957, -0.4299, -2.4141,  ..., -2.4219, -1.6328,  0.6338]]], dtype=torch.float16), 
   'language_model.model.layers.31': tensor([[[-2.8926, -1.1807, -3.9199,  ..., -0.4082,  0.3184, -0.3135]]], dtype=torch.float16), 
   'language_model.model.norm': tensor([[[-1.3906, -0.5674, -1.8203,  ..., -0.1807,  0.1499, -0.1293]]], dtype=torch.float16)
  }, 
   {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]


   {'language_model.model.layers.30': tensor([[[-1.5371,  0.7300,  0.0119,  ...,  0.6328, -1.4775,  0.3359]],

        [[ 1.7988,  1.0762,  3.6641,  ...,  0.6367,  1.1963,  2.7070]],

        [[-0.8418,  2.8926,  3.7891,  ..., -3.9688,  0.4392, -0.9771]],

        ...,

        [[ 0.9272,  1.7617,  2.0684,  ..., -1.3369,  1.8799,  0.5039]],

        [[ 0.2393, -0.1006,  2.0898,  ..., -3.1523,  1.2939, -1.1631]],

        [[ 2.9727, -0.0698,  3.0273,  ...,  0.0320,  2.4062, -0.1626]]],
       dtype=torch.float16), 
    'language_model.model.layers.31': tensor([[[-1.4092, -0.3088, -0.9326,  ...,  2.3848,  1.1406, -1.6641]],

        [[ 4.0156, -2.4219,  3.6680,  ...,  1.1934,  1.9326,  1.3965]],

        [[ 1.8359, -0.9473,  4.1602,  ..., -3.5469,  1.5791, -3.4746]],

        ...,

        [[ 3.1113, -1.2686,  0.7988,  ..., -0.3369,  2.4238, -0.8760]],

        [[ 2.0371, -5.1445,  0.6270,  ..., -2.1641,  2.2734, -4.0117]],

        [[ 5.2500, -3.5137,  2.9082,  ...,  0.4795,  3.0957, -1.9004]]],
       dtype=torch.float16), 
    'language_model.model.norm': tensor([[[-0.7378, -0.1617, -0.4719,  ...,  1.1494,  0.5845, -0.7471]],

        [[ 1.8057, -1.0889,  1.5928,  ...,  0.4939,  0.8506,  0.5386]],

        [[ 0.8247, -0.4255,  1.8066,  ..., -1.4668,  0.6943, -1.3389]],

        ...,

        [[ 1.3086, -0.5332,  0.3245,  ..., -0.1304,  0.9980, -0.3159]],

        [[ 0.8740, -2.2090,  0.2600,  ..., -0.8550,  0.9551, -1.4766]],

        [[ 2.4824, -1.6602,  1.3281,  ...,  0.2085,  1.4316, -0.7705]]],
       dtype=torch.float16)}

{'language_model.model.layers.30': [[[1]], [[2]], [[3]]], 
'language_model.model.layers.31': [[[1]], [[2]], [[3]]], 
'language_model.model.norm': [[[1]], [[2]], [[3]]]}


>>> print_dict_structure_compact(d_batched)
image -> list, len=2
image[0] -> list, len=42
image[0][0] -> str, len=65
image[0][1] -> str, len=65
image[0][2] -> str, len=65
image[0][3] -> str, len=65
image[0][4] -> str, len=65
image[0][5] -> str, len=65
image[0][6] -> str, len=65
image[0][7] -> str, len=65
image[0][8] -> str, len=65
image[0][9] -> str, len=65
image[0][10] -> str, len=65
image[0][11] -> str, len=65
image[0][12] -> str, len=65
image[0][13] -> str, len=65
image[0][14] -> str, len=65
image[0][15] -> str, len=65
image[0][16] -> str, len=65
image[0][17] -> str, len=65
image[0][18] -> str, len=65
image[0][19] -> str, len=65
image[0][20] -> str, len=65
image[0][21] -> str, len=65
image[0][22] -> str, len=65
image[0][23] -> str, len=65
image[0][24] -> str, len=65
image[0][25] -> str, len=65
image[0][26] -> str, len=65
image[0][27] -> str, len=65
image[0][28] -> str, len=65
image[0][29] -> str, len=65
image[0][30] -> str, len=65
image[0][31] -> str, len=65
image[0][32] -> str, len=65
image[0][33] -> str, len=65
image[0][34] -> str, len=65
image[0][35] -> str, len=65
image[0][36] -> str, len=65
image[0][37] -> str, len=65
image[0][38] -> str, len=65
image[0][39] -> str, len=65
image[0][40] -> str, len=65
image[0][41] -> str, len=65
image[1] -> list, len=3
image[1][0] -> str, len=65
image[1][1] -> str, len=65
image[1][2] -> str, len=65
model_predictions -> list, len=2
model_predictions[0] -> list, len=42
model_predictions[0][0] -> str, len=67
model_predictions[0][1] -> str, len=45
model_predictions[0][2] -> str, len=57
model_predictions[0][3] -> str, len=56
model_predictions[0][4] -> str, len=44
model_predictions[0][5] -> str, len=47
model_predictions[0][6] -> str, len=39
model_predictions[0][7] -> str, len=41
model_predictions[0][8] -> str, len=41
model_predictions[0][9] -> str, len=40
model_predictions[0][10] -> str, len=47
model_predictions[0][11] -> str, len=48
model_predictions[0][12] -> str, len=55
model_predictions[0][13] -> str, len=43
model_predictions[0][14] -> str, len=46
model_predictions[0][15] -> str, len=72
model_predictions[0][16] -> str, len=55
model_predictions[0][17] -> str, len=36
model_predictions[0][18] -> str, len=50
model_predictions[0][19] -> str, len=51
model_predictions[0][20] -> str, len=52
model_predictions[0][21] -> str, len=50
model_predictions[0][22] -> str, len=76
model_predictions[0][23] -> str, len=44
model_predictions[0][24] -> str, len=41
model_predictions[0][25] -> str, len=36
model_predictions[0][26] -> str, len=42
model_predictions[0][27] -> str, len=51
model_predictions[0][28] -> str, len=55
model_predictions[0][29] -> str, len=44
model_predictions[0][30] -> str, len=64
model_predictions[0][31] -> str, len=55
model_predictions[0][32] -> str, len=35
model_predictions[0][33] -> str, len=65
model_predictions[0][34] -> str, len=64
model_predictions[0][35] -> str, len=59
model_predictions[0][36] -> str, len=53
model_predictions[0][37] -> str, len=61
model_predictions[0][38] -> str, len=42
model_predictions[0][39] -> str, len=46
model_predictions[0][40] -> str, len=49
model_predictions[0][41] -> str, len=57
model_predictions[1] -> list, len=3
model_predictions[1][0] -> str, len=60
model_predictions[1][1] -> str, len=56
model_predictions[1][2] -> str, len=40
token_of_interest_mask -> list, len=2
token_of_interest_mask[0] -> Tensor, len=42
token_of_interest_mask[1] -> Tensor, len=3
hidden_states -> list, len=2
hidden_states[0] -> dict, len=3
hidden_states[0].language_model.model.layers.30 -> Tensor, len=42
hidden_states[0].language_model.model.layers.31 -> Tensor, len=42
hidden_states[0].language_model.model.norm -> Tensor, len=42
hidden_states[1] -> dict, len=3
hidden_states[1].language_model.model.layers.30 -> Tensor, len=3
hidden_states[1].language_model.model.layers.31 -> Tensor, len=3
hidden_states[1].language_model.model.norm -> Tensor, len=3


image -> list, len=4
image[0] -> list, len=42
image[0][0] -> str, len=65
image[0][1] -> str, len=65
image[0][2] -> str, len=65
image[0][3] -> str, len=65
image[0][4] -> str, len=65
image[0][5] -> str, len=65
image[0][6] -> str, len=65
image[0][7] -> str, len=65
image[0][8] -> str, len=65
image[0][9] -> str, len=65
image[0][10] -> str, len=65
image[0][11] -> str, len=65
image[0][12] -> str, len=65
image[0][13] -> str, len=65
image[0][14] -> str, len=65
image[0][15] -> str, len=65
image[0][16] -> str, len=65
image[0][17] -> str, len=65
image[0][18] -> str, len=65
image[0][19] -> str, len=65
image[0][20] -> str, len=65
image[0][21] -> str, len=65
image[0][22] -> str, len=65
image[0][23] -> str, len=65
image[0][24] -> str, len=65
image[0][25] -> str, len=65
image[0][26] -> str, len=65
image[0][27] -> str, len=65
image[0][28] -> str, len=65
image[0][29] -> str, len=65
image[0][30] -> str, len=65
image[0][31] -> str, len=65
image[0][32] -> str, len=65
image[0][33] -> str, len=65
image[0][34] -> str, len=65
image[0][35] -> str, len=65
image[0][36] -> str, len=65
image[0][37] -> str, len=65
image[0][38] -> str, len=65
image[0][39] -> str, len=65
image[0][40] -> str, len=65
image[0][41] -> str, len=65
image[1] -> str, len=65
image[2] -> str, len=65
image[3] -> str, len=65
model_predictions -> list, len=4
model_predictions[0] -> list, len=42
model_predictions[0][0] -> str, len=67
model_predictions[0][1] -> str, len=45
model_predictions[0][2] -> str, len=57
model_predictions[0][3] -> str, len=56
model_predictions[0][4] -> str, len=44
model_predictions[0][5] -> str, len=47
model_predictions[0][6] -> str, len=39
model_predictions[0][7] -> str, len=41
model_predictions[0][8] -> str, len=41
model_predictions[0][9] -> str, len=40
model_predictions[0][10] -> str, len=47
model_predictions[0][11] -> str, len=48
model_predictions[0][12] -> str, len=55
model_predictions[0][13] -> str, len=43
model_predictions[0][14] -> str, len=46
model_predictions[0][15] -> str, len=72
model_predictions[0][16] -> str, len=55
model_predictions[0][17] -> str, len=36
model_predictions[0][18] -> str, len=50
model_predictions[0][19] -> str, len=51
model_predictions[0][20] -> str, len=52
model_predictions[0][21] -> str, len=50
model_predictions[0][22] -> str, len=76
model_predictions[0][23] -> str, len=44
model_predictions[0][24] -> str, len=41
model_predictions[0][25] -> str, len=36
model_predictions[0][26] -> str, len=42
model_predictions[0][27] -> str, len=51
model_predictions[0][28] -> str, len=55
model_predictions[0][29] -> str, len=44
model_predictions[0][30] -> str, len=64
model_predictions[0][31] -> str, len=55
model_predictions[0][32] -> str, len=35
model_predictions[0][33] -> str, len=65
model_predictions[0][34] -> str, len=64
model_predictions[0][35] -> str, len=59
model_predictions[0][36] -> str, len=53
model_predictions[0][37] -> str, len=61
model_predictions[0][38] -> str, len=42
model_predictions[0][39] -> str, len=46
model_predictions[0][40] -> str, len=49
model_predictions[0][41] -> str, len=57
model_predictions[1] -> str, len=60
model_predictions[2] -> str, len=56
model_predictions[3] -> str, len=40
token_of_interest_mask -> list, len=2
token_of_interest_mask[0] -> Tensor, len=42
token_of_interest_mask[1] -> Tensor, len=3
hidden_states -> list, len=2
hidden_states[0] -> dict, len=3
hidden_states[0].language_model.model.layers.30 -> Tensor, len=42
hidden_states[0].language_model.model.layers.31 -> Tensor, len=42
hidden_states[0].language_model.model.norm -> Tensor, len=42
hidden_states[1] -> dict, len=3
hidden_states[1].language_model.model.layers.30 -> Tensor, len=3
hidden_states[1].language_model.model.layers.31 -> Tensor, len=3
hidden_states[1].language_model.model.norm -> Tensor, len=3


image -> list, len=45
image[0] -> str, len=65
image[1] -> str, len=65
image[2] -> str, len=65
image[3] -> str, len=65
image[4] -> str, len=65
image[5] -> str, len=65
image[6] -> str, len=65
image[7] -> str, len=65
image[8] -> str, len=65
image[9] -> str, len=65
image[10] -> str, len=65
image[11] -> str, len=65
image[12] -> str, len=65
image[13] -> str, len=65
image[14] -> str, len=65
image[15] -> str, len=65
image[16] -> str, len=65
image[17] -> str, len=65
image[18] -> str, len=65
image[19] -> str, len=65
image[20] -> str, len=65
image[21] -> str, len=65
image[22] -> str, len=65
image[23] -> str, len=65
image[24] -> str, len=65
image[25] -> str, len=65
image[26] -> str, len=65
image[27] -> str, len=65
image[28] -> str, len=65
image[29] -> str, len=65
image[30] -> str, len=65
image[31] -> str, len=65
image[32] -> str, len=65
image[33] -> str, len=65
image[34] -> str, len=65
image[35] -> str, len=65
image[36] -> str, len=65
image[37] -> str, len=65
image[38] -> str, len=65
image[39] -> str, len=65
image[40] -> str, len=65
image[41] -> str, len=65
image[42] -> str, len=65
image[43] -> str, len=65
image[44] -> str, len=65
model_predictions -> list, len=45
model_predictions[0] -> str, len=67
model_predictions[1] -> str, len=45
model_predictions[2] -> str, len=57
model_predictions[3] -> str, len=56
model_predictions[4] -> str, len=44
model_predictions[5] -> str, len=47
model_predictions[6] -> str, len=39
model_predictions[7] -> str, len=41
model_predictions[8] -> str, len=41
model_predictions[9] -> str, len=40
model_predictions[10] -> str, len=47
model_predictions[11] -> str, len=48
model_predictions[12] -> str, len=55
model_predictions[13] -> str, len=43
model_predictions[14] -> str, len=46
model_predictions[15] -> str, len=72
model_predictions[16] -> str, len=55
model_predictions[17] -> str, len=36
model_predictions[18] -> str, len=50
model_predictions[19] -> str, len=51
model_predictions[20] -> str, len=52
model_predictions[21] -> str, len=50
model_predictions[22] -> str, len=76
model_predictions[23] -> str, len=44
model_predictions[24] -> str, len=41
model_predictions[25] -> str, len=36
model_predictions[26] -> str, len=42
model_predictions[27] -> str, len=51
model_predictions[28] -> str, len=55
model_predictions[29] -> str, len=44
model_predictions[30] -> str, len=64
model_predictions[31] -> str, len=55
model_predictions[32] -> str, len=35
model_predictions[33] -> str, len=65
model_predictions[34] -> str, len=64
model_predictions[35] -> str, len=59
model_predictions[36] -> str, len=53
model_predictions[37] -> str, len=61
model_predictions[38] -> str, len=42
model_predictions[39] -> str, len=46
model_predictions[40] -> str, len=49
model_predictions[41] -> str, len=57
model_predictions[42] -> str, len=60
model_predictions[43] -> str, len=56
model_predictions[44] -> str, len=40
token_of_interest_mask -> list, len=45
token_of_interest_mask[0] -> Tensor
token_of_interest_mask[1] -> Tensor
token_of_interest_mask[2] -> Tensor
token_of_interest_mask[3] -> Tensor
token_of_interest_mask[4] -> Tensor
token_of_interest_mask[5] -> Tensor
token_of_interest_mask[6] -> Tensor
token_of_interest_mask[7] -> Tensor
token_of_interest_mask[8] -> Tensor
token_of_interest_mask[9] -> Tensor
token_of_interest_mask[10] -> Tensor
token_of_interest_mask[11] -> Tensor
token_of_interest_mask[12] -> Tensor
token_of_interest_mask[13] -> Tensor
token_of_interest_mask[14] -> Tensor
token_of_interest_mask[15] -> Tensor
token_of_interest_mask[16] -> Tensor
token_of_interest_mask[17] -> Tensor
token_of_interest_mask[18] -> Tensor
token_of_interest_mask[19] -> Tensor
token_of_interest_mask[20] -> Tensor
token_of_interest_mask[21] -> Tensor
token_of_interest_mask[22] -> Tensor
token_of_interest_mask[23] -> Tensor
token_of_interest_mask[24] -> Tensor
token_of_interest_mask[25] -> Tensor
token_of_interest_mask[26] -> Tensor
token_of_interest_mask[27] -> Tensor
token_of_interest_mask[28] -> Tensor
token_of_interest_mask[29] -> Tensor
token_of_interest_mask[30] -> Tensor
token_of_interest_mask[31] -> Tensor
token_of_interest_mask[32] -> Tensor
token_of_interest_mask[33] -> Tensor
token_of_interest_mask[34] -> Tensor
token_of_interest_mask[35] -> Tensor
token_of_interest_mask[36] -> Tensor
token_of_interest_mask[37] -> Tensor
token_of_interest_mask[38] -> Tensor
token_of_interest_mask[39] -> Tensor
token_of_interest_mask[40] -> Tensor
token_of_interest_mask[41] -> Tensor
token_of_interest_mask[42] -> Tensor
token_of_interest_mask[43] -> Tensor
token_of_interest_mask[44] -> Tensor
hidden_states -> dict, len=3
hidden_states.language_model.model.layers.30 -> list, len=45
hidden_states.language_model.model.layers.30[0] -> Tensor, len=1
hidden_states.language_model.model.layers.30[1] -> Tensor, len=1
hidden_states.language_model.model.layers.30[2] -> Tensor, len=1
hidden_states.language_model.model.layers.30[3] -> Tensor, len=1
hidden_states.language_model.model.layers.30[4] -> Tensor, len=1
hidden_states.language_model.model.layers.30[5] -> Tensor, len=1
hidden_states.language_model.model.layers.30[6] -> Tensor, len=1
hidden_states.language_model.model.layers.30[7] -> Tensor, len=1
hidden_states.language_model.model.layers.30[8] -> Tensor, len=1
hidden_states.language_model.model.layers.30[9] -> Tensor, len=1
hidden_states.language_model.model.layers.30[10] -> Tensor, len=1
hidden_states.language_model.model.layers.30[11] -> Tensor, len=1
hidden_states.language_model.model.layers.30[12] -> Tensor, len=1
hidden_states.language_model.model.layers.30[13] -> Tensor, len=1
hidden_states.language_model.model.layers.30[14] -> Tensor, len=1
hidden_states.language_model.model.layers.30[15] -> Tensor, len=1
hidden_states.language_model.model.layers.30[16] -> Tensor, len=1
hidden_states.language_model.model.layers.30[17] -> Tensor, len=1
hidden_states.language_model.model.layers.30[18] -> Tensor, len=1
hidden_states.language_model.model.layers.30[19] -> Tensor, len=1
hidden_states.language_model.model.layers.30[20] -> Tensor, len=1
hidden_states.language_model.model.layers.30[21] -> Tensor, len=1
hidden_states.language_model.model.layers.30[22] -> Tensor, len=1
hidden_states.language_model.model.layers.30[23] -> Tensor, len=1
hidden_states.language_model.model.layers.30[24] -> Tensor, len=1
hidden_states.language_model.model.layers.30[25] -> Tensor, len=1
hidden_states.language_model.model.layers.30[26] -> Tensor, len=1
hidden_states.language_model.model.layers.30[27] -> Tensor, len=1
hidden_states.language_model.model.layers.30[28] -> Tensor, len=1
hidden_states.language_model.model.layers.30[29] -> Tensor, len=1
hidden_states.language_model.model.layers.30[30] -> Tensor, len=1
hidden_states.language_model.model.layers.30[31] -> Tensor, len=1
hidden_states.language_model.model.layers.30[32] -> Tensor, len=1
hidden_states.language_model.model.layers.30[33] -> Tensor, len=1
hidden_states.language_model.model.layers.30[34] -> Tensor, len=1
hidden_states.language_model.model.layers.30[35] -> Tensor, len=1
hidden_states.language_model.model.layers.30[36] -> Tensor, len=1
hidden_states.language_model.model.layers.30[37] -> Tensor, len=1
hidden_states.language_model.model.layers.30[38] -> Tensor, len=1
hidden_states.language_model.model.layers.30[39] -> Tensor, len=1
hidden_states.language_model.model.layers.30[40] -> Tensor, len=1
hidden_states.language_model.model.layers.30[41] -> Tensor, len=1
hidden_states.language_model.model.layers.30[42] -> Tensor, len=1
hidden_states.language_model.model.layers.30[43] -> Tensor, len=1
hidden_states.language_model.model.layers.30[44] -> Tensor, len=1
hidden_states.language_model.model.layers.31 -> list, len=45
hidden_states.language_model.model.layers.31[0] -> Tensor, len=1
hidden_states.language_model.model.layers.31[1] -> Tensor, len=1
hidden_states.language_model.model.layers.31[2] -> Tensor, len=1
hidden_states.language_model.model.layers.31[3] -> Tensor, len=1
hidden_states.language_model.model.layers.31[4] -> Tensor, len=1
hidden_states.language_model.model.layers.31[5] -> Tensor, len=1
hidden_states.language_model.model.layers.31[6] -> Tensor, len=1
hidden_states.language_model.model.layers.31[7] -> Tensor, len=1
hidden_states.language_model.model.layers.31[8] -> Tensor, len=1
hidden_states.language_model.model.layers.31[9] -> Tensor, len=1
hidden_states.language_model.model.layers.31[10] -> Tensor, len=1
hidden_states.language_model.model.layers.31[11] -> Tensor, len=1
hidden_states.language_model.model.layers.31[12] -> Tensor, len=1
hidden_states.language_model.model.layers.31[13] -> Tensor, len=1
hidden_states.language_model.model.layers.31[14] -> Tensor, len=1
hidden_states.language_model.model.layers.31[15] -> Tensor, len=1
hidden_states.language_model.model.layers.31[16] -> Tensor, len=1
hidden_states.language_model.model.layers.31[17] -> Tensor, len=1
hidden_states.language_model.model.layers.31[18] -> Tensor, len=1
hidden_states.language_model.model.layers.31[19] -> Tensor, len=1
hidden_states.language_model.model.layers.31[20] -> Tensor, len=1
hidden_states.language_model.model.layers.31[21] -> Tensor, len=1
hidden_states.language_model.model.layers.31[22] -> Tensor, len=1
hidden_states.language_model.model.layers.31[23] -> Tensor, len=1
hidden_states.language_model.model.layers.31[24] -> Tensor, len=1
hidden_states.language_model.model.layers.31[25] -> Tensor, len=1
hidden_states.language_model.model.layers.31[26] -> Tensor, len=1
hidden_states.language_model.model.layers.31[27] -> Tensor, len=1
hidden_states.language_model.model.layers.31[28] -> Tensor, len=1
hidden_states.language_model.model.layers.31[29] -> Tensor, len=1
hidden_states.language_model.model.layers.31[30] -> Tensor, len=1
hidden_states.language_model.model.layers.31[31] -> Tensor, len=1
hidden_states.language_model.model.layers.31[32] -> Tensor, len=1
hidden_states.language_model.model.layers.31[33] -> Tensor, len=1
hidden_states.language_model.model.layers.31[34] -> Tensor, len=1
hidden_states.language_model.model.layers.31[35] -> Tensor, len=1
hidden_states.language_model.model.layers.31[36] -> Tensor, len=1
hidden_states.language_model.model.layers.31[37] -> Tensor, len=1
hidden_states.language_model.model.layers.31[38] -> Tensor, len=1
hidden_states.language_model.model.layers.31[39] -> Tensor, len=1
hidden_states.language_model.model.layers.31[40] -> Tensor, len=1
hidden_states.language_model.model.layers.31[41] -> Tensor, len=1
hidden_states.language_model.model.layers.31[42] -> Tensor, len=1
hidden_states.language_model.model.layers.31[43] -> Tensor, len=1
hidden_states.language_model.model.layers.31[44] -> Tensor, len=1
hidden_states.language_model.model.norm -> list, len=45
hidden_states.language_model.model.norm[0] -> Tensor, len=1
hidden_states.language_model.model.norm[1] -> Tensor, len=1
hidden_states.language_model.model.norm[2] -> Tensor, len=1
hidden_states.language_model.model.norm[3] -> Tensor, len=1
hidden_states.language_model.model.norm[4] -> Tensor, len=1
hidden_states.language_model.model.norm[5] -> Tensor, len=1
hidden_states.language_model.model.norm[6] -> Tensor, len=1
hidden_states.language_model.model.norm[7] -> Tensor, len=1
hidden_states.language_model.model.norm[8] -> Tensor, len=1
hidden_states.language_model.model.norm[9] -> Tensor, len=1
hidden_states.language_model.model.norm[10] -> Tensor, len=1
hidden_states.language_model.model.norm[11] -> Tensor, len=1
hidden_states.language_model.model.norm[12] -> Tensor, len=1
hidden_states.language_model.model.norm[13] -> Tensor, len=1
hidden_states.language_model.model.norm[14] -> Tensor, len=1
hidden_states.language_model.model.norm[15] -> Tensor, len=1
hidden_states.language_model.model.norm[16] -> Tensor, len=1
hidden_states.language_model.model.norm[17] -> Tensor, len=1
hidden_states.language_model.model.norm[18] -> Tensor, len=1
hidden_states.language_model.model.norm[19] -> Tensor, len=1
hidden_states.language_model.model.norm[20] -> Tensor, len=1
hidden_states.language_model.model.norm[21] -> Tensor, len=1
hidden_states.language_model.model.norm[22] -> Tensor, len=1
hidden_states.language_model.model.norm[23] -> Tensor, len=1
hidden_states.language_model.model.norm[24] -> Tensor, len=1
hidden_states.language_model.model.norm[25] -> Tensor, len=1
hidden_states.language_model.model.norm[26] -> Tensor, len=1
hidden_states.language_model.model.norm[27] -> Tensor, len=1
hidden_states.language_model.model.norm[28] -> Tensor, len=1
hidden_states.language_model.model.norm[29] -> Tensor, len=1
hidden_states.language_model.model.norm[30] -> Tensor, len=1
hidden_states.language_model.model.norm[31] -> Tensor, len=1
hidden_states.language_model.model.norm[32] -> Tensor, len=1
hidden_states.language_model.model.norm[33] -> Tensor, len=1
hidden_states.language_model.model.norm[34] -> Tensor, len=1
hidden_states.language_model.model.norm[35] -> Tensor, len=1
hidden_states.language_model.model.norm[36] -> Tensor, len=1
hidden_states.language_model.model.norm[37] -> Tensor, len=1
hidden_states.language_model.model.norm[38] -> Tensor, len=1
hidden_states.language_model.model.norm[39] -> Tensor, len=1
hidden_states.language_model.model.norm[40] -> Tensor, len=1
hidden_states.language_model.model.norm[41] -> Tensor, len=1
hidden_states.language_model.model.norm[42] -> Tensor, len=1
hidden_states.language_model.model.norm[43] -> Tensor, len=1
hidden_states.language_model.model.norm[44] -> Tensor, len=1