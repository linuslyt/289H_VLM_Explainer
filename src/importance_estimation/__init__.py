# single_inference()
# take image as input
# get image caption from model
# save hidden state at layer 31
# save logits
# return ['caption', 'tokens']

# create_concept_dict_for_token()

# create_concept_dicts_for_caption()
# run concept dict extraction pipeline for list of tokens
# return { token: {dict}, token2: {dict} }

# project_on_concept_dict()
# argmin thing

# reconstruct_differentiable_hidden_state()
# create h for patching into the model

# calculate_gradient_concept()
# get logit of token in output caption
# call backward on token
# calculate gradients
# calculate gradient * concepts
# return ordered list of concepts + (gradient * concept) scores