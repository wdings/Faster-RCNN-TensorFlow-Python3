from sotabencheval.image_classification import ImageNetEvaluator

evaluator = ImageNetEvaluator(
    # automatically compare to this paper
    model_name='Faster-RCNN-TensorFlow-Python3',
    paper_arxiv_id='1506.01497'
)

predictions = dict(zip(image_ids, batch_output)) # use your model to make predictions

evaluator.add(predictions)
evaluator.save()
