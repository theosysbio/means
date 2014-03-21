"""
Parameter Inference Parallelisation
----

This part of the package provides helper functions to make parameter inference run in parallel.

"""

import multiprocessing

def multiprocessing_pool_initialiser(objects, infer_args, infer_kwargs):
    global inference_objects  # Global is ok here as this function will be called for each process on separate threads
    inference_objects = objects
    global inference_args, inference_kwargs
    inference_args, inference_kwargs = infer_args, infer_kwargs


def multiprocessing_apply_infer(object_id):
    """
    Used in the InferenceWithRestarts class.
    Needs to be in global scope for multiprocessing module to pick it up

    """
    global inference_objects, inference_args, inference_kwargs
    return inference_objects[object_id]._infer_raw(*inference_args, **inference_kwargs)

def raw_results_in_parallel(inference_objects, number_of_processes, *args, **kwargs):
    p = multiprocessing.Pool(number_of_processes, initializer=multiprocessing_pool_initialiser,
                             initargs=[inference_objects, args, kwargs])
    results = p.map(multiprocessing_apply_infer, range(len(inference_objects)))
    p.close()

    return results
