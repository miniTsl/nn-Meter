# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from sklearn.ensemble import RandomForestRegressor


__PREDICTOR_ZOO__ = {
    "conv-bn-relu": {
        "tflite_cpu": {
            "max_depth": 70,
            "n_estimators": 320,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 6,
            "oob_score": True,
            "random_state": 10,
        },
        "tflite_gpu": {
            "max_depth": 80,
            "n_estimators": 550,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 5,
            "oob_score": True,
            "n_jobs": 32,
            "random_state": 10,
        },
        "openvino_vpu": {
            "max_depth": 100,
            "n_estimators": 500,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 5,
            "oob_score": True,
            "n_jobs": 32,
            "random_state": 10,
        },
        "pytorch_gpu": {
            "max_depth": 70,
            "n_estimators": 400,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 8,
            "oob_score": True,
            "n_jobs": 32,
            "random_state": 10,
        },
    },
    "dwconv-bn-relu": {
        "tflite_cpu": {
            "max_depth": 50,
            "n_estimators": 240,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 6,
            "oob_score": True,
            "random_state": 10,
        },
        "tflite_gpu": {
            "max_depth": 40,
            "n_estimators": 240,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 7,
            "oob_score": True,
            "random_state": 10,
        },
        "openvino_vpu": {
            "max_depth": 100,
            "n_estimators": 650,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 5,
            "oob_score": True,
            "n_jobs": 32,
            "random_state": 10,
        }
    },
    "fc": {
        "tflite_cpu": {
            "max_depth": 50,
            "n_estimators": 370,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        },
        "tflite_gpu": {
            "max_depth": 70,
            "n_estimators": 330,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 4,
            "oob_score": True,
            "random_state": 10,
        },
        "openvino_vpu": {
            "max_depth": 70,
            "n_estimators": 330,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 4,
            "oob_score": True,
            "n_jobs": 32,
            "random_state": 10,
        },
        "pytorch_gpu": {
            "max_depth": 5,
            "n_estimators": 20,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "n_jobs": 32,
            "random_state": 10,
        }
    },
    "channelshuffle": {
        "tflite_cpu": {
            "max_depth": 50,
            "n_estimators": 370,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        },
        "openvino_vpu": {
            "max_depth": 50,
            "n_estimators": 370,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        }
    },
    "se": {
        "tflite_cpu": {
            "max_depth": 20,
            "n_estimators": 290,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        },
        "tflite_gpu": {
            "max_depth": 50,
            "n_estimators": 190,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        },
        "openvino_vpu": {
            "max_depth": 50,
            "n_estimators": 110,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        }
    },
    "maxpool": {
        "tflite_cpu": {
            "max_depth": 50,
            "n_estimators": 210,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 5,
            "oob_score": True,
            "random_state": 10,
        },
        "tflite_gpu": {
            "max_depth": 50,
            "n_estimators": 370,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 5,
            "oob_score": True,
            "random_state": 10,
        },
        "openvino_vpu": {
            "max_depth": 50,
            "n_estimators": 370,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 5,
            "oob_score": True,
            "random_state": 10,
        }
    },
    "global-avgpool": {
        "tflite_cpu": {
            "max_depth": 70,
            "n_estimators": 370,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        }
    },
    "hswish": {
        "tflite_cpu": {
            "max_depth": 50,
            "n_estimators": 190,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        },
        "tflite_gpu": {
            "max_depth": 50,
            "n_estimators": 190,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        },
        "openvino_vpu": {
            "max_depth": 50,
            "n_estimators": 110,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        }
    },
    "avgpool": {
        "tflite_cpu": {
            "max_depth": 50,
            "n_estimators": 370,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 5,
            "oob_score": True,
            "random_state": 10,
        },
        "tflite_gpu": {
            "max_depth": 50,
            "n_estimators": 370,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 5,
            "oob_score": True,
            "random_state": 10,
        },
        "openvino_vpu": {
            "max_depth": 50,
            "n_estimators": 390,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 5,
            "oob_score": True,
            "random_state": 10,
        }
    },
    "bnrelu": {
        "tflite_cpu": {
            "max_depth": 50,
            "n_estimators": 370,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        },
        "tflite_gpu": {
            "max_depth": 50,
            "n_estimators": 190,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        },
        "openvino_vpu": {
            "max_depth": 50,
            "n_estimators": 570,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        }
    },
    "relu": {
        "tflite_cpu": {
            "max_depth": 50,
            "n_estimators": 370,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        },
        "tflite_gpu": {
            "max_depth": 50,
            "n_estimators": 190,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        },
        "openvino_vpu": {
            "max_depth": 50,
            "n_estimators": 190,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        }
    },
    "bn": {
        "tflite_cpu": {
            "max_depth": 50,
            "n_estimators": 370,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        },
        "tflite_gpu": {
            "max_depth": 50,
            "n_estimators": 190,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        },
        "openvino_vpu": {
            "max_depth": 50,
            "n_estimators": 390,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        }
    },
    "concat": {
        "tflite_cpu": {
            "max_depth": 100,
            "n_estimators": 690,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 5,
            "oob_score": True,
            "random_state": 10,
        },
        "tflite_gpu": {
            "max_depth": 100,
            "n_estimators": 690,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 5,
            "oob_score": True,
            "random_state": 10,
        }
    },
    "addrelu": {
        "tflite_cpu": {
            "max_depth": 50,
            "n_estimators": 570,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 3,
            "oob_score": True,
            "random_state": 10,
        },
        "tflite_gpu": {
            "max_depth": 50,
            "n_estimators": 570,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 3,
            "oob_score": True,
            "random_state": 10,
        },
        "openvino_vpu": {
            "max_depth": 50,
            "n_estimators": 570,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 3,
            "oob_score": True,
            "random_state": 10,
        }
    },
    "add": {
        "tflite_cpu": {
            "max_depth": 50,
            "n_estimators": 570,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 3,
            "oob_score": True,
            "random_state": 10,
        },
        "tflite_gpu": {
            "max_depth": 50,
            "n_estimators": 570,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 3,
            "oob_score": True,
            "random_state": 10,
        },
        "openvino_vpu": {
            "max_depth": 50,
            "n_estimators": 570,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 3,
            "oob_score": True,
            "random_state": 10,
        },
        "pytorch_gpu": {
            "max_depth": 20,
            "n_estimators": 80,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 4,
            "oob_score": True,
            "random_state": 10,
        }
    },
    "split": {
        "tflite_cpu": {
            "max_depth": 50,
            "n_estimators": 190,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        },
        "pytorch_gpu": {
            "max_depth": 20,
            "n_estimators": 250,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": 2,
            "oob_score": True,
            "random_state": 10,
        }

    }
}


def init_predictor(kernel_type, backend):
    try:
        model_param = __PREDICTOR_ZOO__[kernel_type][backend]
        model = RandomForestRegressor(**model_param)
    except:
        model = RandomForestRegressor(
            max_depth = 50,
            n_estimators = 370,
            min_samples_leaf = 1,
            min_samples_split = 2,
            # TODO: check the max_features for different kernels, the original "auto" is deprecated: The 'max_features' parameter of RandomForestRegressor must be an int in the range [1, inf), a float in the range (0.0, 1.0], a str among {'log2', 'sqrt'} or None. Got 'auto' instead.
            # TODO: check the number of features for different kernels, here we use 4 as default
            max_features = 4,
            oob_score = True,
            random_state = 10,
        )
    return model
