from hydlpy.model import DplHydroModel

config = {
    "hydrology": {
        "name": "exphydro",
        "hidden_size": 16
    },
    "static_parameter_estimator":{
        "name": None       
    },
    "dynamic_parameter_estimator":{
        "name": "direct"
    },
    "dynamic_parameter_estimator":{
        "name": "direct"
    }
}

model =  DplHydroModel(config)