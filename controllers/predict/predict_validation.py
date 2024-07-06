# Exteranal Imports
# Model Validation
from pydantic import BaseModel
import pandas as pd

# Internal Imports

# Model
class PredictRequest(BaseModel):
    """
      data = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal,
            'target': target,
        }
        Create a Pydantic Model to validate the incoming data
        set Message to the incoming data
    """
    age: int
    sex :int
    cp :int
    trestbps :int
    chol :int
    fbs :int
    restecg :int
    thalach :int
    exang :int
    oldpeak :int
    slope :int
    ca :int
    thal :int

    def dict(self):
        """
        Convert the incoming data to a dictionary
        """
        return {
            'age': self.age,
            'sex': self.sex,
            'cp': self.cp,
            'trestbps': self.trestbps,
            'chol': self.chol,
            'fbs': self.fbs,
            'restecg': self.restecg,
            'thalach': self.thalach,
            'exang': self.exang,
            'oldpeak': self.oldpeak,
            'slope': self.slope,
            'ca': self.ca,
            'thal': self.thal,
        }