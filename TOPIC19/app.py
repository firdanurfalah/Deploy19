import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = [ "age", "RestingBP","Cholesterol","FastingBS", "MaxHR", "Oldpeak ",
                       "  Sex_F", "Sex_M", "ChestPainType_ASY", "ChestPainType_ATA", "ChestPainType_NAP"," ChestPainType_TA ",
                        "RestingECG_LVH ","RestingECG_Normal","RestingECG_ST","ExerciseAngina_N","ExerciseAngina_Y",
                        "ST_Slope_Down ","ST_Slope_Flat","ST_Slope_Up"
                    ]
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
        
    if output == 1:
        res_val = "** heart disease **"
    else:
        res_val = "no heart disease "
        

    return render_template('index.html', prediction_text='Patient has {}'.format(res_val))

if __name__ == "__main__":
    app.run()
