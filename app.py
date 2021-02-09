from flask import Flask, jsonify
from pyFTS.data import Enrollments
from pyFTS.partitioners import Grid
from pyFTS.models import chen, cheng

app = Flask(__name__)



@app.route('/')
def index():
    train = [Enrollments.get_data()]
    test = [Enrollments.get_data()]
    fs = Grid.GridPartitioner(data=train, npart=10)
    model = chen.ConventionalFTS(partitioner=fs)
    model.fit(train)
    forecasts = model.predict(test)

    data = {
        'train': train,
        'test': test,
        'forecast': forecasts,
        # 'rmse': er
    }

    # return jsonify({'message': 'Hello World'})
    return jsonify(data)

app.run(debug=True)
