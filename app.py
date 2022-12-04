from flask import Flask, render_template, request
from parser.parser_links import make_links
from parser.download_images import get_new_images
from model.add_category import add_category

app = Flask('__name__')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=["POST"])
def predict():
    category = [x for x in request.form.values()][0]

    make_links(category)
    get_new_images(category)

    return render_template('index.html', prediction_text='Category {} is added.'.format(category))


if __name__=='__main__':
    app.run(debug=True)
