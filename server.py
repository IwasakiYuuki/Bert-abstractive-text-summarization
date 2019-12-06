from flask import Flask, render_template, request
import urllib.request
import json

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def summarize_form():
    if request.method == 'GET':
        title = 'GET'
        summary = ''
        return render_template('summarization_form.html', title=title, summary=summary)
    else:
        title = 'POST'
        source_text = request.form['source_text']
        return render_template(template_name_or_list='summarization_form.html', title=title, summary=source_text)


def summarize(source_text):
    url = "http://127.0.0.1:5000/"
    method = "POST"
    headers = {"Content-Type" : "application/json"}

    # PythonオブジェクトをJSONに変換する
    obj = {"source_text" : source_text}
    json_data = json.dumps(obj).encode("utf-8")

    # httpリクエストを準備してPOST
    requests = urllib.request.Request(url, data=json_data, method=method, headers=headers)
    with urllib.request.urlopen(requests) as response:
        response_body = response.read().decode("utf-8")
        return response_body



if __name__ == "__main__":
    app.run(debug=True)