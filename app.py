from flask import Flask, request, render_template_string
from tagger.tagger import IntelligentTagger
import os

app = Flask(__name__)
DATA_DIR = "data"

T = IntelligentTagger()
T.ingest_folder(DATA_DIR)
T.fit_tfidf()

TEMPLATE = """
<html>
  <head><title>Intelligent Tagger Demo</title></head>
  <body>
    <h1>Documents</h1>
    <ul>
    {% for f in files %}
      <li><a href="/doc/{{f}}">{{f}}</a></li>
    {% endfor %}
    </ul>
    {% if doc %}
    <h2>{{ doc }}</h2>
    <pre>{{ text }}</pre>
    <h3>Suggested tags</h3>
    <form method="post" action="/feedback/{{doc}}">
      {% for tag, score in tags %}
        <div>
          <b>{{ tag }}</b> ({{ "%.3f"|format(score) }})
          <button name="action" value="approve-{{tag}}">Approve</button>
          <button name="action" value="reject-{{tag}}">Reject</button>
        </div>
      {% endfor %}
    </form>
    {% endif %}
  </body>
</html>
"""

@app.route("/")
def index():
    files = T.corpus_filenames
    return render_template_string(TEMPLATE, files=files, doc=None)

@app.route("/doc/<docname>")
def view_doc(docname):
    idx = T.corpus_filenames.index(docname)
    text = T.corpus_texts[idx]
    tags = T.suggest_tags_for_doc(idx, top_k=6)
    return render_template_string(TEMPLATE, files=T.corpus_filenames, doc=docname, text=text, tags=tags)

@app.route("/feedback/<docname>", methods=["POST"])
def feedback(docname):
    action = request.form.get("action")
    if not action:
        return "no action"
    if action.startswith("approve-"):
        tag = action.replace("approve-", "")
        T.apply_feedback(docname, tag, approve=True)
    else:
        tag = action.replace("reject-", "")
        T.apply_feedback(docname, tag, approve=False)
    return ("", 302, {"Location": f"/doc/{docname}"})

if __name__ == "__main__":
    app.run(debug=True, port=5001)

