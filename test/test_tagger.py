
import os
from tagger.tagger import IntelligentTagger

def test_ingest_and_suggest(tmp_path):
    # create two small files
    p = tmp_path / "data"
    p.mkdir()
    f1 = p / "a.txt"
    f1.write_text("Machine learning model for churn prediction. Features: age, usage, billing.")
    f2 = p / "b.txt"
    f2.write_text("Meeting notes: discuss model deployment and monitoring metrics.")
    tg = IntelligentTagger(feedback_path=os.path.join(tmp_path, "feedback.json"))
    files = tg.ingest_folder(str(p))
    assert set(files) == {"a.txt", "b.txt"}
    tg.fit_tfidf()
    results = tg.suggest_all(top_k=3)
    assert "a.txt" in results and "b.txt" in results
    # suggestions should be non-empty lists
    assert len(results["a.txt"]) > 0
    assert len(results["b.txt"]) > 0

def test_feedback_adjusts_weights(tmp_path):
    tg = IntelligentTagger(feedback_path=os.path.join(tmp_path, "feedback.json"))
    w1 = tg.apply_feedback("doc1", "churn", approve=True)
    assert w1 > 0
    w2 = tg.apply_feedback("doc1", "churn", approve=False)
    assert w2 < w1
