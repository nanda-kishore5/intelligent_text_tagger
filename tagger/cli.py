import argparse
import os
from .tagger import IntelligentTagger

def suggest(args):
    tagger = IntelligentTagger()
    files = tagger.ingest_folder(args.data_dir)
    tagger.fit_tfidf()
    results = tagger.suggest_all(top_k=args.top_k)
    for fname, tags in results.items():
        print(f"\n{fname}")
        for tag, score in tags:
            print(f"  {tag}  ({score:.4f})")

def feedback(args):
    tagger = IntelligentTagger()
    _ = tagger.ingest_folder(args.data_dir) if args.data_dir else None
    w = tagger.apply_feedback(args.doc, args.tag, approve=(args.action == "approve"))
    print(f"Updated weight for tag '{args.tag}': {w:.3f}")

def show_weights(args):
    tagger = IntelligentTagger()
    weights = tagger.get_weights()
    if not weights:
        print("No weights yet.")
        return
    for tag, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"{tag}: {w:.3f}")

def main():
    parser = argparse.ArgumentParser(prog="intelligent-tagger")
    sub = parser.add_subparsers(dest="cmd")

    p_suggest = sub.add_parser("suggest")
    p_suggest.add_argument("--data-dir", required=True)
    p_suggest.add_argument("--top-k", type=int, default=5)
    p_suggest.set_defaults(func=suggest)

    p_fb = sub.add_parser("feedback")
    p_fb.add_argument("--data-dir", required=False, default=None, help="Path where the doc is located (optional)")
    p_fb.add_argument("--doc", required=True, help="Filename (e.g., sample1.txt)")
    p_fb.add_argument("--tag", required=True)
    p_fb.add_argument("--action", choices=["approve", "reject"], required=True)
    p_fb.set_defaults(func=feedback)

    p_show = sub.add_parser("show-weights")
    p_show.set_defaults(func=show_weights)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)

if __name__ == "__main__":
    main()
