"""Deterministic scoring of supplied predictions against synthetic gold labels."""
import argparse
import json
from pathlib import Path
from core import parse_chat, validate_analysis, analyze

ROOT = Path(__file__).resolve().parent / 'fixtures' / 'eval'


def signature(event):
    ids = event.get('evidence_ids') or [x['message_id'] for x in event['evidence']]
    return event['category'], frozenset(ids)


def score(expected, predicted):
    gold, found = set(map(signature, expected)), set(map(signature, predicted))
    tp = len(gold & found)
    precision = tp / len(found) if found else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'expected': len(gold), 'predicted': len(found), 'matched': tp, 'precision': round(precision, 3), 'recall': round(recall, 3), 'f1': round(f1, 3)}


def evaluate(predictions):
    results = {}
    for path in sorted(ROOT.glob('*.json')):
        case = json.loads(path.read_text())
        if path.stem not in predictions:
            continue
        messages = parse_chat(json.dumps(case['chat'], ensure_ascii=False))
        checked = validate_analysis({'events': predictions[path.stem]}, messages)
        results[path.stem] = {**score(case['expected'], checked['events']), 'rejected_invalid_evidence': checked['rejected']}
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score model predictions for synthetic cases; no API calls.')
    parser.add_argument('predictions', nargs='?', help='JSON mapping case name to model events array')
    parser.add_argument('--run-model', action='store_true', help='Call configured LLM for synthetic cases only')
    args = parser.parse_args()
    if args.run_model and args.predictions:
        parser.error('Choose a prediction file or --run-model.')
    if not args.run_model and not args.predictions:
        parser.error('Provide predictions JSON or --run-model.')
    if args.run_model:
        predictions = {}
        for path in sorted(ROOT.glob('*.json')):
            case = json.loads(path.read_text())
            messages = parse_chat(json.dumps(case['chat'], ensure_ascii=False))
            predictions[path.stem] = analyze(messages)['events']
    else:
        predictions = json.loads(Path(args.predictions).read_text())
    results = evaluate(predictions)
    if not results:
        parser.error('No matching case names. Expected: ' + ', '.join(p.stem for p in ROOT.glob('*.json')))
    print(json.dumps(results, ensure_ascii=False, indent=2))
