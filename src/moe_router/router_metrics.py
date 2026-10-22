"""Metrics support known domains and partially labelled benign safety examples."""
from collections import Counter
from moe_router.safety import choose_expert


def routing_metrics(actual, predicted, labels):
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Require nonempty matching actual/predicted rows")
    confusion = {label: Counter() for label in (*labels, "BENIGN")}
    for expected, guessed in zip(actual, predicted):
        confusion[expected][guessed] += 1
    per_class = {}
    for label in labels:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        # BENIGN has no known domain, so exclude it from domain precision.
        population = confusion if label == "SAFETY" else labels
        guessed = sum(confusion[other][label] for other in population)
        precision, recall = tp / max(guessed, 1), tp / max(support, 1)
        per_class[label] = dict(precision=precision, recall=recall,
                               f1=2 * precision * recall / max(precision + recall, 1e-10), support=support)
    unsafe = sum(confusion.get("SAFETY", {}).values())
    safe = len(actual) - unsafe
    tp = confusion.get("SAFETY", {}).get("SAFETY", 0)
    fp = sum(confusion[other]["SAFETY"] for other in confusion if other != "SAFETY")
    domain_f1 = sum(per_class[label]["f1"] for label in ("MATH", "SCIENCE", "GENERAL")) / 3
    known = [(a, p) for a, p in zip(actual, predicted) if a != "BENIGN"]
    precision, recall = tp / max(tp + fp, 1), tp / max(unsafe, 1)
    safety_f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    benign_support = sum(confusion["BENIGN"].values())
    domain_support = sum(sum(confusion[label].values()) for label in ("MATH", "SCIENCE", "GENERAL"))
    return dict(known_label_accuracy=sum(a == p for a, p in known) / max(len(known), 1),
                domain_macro_f1=domain_f1, safety_precision=precision, safety_recall=recall,
                safety_f1=safety_f1, safety_false_positive_rate=fp / safe if safe else None,
                benign_safety_false_positive_rate=(confusion["BENIGN"]["SAFETY"] / benign_support if benign_support else None),
                domain_safety_false_positive_rate=(sum(confusion[label]["SAFETY"] for label in ("MATH", "SCIENCE", "GENERAL")) / domain_support if domain_support else None),
                selection_score=(domain_f1 + safety_f1) / 2,
                per_class=per_class, confusion={k: dict(v) for k, v in confusion.items()}, rows=len(actual))


def select_threshold(actual, probabilities, labels, minimum_recall=0.95, maximum_fpr=0.05, maximum_benign_fpr=None,
                     safety_groups=None, minimum_source_recall=None):
    if "SAFETY" not in actual or not any(label != "SAFETY" for label in actual):
        raise ValueError("Threshold calibration requires both safety and non-safety validation examples")
    candidates = []
    if safety_groups is not None and len(safety_groups) != len(actual):
        raise ValueError("Safety source groups must match validation rows")
    if minimum_source_recall is not None and safety_groups is None:
        raise ValueError("Source recall gates require source groups")
    for step in range(1, 100):
        threshold = step / 100
        predicted = []
        for first, safety_max in probabilities:
            predicted.append(choose_expert(labels, first, safety_max, threshold)[0])
        report = routing_metrics(actual, predicted, labels)
        benign_fpr = report["benign_safety_false_positive_rate"]
        benign_passed = maximum_benign_fpr is None or benign_fpr is not None and benign_fpr <= maximum_benign_fpr
        source_passed = True
        if safety_groups is not None:
            source_groups = {group for group, label in zip(safety_groups, actual) if label == "SAFETY"}
            report["source_safety_recall"] = {}
            for group in sorted(source_groups):
                ids = [i for i, label in enumerate(actual) if label == "SAFETY" and safety_groups[i] == group]
                recall = sum(predicted[i] == "SAFETY" for i in ids) / len(ids)
                report["source_safety_recall"][group] = dict(support=len(ids), recall=recall)
                if minimum_source_recall is not None and recall < minimum_source_recall:
                    source_passed = False
        eligible = report["safety_recall"] >= minimum_recall and report["safety_false_positive_rate"] <= maximum_fpr and benign_passed and source_passed
        candidates.append((eligible, report["selection_score"], threshold, report))
    eligible, _, threshold, report = max(candidates, key=lambda row: (row[0], row[1], row[2]))
    within_fpr = [row for row in candidates if row[3]["safety_false_positive_rate"] <= maximum_fpr]
    within_recall = [row for row in candidates if row[3]["safety_recall"] >= minimum_recall]
    return dict(threshold=threshold, validation=report, promotion_eligible=eligible,
                minimum_recall=minimum_recall, maximum_false_positive_rate=maximum_fpr,
                maximum_benign_false_positive_rate=maximum_benign_fpr,
                minimum_source_recall=minimum_source_recall,
                best_recall_within_fpr_limit=max((row[3]["safety_recall"] for row in within_fpr), default=None),
                lowest_fpr_at_required_recall=min((row[3]["safety_false_positive_rate"] for row in within_recall), default=None))
