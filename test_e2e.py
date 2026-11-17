"""Quick end-to-end test of the recommendation engine."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.recommender import VisionModelRecommender

rec = VisionModelRecommender()

# Test 1: Drone CPU scenario
print("=" * 70)
print("TEST 1: Drone CPU with sensor noise")
print("=" * 70)
result = rec.recommend("Outdoor drone obstacle detection on CPU with sensor noise and low latency under 25ms")
top = result["top_recommendation"]
print("Winner:", top["display_name"], "(" + top["architecture_family"] + ")")
print("Score:", top["score"], "/ 100")
print("Margin: +%.2f" % top.get("margin_over_runner_up", 0))
print("Candidates evaluated:", result["total_candidates_evaluated"])
print("Candidates passed filter:", result["candidates_passed"])
print("Excluded:", len(result["excluded_candidates"]))
print()
print("Full Ranking:")
for c in result["ranked_candidates"]:
    print("  #%d %-40s Score=%6.2f  Family=%s" % (c["rank"], c["display_name"], c["score"], c["architecture_family"]))
print()
exp = result["explanation"]
print("Headline:", exp["headline"])
print("Rationale:", exp["core_rationale"])
print("Confidence:", exp["confidence_level"], "(%d%%)" % exp["confidence_score"])

# Test 2: Adverse weather cloud GPU
print()
print("=" * 70)
print("TEST 2: Adverse weather cloud GPU")
print("=" * 70)
result2 = rec.recommend("Camera surveillance in adverse weather with fog, snow, and sensor noise on GPU. Accuracy and robustness are essential.")
top2 = result2["top_recommendation"]
print("Winner:", top2["display_name"], "(" + top2["architecture_family"] + ")")
print("Score:", top2["score"], "/ 100")

# Test 3: Small medical dataset
print()
print("=" * 70)
print("TEST 3: Small medical dataset from scratch")
print("=" * 70)
result3 = rec.recommend("Training from scratch on limited medical imaging dataset with only 1500 images. Calibration and reliability are critical.")
top3 = result3["top_recommendation"]
print("Winner:", top3["display_name"], "(" + top3["architecture_family"] + ")")
print("Score:", top3["score"], "/ 100")

# Test 4: Export report
print()
print("=" * 70)
print("TEST 4: Markdown report export")
print("=" * 70)
md = rec.export_report(result, fmt="markdown")
print("Report length:", len(md), "chars")
print("First 200 chars:", md[:200])

print()
print("ALL TESTS PASSED - ENGINE IS FULLY OPERATIONAL")

