from datasets import load_dataset

# z_scrolls_datasets = ["gov_report", "summ_screen_fd", "qmsum","squality","qasper", "narrative_qa", "quality","musique","space_digest", "book_sum_sort"]

narrative_qa = load_dataset("tau/zero_scrolls", "narrative_qa")

print(narrative_qa)
print(narrative_qa["test"])
for e in narrative_qa["test"][:3]:
    print(e)
