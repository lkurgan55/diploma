from datasets import load_dataset

# spider validation dataset
dataset = load_dataset("spider", split="validation")
dataset.save_to_disk("./datasets/spider_validation")
print("✅ Датасет збережено у ./datasets/spider_validation")

dataset = load_dataset("birdsql/bird_mini_dev", split="mini_dev_pg")
dataset.save_to_disk("./datasets/bird_mini_dev")
print("✅ Датасет збережено у ./birdsql/bird_mini_dev")