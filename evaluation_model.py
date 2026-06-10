import argparse
import pandas as pd
import editdistance
from transformers import T5ForConditionalGeneration, AutoTokenizer
from utils.data_processing_glaff import DataProcessingGlaff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_data", type=str, required=True, help="Path to eval CSV")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    args = parser.parse_args()
    df = pd.read_csv(args.eval_data)

    grapheme = [g for g in df["word"]]
    phoneme = [p for p in df["phoneme"]]

    MODELS = {
        "french_g2p_model_final": {
            "model_path": "/Users/jungmin/Desktop/Seq2Seq project internship/my_g2p_model_final",
            "prefix": "",
        },
        "byt5-multilingual (baseline)": {
            "model_path": "fdemelo/g2p-multilingual-byt5-tiny-8l-ipa-childes",
            "prefix": "<fr-na>: ",
        },
        "byt5-base": {
            "model_path": "google/byt5-small",
            "prefix": "",
        },
    }


    def run_inference(model, tokenizer, grapheme_list, prefix):
        results = []
        for word in grapheme_list:
            model_inputs = tokenizer(
                f"{prefix}{word}",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            preds = model.generate(**model_inputs, num_beams=1, max_length=512)
            phones = tokenizer.batch_decode(preds, skip_special_tokens=True)
            results.append(phones[0])
        return results


    processor = DataProcessingGlaff()
    results_summary = {}

    for name, cfg in MODELS.items():
        print(f"\nEvaluating: {name}")
        model = T5ForConditionalGeneration.from_pretrained(cfg["model_path"]).to("cpu")
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])

        predictions = run_inference(model, tokenizer, grapheme, cfg["prefix"])
        PER = processor.evaluation(zip(phoneme, predictions))
        results_summary[name] = PER
        print(f"  PER: {PER:.4f}")

        del model, tokenizer 

    print("\n" + "=" * 50)
    print(f"{'Model':<40} {'PER':>8}")
    print("=" * 50)
    for name, per in results_summary.items():
        print(f"{name:<40} {per:>8.4f}")
    print("=" * 50)
    best = min(results_summary, key=results_summary.get)
    print(f"Best model: {best} (PER = {results_summary[best]:.4f})")

if __name__ == "__main__":
    main()
